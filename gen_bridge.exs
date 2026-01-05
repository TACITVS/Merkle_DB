defmodule BridgeGeneratorV7 do
  @include_dir "native/fp_lib/include"
  @c_nif_out "native/generated_nif.c"
  @ex_module_out "lib/merkle_db/asm.ex"

  @allowed_headers [
    "fp_core.h", "fp_stats.h", "fp_linear_regression.h", "fp_monads.h", 
    "fp_compose.h", "fp_3d_math_wrapper.h", "fp_gpu_math.h", "fp_math.h",
    "fp_pca.h", "fp_kmeans.h", "fp_naive_bayes.h", "fp_neural_network.h"
  ]

  @destructors %{
    "KMeansResult" => "fp_kmeans_free",
    "PCAModel" => "fp_pca_free_model",
    "PCAResult" => "fp_pca_free_result_internal",
    "GaussianNBModel" => "fp_nb_free_gaussian_model",
    "MultinomialNBModel" => "fp_nb_free_multinomial_model",
    "NeuralNetwork" => "fp_neural_network_free",
    "TrainingResult" => "fp_training_result_free"
  }

  @struct_fields %{
    "KMeansResult" => [
      {:centroids, "double*", :binary},
      {:assignments, "int*", :binary},
      {:inertia, "double", :scalar},
      {:converged, "int", :scalar}
    ],
    "PCAModel" => [
      {:n_components, "int", :scalar},
      {:eigenvalues, "double*", :binary},
      {:total_variance, "double", :scalar}
    ],
    "PCAResult" => [
      {:converged, "int", :scalar}
    ]
  }

  # Functions that require dirty scheduler (CPU-bound, >1ms execution)
  @dirty_functions [
    "fp_kmeans_f64",
    "fp_pca_fit",
    "fp_pca_generate_ellipse_data",
    "fp_pca_generate_low_rank_data",
    "fp_neural_network_train",
    "fp_neural_network_create",
    "fp_gaussian_nb_train",
    "fp_multinomial_nb_train"
  ]

  @extra_nif_entries [
    {"fp_job_start", 3, "nif_fp_job_start", 0},
    {"fp_job_status", 1, "nif_fp_job_status", 0},
    {"fp_job_result", 1, "nif_fp_job_result", 0},
    {"fp_job_cancel", 1, "nif_fp_job_cancel", 0},
    {"fp_query_gemv_columnar", 4, "nif_fp_query_gemv_columnar", "ERL_NIF_DIRTY_JOB_CPU_BOUND"},
    {"fp_query_gemv_indexed", 5, "nif_fp_query_gemv_indexed", "ERL_NIF_DIRTY_JOB_CPU_BOUND"},
    {"fp_query_topk", 4, "nif_fp_query_topk", 0},
    {"fp_quantize_f64_to_u8", 3, "nif_fp_quantize_f64_to_u8", "ERL_NIF_DIRTY_JOB_CPU_BOUND"},
    {"fp_query_gemv_quantized", 5, "nif_fp_query_gemv_quantized", "ERL_NIF_DIRTY_JOB_CPU_BOUND"},
    {"fp_sparse_dotp", 4, "nif_fp_sparse_dotp", 0},
    {"fp_hnsw_create", 4, "nif_fp_hnsw_create", 0},
    {"fp_hnsw_insert", 5, "nif_fp_hnsw_insert", "ERL_NIF_DIRTY_JOB_CPU_BOUND"},
    {"fp_hnsw_search", 6, "nif_fp_hnsw_search", "ERL_NIF_DIRTY_JOB_CPU_BOUND"},
    {"fp_pca_transform_result", 3, "nif_fp_pca_transform_result", "ERL_NIF_DIRTY_JOB_CPU_BOUND"},
    {"fp_pca_result_n_components", 1, "nif_fp_pca_result_n_components", 0},
    {"fp_pca_result_total_variance", 1, "nif_fp_pca_result_total_variance", 0},
    {"fp_pca_result_explained_variance", 2, "nif_fp_pca_result_explained_variance", 0},
    {"fp_pca_result_cumulative_variance", 2, "nif_fp_pca_result_cumulative_variance", 0},
    {"fp_blake3_hash", 1, "nif_fp_blake3_hash", 0}
  ]

  @extra_elixir_defs [
    {"fp_job_start", ["op", "args", "opts"]},
    {"fp_job_status", ["job"]},
    {"fp_job_result", ["job"]},
    {"fp_job_cancel", ["job"]},
    {"fp_quantize_f64_to_u8", ["in_bin", "min_val", "inv_scale"]},
    {"fp_query_gemv_quantized", ["columns_tuple", "query_bin", "bias", "count", "dim"]},
    {"fp_sparse_dotp", ["indices_a", "values_a", "indices_b", "values_b"]},
    {"fp_hnsw_create", ["dim", "m", "ef_construction", "capacity"]},
    {"fp_hnsw_insert", ["hnsw_res", "vector_idx", "vec_bin", "columns_tuple", "count"]},
    {"fp_hnsw_search", ["hnsw_res", "query_bin", "k", "ef_search", "columns_tuple", "count"]},
    {"fp_pca_transform_result", ["pca_result", "data_bin", "count"]},
    {"fp_pca_result_n_components", ["pca_result"]},
    {"fp_pca_result_total_variance", ["pca_result"]},
    {"fp_pca_result_explained_variance", ["pca_result", "size"]},
    {"fp_pca_result_cumulative_variance", ["pca_result", "size"]}
  ]

  defp should_use_dirty?(func_name) do
    func_name in @dirty_functions or
    String.contains?(func_name, ["train", "fit", "generate_low_rank", "generate_ellipse"])
  end

  def run do
    IO.puts "--- 🏗️  BRIDGE GENERATOR V7 (ZERO-COPY / ACCESSORS) 🏗️  ---"
    headers = @allowed_headers |> Enum.map(&Path.join(@include_dir, &1)) |> Enum.filter(&File.exists?/1)
    functions = headers |> Enum.flat_map(&parse_header/1) |> Enum.uniq_by(& &1.name) |> Enum.sort_by(& &1.name)
    bridgable_functions = Enum.filter(functions, &supported_signature?/1)
    IO.puts "✅ Found #{length(bridgable_functions)} bridgable functions."
    
    generate_c_nif(bridgable_functions, headers)
    generate_elixir_module(bridgable_functions)
    IO.puts "--- 🚀 BRIDGE V7 COMPLETE 🚀 ---"
  end

  defp parse_header(file) do
    content = File.read!(file)
    regex = ~r/^(?!typedef|struct)\s*(?<ret>[a-zA-Z0-9_*]+)\s+(?<name>fp_[a-zA-Z0-9_]+)\s*\((?<args>[^;{}]*)\)\s*;/m
    Regex.scan(regex, content, capture: :all_but_first)
    |> Enum.map(fn [ret, name, args_str] -> %{name: name, return_type: String.trim(ret), args: parse_args(args_str)} end)
  end

  defp parse_args(args_str) do
    if String.trim(args_str) == "void" or String.trim(args_str) == "" do
      []
    else
      args_str
      |> String.split(",")
      |> Enum.map(&String.trim/1)
      |> Enum.map(fn arg -> 
        is_const = String.contains?(arg, "const"); is_ptr = String.contains?(arg, "*")
        parts = arg |> String.replace("const", "") |> String.replace("*", "") |> String.split()
        type_part = Enum.drop(parts, -1) |> Enum.join(" ") |> String.trim()
        var_name = List.last(parts) |> String.trim()
        %{raw: arg, name: var_name, type: type_part, is_ptr: is_ptr, is_const: is_const}
      end)
    end
  end

  defp supported_signature?(func) do
    supported_scalars = ["void", "bool", "int", "unsigned int", "size_t", "double", "float", "int8_t", "int16_t", "int32_t", "int64_t", "uint8_t", "uint16_t", "uint32_t", "uint64_t"]
    supported_structs = Map.keys(@destructors)
    ret_ok = func.return_type in supported_scalars or func.return_type in supported_structs
    args_ok = Enum.all?(func.args, fn a -> (a.type in supported_scalars or a.type in supported_structs) and not String.contains?(a.raw, "(") and not String.contains?(a.raw, "struct") end)
    ret_ok and args_ok
  end

  defp generate_c_nif(funcs, headers) do
    include_directives = Enum.map_join(headers, "\n", fn h -> "#include \"fp_lib/include/#{Path.basename(h)}\"" end)
    res_decls = Enum.map_join(@destructors, "\n", fn {type, _} -> "ErlNifResourceType* RES_TYPE_#{type};" end)
    extra_decls = Enum.map_join(@extra_nif_entries, "\n", fn {_, _, func, _} ->
      "static ERL_NIF_TERM #{func}(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[]);"
    end)
    destructors_code = Enum.map_join(@destructors, "\n\n", fn {type, free_fn} -> "void dtor_#{type}(ErlNifEnv* env, void* obj) { #{type}* res = (#{type}*)obj; #{free_fn}(res); }" end)
    res_init = Enum.map_join(@destructors, "\n    ", fn {type, _} -> "RES_TYPE_#{type} = enif_open_resource_type(env, NULL, \"#{type}\", dtor_#{type}, ERL_NIF_RT_CREATE | ERL_NIF_RT_TAKEOVER, NULL);" end)

    accessors = Enum.map_join(@struct_fields, "\n", fn {struct_type, fields} -> 
      Enum.map_join(fields, "\n", fn {field_name, field_type, mode} -> 
        generate_accessor_c(struct_type, field_name, field_type, mode)
      end)
    end)

    preamble = """
    #include <stdbool.h>
    #include <string.h>
    #ifdef __GNUC__
      #define _SAVED_GNUC_ __GNUC__
      #undef __GNUC__
    #endif
    #include <erl_nif.h>
    #ifdef _SAVED_GNUC_
      #define __GNUC__ _SAVED_GNUC__
      #undef _SAVED_GNUC_
    #endif
    #{include_directives}
    #{res_decls}
    #{extra_decls}

    // Size validation helpers to prevent OOB access
    static inline int validate_binary_size(ErlNifBinary* bin, size_t elem_size, size_t min_count) {
        if (min_count == 0) return 1;  // Allow empty when 0 count expected
        size_t required = elem_size * min_count;
        return bin->size >= required && (bin->size % elem_size) == 0;
    }

    static void fp_pca_free_result_internal(PCAResult* res) { fp_pca_free_model(&res->model); }
    #{destructors_code}
    #{accessors}
    """
    
    accessor_entries = Enum.flat_map(@struct_fields, fn {struct_type, fields} ->
      Enum.map(fields, fn {field_name, _, _} ->
        "{\"get_#{struct_type}_#{field_name}\", 2, nif_get_#{struct_type}_#{field_name}, 0}"
      end)
    end) |> Enum.join(",\n    ")

    wrappers = Enum.map_join(funcs, "\n\n", &generate_c_wrapper/1)
    entries = Enum.map_join(funcs, ",\n    ", fn f ->
      flags = if should_use_dirty?(f.name), do: "ERL_NIF_DIRTY_JOB_CPU_BOUND", else: "0"
      "{\"#{f.name}\", #{length(f.args)}, nif_#{f.name}, #{flags}}"
    end)
    extra_entries = Enum.map_join(@extra_nif_entries, ",\n    ", fn {name, arity, func, flags} ->
      "{\"#{name}\", #{arity}, #{func}, #{flags}}"
    end)
    all_entries =
      [entries, accessor_entries, extra_entries]
      |> Enum.reject(&(&1 == ""))
      |> Enum.join(",\n    ")
    
    File.write!(@c_nif_out, "// GENERATED V7\n#{preamble}\n#{wrappers}\nstatic ErlNifFunc generated_nif_funcs[] = { \n    #{all_entries} \n}; \nstatic int load_resources(ErlNifEnv* env, void** priv_data, ERL_NIF_TERM load_info) { #{res_init}\nreturn 0; }")
  end

  defp generate_accessor_c(struct_type, field_name, field_type, mode) do
    """
    static ERL_NIF_TERM nif_get_#{struct_type}_#{field_name}(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[]) {
        #{struct_type}* res;
        if (!enif_get_resource(env, argv[0], RES_TYPE_#{struct_type}, (void**)&res)) return enif_make_badarg(env);
        #{case mode do
          :scalar ->
            case field_type do
              "double" -> "return enif_make_double(env, res->#{field_name});"
              "int" -> "return enif_make_int(env, res->#{field_name});"
              _ -> "return enif_make_badarg(env);"
            end
          :binary ->
            "ErlNifUInt64 size; if (!enif_get_uint64(env, argv[1], &size)) return enif_make_badarg(env); ErlNifBinary bin; enif_alloc_binary((size_t)size, &bin); memcpy(bin.data, res->#{field_name}, (size_t)size); return enif_make_binary(env, &bin);"
        end}
    }
    """
  end

  # Get element size for a pointer type
  defp elem_size_for_type(type) do
    case type do
      "double" -> 8
      "float" -> 4
      "int64_t" -> 8
      "uint64_t" -> 8
      "int32_t" -> 4
      "uint32_t" -> 4
      "int" -> 4
      "unsigned int" -> 4
      "int16_t" -> 2
      "uint16_t" -> 2
      "int8_t" -> 1
      "uint8_t" -> 1
      "size_t" -> 8
      "bool" -> 1
      _ -> 1  # Unknown type, assume 1 byte (will still catch gross mismatches)
    end
  end

  # Find the count argument that follows a pointer arg (pattern: const T* data, size_t n)
  defp find_count_arg(args, ptr_idx) do
    # Look for next arg that's a size_t, int, or similar count type
    count_types = ["size_t", "int", "int32_t", "int64_t", "uint64_t"]
    Enum.with_index(args)
    |> Enum.find(fn {arg, i} ->
      i > ptr_idx and arg.type in count_types and not arg.is_ptr
    end)
  end

  defp generate_c_wrapper(func) do
    # Build setup code with size validation for const pointer inputs
    setup = Enum.with_index(func.args) |> Enum.map_join("\n    ", fn {arg, i} ->
      cond do
        arg.type in Map.keys(@destructors) ->
            "#{arg.type}* res_#{arg.name}; if (!enif_get_resource(env, argv[#{i}], RES_TYPE_#{arg.type}, (void**)&res_#{arg.name})) return enif_make_badarg(env);"
        arg.is_ptr ->
            if arg.is_const do
                # For const pointers, add size validation if we can find the count arg
                case find_count_arg(func.args, i) do
                  {count_arg, _} ->
                    # Validate after both binary and count are parsed
                    "ErlNifBinary bin_#{arg.name}; if (!enif_inspect_binary(env, argv[#{i}], &bin_#{arg.name})) return enif_make_badarg(env); #{arg.type}* ptr_#{arg.name} = (#{arg.type}*)bin_#{arg.name}.data; /* validated with val_#{count_arg.name} */"
                  nil ->
                    "ErlNifBinary bin_#{arg.name}; if (!enif_inspect_binary(env, argv[#{i}], &bin_#{arg.name})) return enif_make_badarg(env); #{arg.type}* ptr_#{arg.name} = (#{arg.type}*)bin_#{arg.name}.data;"
                end
            else
                "ErlNifUInt64 size_#{arg.name}; if (!enif_get_uint64(env, argv[#{i}], &size_#{arg.name})) return enif_make_badarg(env); ErlNifBinary out_bin_#{arg.name}; enif_alloc_binary((size_t)size_#{arg.name}, &out_bin_#{arg.name}); #{arg.type}* ptr_#{arg.name} = (#{arg.type}*)out_bin_#{arg.name}.data;"
            end
        true -> parse_scalar(arg.type, "val_#{arg.name}", i)
      end
    end)

    # Generate validation checks after all args are parsed
    # Skip resource types (they use res_ prefix, not bin_)
    validations = Enum.with_index(func.args)
    |> Enum.filter(fn {arg, _} -> arg.is_ptr and arg.is_const and not (arg.type in Map.keys(@destructors)) end)
    |> Enum.map(fn {arg, i} ->
      elem_size = elem_size_for_type(arg.type)
      case find_count_arg(func.args, i) do
        {count_arg, _} ->
          "if (!validate_binary_size(&bin_#{arg.name}, #{elem_size}, (size_t)val_#{count_arg.name})) return enif_make_badarg(env);"
        nil -> nil
      end
    end)
    |> Enum.reject(&is_nil/1)
    |> Enum.join("\n    ")

    call_args = Enum.map_join(func.args, ", ", fn arg -> 
        cond do
            arg.type in Map.keys(@destructors) -> (if arg.is_ptr, do: "res_#{arg.name}", else: "*res_#{arg.name}")
            arg.is_ptr -> "ptr_#{arg.name}"
            true -> "val_#{arg.name}"
        end
    end)
    
    body = if func.return_type in Map.keys(@destructors) do
      "#{func.return_type} res = #{func.name}(#{call_args}); #{func.return_type}* res_ptr = enif_alloc_resource(RES_TYPE_#{func.return_type}, sizeof(#{func.return_type})); *res_ptr = res; ERL_NIF_TERM ret_res = enif_make_resource(env, res_ptr); enif_release_resource(res_ptr);"
    else
      if func.return_type == "void" do
        "#{func.name}(#{call_args});"
      else
        "#{func.return_type} res = #{func.name}(#{call_args});"
      end
    end

    return_stmt = construct_return_v3(func.return_type, Enum.filter(func.args, fn a -> a.is_ptr and not a.is_const and not (a.type in Map.keys(@destructors)) end))

    # Include validation block if we have validations
    validation_block = if validations != "", do: "\n    #{validations}", else: ""

    "static ERL_NIF_TERM nif_#{func.name}(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[]) {\n    #{setup}#{validation_block}\n    #{body}\n    #{return_stmt}\n}"
  end

  defp parse_scalar(type, var, i) do
    "#{type} #{var}; " <> case type do
      "int64_t" -> "if (!enif_get_int64(env, argv[#{i}], (ErlNifSInt64*)&#{var})) return enif_make_badarg(env);"
      "uint64_t" -> "if (!enif_get_uint64(env, argv[#{i}], (ErlNifUInt64*)&#{var})) return enif_make_badarg(env);"
      "int32_t" -> "if (!enif_get_int(env, argv[#{i}], (int*)&#{var})) return enif_make_badarg(env);"
      "int" -> "if (!enif_get_int(env, argv[#{i}], (int*)&#{var})) return enif_make_badarg(env);"
      "uint32_t" -> "if (!enif_get_uint(env, argv[#{i}], (unsigned int*)&#{var})) return enif_make_badarg(env);"
      "unsigned int" -> "if (!enif_get_uint(env, argv[#{i}], (unsigned int*)&#{var})) return enif_make_badarg(env);"
      "size_t" -> "if (!enif_get_uint64(env, argv[#{i}], (ErlNifUInt64*)&#{var})) return enif_make_badarg(env);"
      "double" -> "if (!enif_get_double(env, argv[#{i}], &#{var})) return enif_make_badarg(env);"
      "float" -> "double tmp_#{i}; if (!enif_get_double(env, argv[#{i}], &tmp_#{i})) return enif_make_badarg(env); #{var} = (float)tmp_#{i};"
      "bool" -> "char atom_#{i}[6]; if(enif_get_atom(env, argv[#{i}], atom_#{i}, 6, ERL_NIF_LATIN1)) #{var} = (strcmp(atom_#{i}, \"true\") == 0); else #{var} = 0;"
      "int16_t" -> "int tmp_#{i}; if (!enif_get_int(env, argv[#{i}], &tmp_#{i})) return enif_make_badarg(env); #{var} = (int16_t)tmp_#{i};"
      "uint16_t" -> "unsigned int tmp_#{i}; if (!enif_get_uint(env, argv[#{i}], &tmp_#{i})) return enif_make_badarg(env); #{var} = (uint16_t)tmp_#{i};"
      "int8_t" -> "int tmp_#{i}; if (!enif_get_int(env, argv[#{i}], &tmp_#{i})) return enif_make_badarg(env); #{var} = (int8_t)tmp_#{i};"
      "uint8_t" -> "unsigned int tmp_#{i}; if (!enif_get_uint(env, argv[#{i}], &tmp_#{i})) return enif_make_badarg(env); #{var} = (uint8_t)tmp_#{i};"
      _ -> "return enif_make_badarg(env);"
    end
  end

  defp construct_return_v3(ret_type, outputs) do
    if ret_type in Map.keys(@destructors) do
      out_terms = Enum.map(outputs, fn arg -> "enif_make_binary(env, &out_bin_#{arg.name})" end)
      if length(out_terms) == 0, do: "return ret_res;", else: "return enif_make_tuple#{length(out_terms) + 1}(env, #{Enum.join(["ret_res" | out_terms], ", ")});"
    else
      ret_term = if ret_type == "void", do: nil, else: box_return(ret_type)
      out_terms = Enum.map(outputs, fn arg -> "enif_make_binary(env, &out_bin_#{arg.name})" end)
      all_terms = if ret_term, do: [ret_term | out_terms], else: out_terms
      case length(all_terms) do
        0 -> "return enif_make_atom(env, \"ok\");"
        1 -> "return #{List.first(all_terms)};"
        n -> "return enif_make_tuple#{n}(env, #{Enum.join(all_terms, ", ")});"
      end
    end
  end

  defp box_return("int64_t"), do: "enif_make_int64(env, res)"
  defp box_return("uint64_t"), do: "enif_make_uint64(env, res)"
  defp box_return("int32_t"), do: "enif_make_int(env, res)"
  defp box_return("int"), do: "enif_make_int(env, res)"
  defp box_return("uint32_t"), do: "enif_make_uint(env, res)"
  defp box_return("unsigned int"), do: "enif_make_uint(env, res)"
  defp box_return("size_t"), do: "enif_make_uint64(env, res)"
  defp box_return("double"), do: "enif_make_double(env, res)"
  defp box_return("float"), do: "enif_make_double(env, (double)res)"
  defp box_return("bool"), do: "res ? enif_make_atom(env, \"true\") : enif_make_atom(env, \"false\")"
  # Small integer types - cast to appropriate Erlang term type
  defp box_return("int8_t"), do: "enif_make_int(env, (int)res)"
  defp box_return("uint8_t"), do: "enif_make_uint(env, (unsigned int)res)"
  defp box_return("int16_t"), do: "enif_make_int(env, (int)res)"
  defp box_return("uint16_t"), do: "enif_make_uint(env, (unsigned int)res)"
  # Catch-all should fail loudly rather than silently return 0
  defp box_return(unknown_type), do: raise "BUG: Unsupported return type '#{unknown_type}' in box_return/1"

  defp generate_elixir_module(funcs) do
    reserved = ["end", "fn", "do", "in", "true", "false", "nil", "after", "catch", "else", "rescue", "quote", "unquote"]
    defs = Enum.map_join(funcs, "\n\n", fn f ->
      args = Enum.map_join(f.args, ", ", fn a ->
        name = if a.is_ptr and not a.is_const and not (a.type in Map.keys(@destructors)), do: "size_#{a.name}", else: "#{a.name}"
        name = if name in reserved, do: "#{name}_", else: name
        # Prefix with underscore since NIF stubs don't use the params
        "_#{name}"
      end)
      "@doc \"Calls C function: #{f.name}\"\ndef #{f.name}(#{args}), do: :erlang.nif_error(:nif_not_loaded)"
    end)

    extra_defs = Enum.map_join(@extra_elixir_defs, "\n", fn {name, args} ->
      args = Enum.map_join(args, ", ", fn arg -> "_#{arg}" end)
      "@doc \"Calls C function: #{name}\"\ndef #{name}(#{args}), do: :erlang.nif_error(:nif_not_loaded)"
    end)

    accessor_defs = Enum.map_join(@struct_fields, "\n", fn {struct_type, fields} ->
      Enum.map_join(fields, "\n", fn {field_name, _, _} ->
        "def get_#{struct_type}_#{field_name}(_res, _size \\\\ 0), do: :erlang.nif_error(:nif_not_loaded)"
      end)
    end)

    # Query functions (manually maintained, not auto-generated from headers)
    query_functions = """

# --- Query Kernels (manually added) ---

@doc \"\"\"
Single-pass columnar GEMV for vector similarity search.
Replaces the 64-allocation loop with one NIF call.

columns_tuple: tuple of dim binaries (each count*8 bytes)
query_bin: normalized query vector (dim*8 bytes)
count: number of vectors
dim: dimension

Returns: scores binary (count*8 bytes)
\"\"\"
def fp_query_gemv_columnar(_columns_tuple, _query_bin, _count, _dim), do: :erlang.nif_error(:nif_not_loaded)

@doc \"\"\"
Indexed columnar GEMV for IVF search.
Only computes scores for specified row indices - O(num_indices * dim) instead of O(count * dim).

columns_tuple: tuple of dim column binaries
query_bin: normalized query vector (dim*8 bytes)
indices_bin: int32 array of candidate row indices
count: total number of vectors
dim: vector dimension

Returns: scores binary (num_indices*8 bytes, in same order as indices)
\"\"\"
def fp_query_gemv_indexed(_columns_tuple, _query_bin, _indices_bin, _count, _dim), do: :erlang.nif_error(:nif_not_loaded)

@doc \"\"\"
Top-K selection from scores array.

scores_bin: scores binary (count*8 bytes)
count: number of scores
k: number of top results
threshold: minimum score threshold

Returns: {result_count, indices_bin, scores_bin}
  - indices_bin: int32 indices of top results
  - scores_bin: float64 scores of top results
\"\"\"
def fp_query_topk(_scores_bin, _count, _k, _threshold), do: :erlang.nif_error(:nif_not_loaded)

# --- BLAKE3 Cryptographic Hash ---

@doc \"\"\"
BLAKE3 cryptographic hash (3-5x faster than SHA-256).

input: binary data to hash

Returns: 32-byte hash binary
\"\"\"
def fp_blake3_hash(_input), do: :erlang.nif_error(:nif_not_loaded)
"""

    File.write!(@ex_module_out, "defmodule MerkleDb.ASM do\n  @on_load :load_nifs\n  def load_nifs do\n    priv_dir =\n      case :code.priv_dir(:merkle_db) do\n        {:error, _} ->\n          Path.expand(\"../../priv\", __DIR__)\n        dir ->\n          List.to_string(dir)\n      end\n\n    path = Path.join(priv_dir, \"merkle_nif\") |> String.to_charlist()\n\n    case :erlang.load_nif(path, 0) do\n      :ok -> :ok\n      {:error, {:already_loaded, _}} -> :ok\n      {:error, reason} -> {:error, reason}\n    end\n  end\n#{defs}\n\n#{extra_defs}\n\n# --- Struct Accessors ---\n#{accessor_defs}#{query_functions}\nend")
  end
end

BridgeGeneratorV7.run()
