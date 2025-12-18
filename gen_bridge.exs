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
    static void fp_pca_free_result_internal(PCAResult* res) { fp_pca_free_model(&res->model); }
    #{destructors_code}
    #{accessors}
    """
    
    accessor_entries = Enum.flat_map(@struct_fields, fn {struct_type, fields} -> 
      Enum.map(fields, fn {field_name, _, _} -> 
        "{\"get_#{struct_type}_#{field_name}\", 2, nif_get_#{struct_type}_#{field_name}}"
      end)
    end) |> Enum.join(",\n    ")

    wrappers = Enum.map_join(funcs, "\n\n", &generate_c_wrapper/1)
    entries = Enum.map_join(funcs, ",\n    ", fn f -> "{\"#{f.name}\", #{length(f.args)}, nif_#{f.name}}" end)
    
    File.write!(@c_nif_out, "// GENERATED V7\n#{preamble}\n#{wrappers}\nstatic ErlNifFunc generated_nif_funcs[] = { \n    #{entries},\n    #{accessor_entries} \n}; \nstatic int load_resources(ErlNifEnv* env, void** priv_data, ERL_NIF_TERM load_info) { #{res_init}\nreturn 0; }")
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
    ""
  end

  defp generate_c_wrapper(func) do
    setup = Enum.with_index(func.args) |> Enum.map_join("\n    ", fn {arg, i} ->
      cond do
        arg.type in Map.keys(@destructors) -> 
            "#{arg.type}* res_#{arg.name}; if (!enif_get_resource(env, argv[#{i}], RES_TYPE_#{arg.type}, (void**)&res_#{arg.name})) return enif_make_badarg(env);"
        arg.is_ptr -> 
            if arg.is_const do
                "ErlNifBinary bin_#{arg.name}; if (!enif_inspect_binary(env, argv[#{i}], &bin_#{arg.name})) return enif_make_badarg(env); #{arg.type}* ptr_#{arg.name} = (#{arg.type}*)bin_#{arg.name}.data;"
            else
                "ErlNifUInt64 size_#{arg.name}; if (!enif_get_uint64(env, argv[#{i}], &size_#{arg.name})) return enif_make_badarg(env); ErlNifBinary out_bin_#{arg.name}; enif_alloc_binary((size_t)size_#{arg.name}, &out_bin_#{arg.name}); #{arg.type}* ptr_#{arg.name} = (#{arg.type}*)out_bin_#{arg.name}.data;"
            end
        true -> parse_scalar(arg.type, "val_#{arg.name}", i)
      end
    end)

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

    "static ERL_NIF_TERM nif_#{func.name}(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[]) {\n    #{setup}\n    #{body}\n    #{return_stmt}\n}"
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
  defp box_return(_), do: "enif_make_int(env, 0)"

  defp generate_elixir_module(funcs) do
    reserved = ["end", "fn", "do", "in", "true", "false", "nil", "after", "catch", "else", "rescue", "quote", "unquote"]
    defs = Enum.map_join(funcs, "\n\n", fn f ->
      args = Enum.map_join(f.args, ", ", fn a -> 
        name = if a.is_ptr and not a.is_const and not (a.type in Map.keys(@destructors)), do: "size_#{a.name}", else: "#{a.name}"
        if name in reserved, do: "#{name}_", else: name
      end)
      "@doc \"Calls C function: #{f.name}\"\ndef #{f.name}(#{args}), do: :erlang.nif_error(:nif_not_loaded)"
    end)

    accessor_defs = Enum.map_join(@struct_fields, "\n", fn {struct_type, fields} -> 
      Enum.map_join(fields, "\n", fn {field_name, _, _} -> 
        "def get_#{struct_type}_#{field_name}(res, size \\ 0), do: :erlang.nif_error(:nif_not_loaded)"
      end)
    end)

    File.write!(@ex_module_out, "defmodule MerkleDb.ASM do\n  @on_load :load_nifs\n  def load_nifs do\n    path = :code.priv_dir(:merkle_db) |> Path.join(\"merkle_nif\") |> String.to_charlist()\n    :erlang.load_nif(path, 0)\n  end\n#{defs}\n\n# --- Struct Accessors ---\n#{accessor_defs}\nend")
  end
end

BridgeGeneratorV7.run()
