defmodule NifGenerator do
  def generate do
    kernels = Code.eval_file("kernels.exs") |> elem(0)
    
    # 1. Generate Extern Declarations
    externs = Enum.map(kernels, fn {_, c_name} ->
      "extern int64_t #{c_name}(const int64_t* in, size_t n);"
    end) |> Enum.join("\n")

    # 2. Generate the Kernel Table
    table_entries = Enum.map(kernels, fn {atom, c_name} ->
      "    {\"#{atom}\", #{c_name}},"
    end) |> Enum.join("\n")

    # 3. The Template
    c_code = """
    // -----------------------------------------------------------------------------
    // AUTOMATICALLY GENERATED KERNEL REGISTRY
    // DO NOT EDIT MANUALLY - RUN gen_nif.exs
    // -----------------------------------------------------------------------------
    
    // Function Signature
    typedef int64_t (*KernelFunc)(const int64_t* data, size_t length);

    // Externs
    #{externs}

    // Registry Table
    typedef struct { const char* name; KernelFunc func; } KernelEntry;
    
    static KernelEntry KERNEL_TABLE[] = {
    #{table_entries}
        {NULL, NULL}
    };

    // Lookup Helper
    KernelFunc get_kernel(const char* name) {
        for (int i = 0; KERNEL_TABLE[i].name != NULL; i++) {
            if (strcmp(KERNEL_TABLE[i].name, name) == 0) return KERNEL_TABLE[i].func;
        }
        return NULL;
    }
    // -----------------------------------------------------------------------------
    // END GENERATED CODE
    // -----------------------------------------------------------------------------
    """
    
    IO.puts c_code
  end
end

NifGenerator.generate()