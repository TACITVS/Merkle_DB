# Dialyzer Warnings to Ignore
# Add patterns here to suppress specific warnings

[
  # Example: Ignore warnings about specific functions
  # {:warn_unknown, "lib/merkle_db/some_module.ex:123"},

  # NIF functions may show warnings because the Erlang side can't see the native implementation
  ~r/lib\/merkle_db\/asm.ex.*Function .* has no local return/,

  # Ra library may have some type spec issues
  ~r/deps\/ra\//
]
