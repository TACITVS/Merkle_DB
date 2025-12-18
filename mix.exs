defmodule MerkleDb.MixProject do
  use Mix.Project

  def project do
    [
      app: :merkle_db,
      version: "0.1.0",
      elixir: "~> 1.14",
      start_permanent: Mix.env() == :prod,
	  compilers: [:elixir_make] ++ Mix.compilers(), # <--- ADD THIS
      make_targets: ["all"],                        # <--- ADD THIS
      make_clean: ["clean"],                        # <--- ADD THIS
      deps: deps()
    ]
  end

  # Run "mix help compile.app" to learn about applications.
  def application do
    [
      extra_applications: [:logger],
      mod: {MerkleDb.Application, []}
    ]
  end

  # Run "mix help deps" to learn about dependencies.
  defp deps do
    [
      # {:dep_from_hexpm, "~> 0.3.0"},
      # {:dep_from_git, git: "https://github.com/elixir-lang/my_dep.git", tag: "0.1.0"}
      {:elixir_make, "~> 0.7", runtime: false},
      {:plug_cowboy, "~> 2.6"}
    ]
  end
end


