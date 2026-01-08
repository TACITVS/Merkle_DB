defmodule MerkleDb.MixProject do
  use Mix.Project

  @version "0.2.0"
  @source_url "https://github.com/merkle-db/merkle-db"

  def project do
    [
      app: :merkle_db,
      version: @version,
      elixir: "~> 1.14",
      start_permanent: Mix.env() == :prod,
      compilers: [:elixir_make] ++ Mix.compilers(),
      make_targets: ["all"],
      make_clean: ["clean"],
      deps: deps(),
      aliases: aliases(),

      # Releases configuration
      releases: releases(),

      # Dialyzer configuration
      dialyzer: dialyzer(),

      # Documentation
      name: "MerkleDb",
      description: "High-performance vector database with Raft consensus and AVX2 acceleration",
      source_url: @source_url,
      docs: docs(),

      # Package metadata
      package: package()
    ]
  end

  def application do
    [
      extra_applications: [:logger, :crypto, :ssl],
      mod: {MerkleDb.Application, []}
    ]
  end

  defp deps do
    [
      # Core dependencies
      {:elixir_make, "~> 0.7", runtime: false},
      {:plug_cowboy, "~> 2.6"},
      {:jason, "~> 1.4"},
      {:ra, "~> 2.13"},

      # Development and testing
      {:dialyxir, "~> 1.4", only: [:dev, :test], runtime: false},
      {:credo, "~> 1.7", only: [:dev, :test], runtime: false},
      {:ex_doc, "~> 0.31", only: :dev, runtime: false}
    ]
  end

  defp aliases do
    [
      # Quality checks
      quality: ["format --check-formatted", "credo --strict", "dialyzer"],

      # Full test suite
      "test.all": ["test", "quality"],

      # Release preparation
      "release.build": ["deps.get", "compile", "release"],

      # Database reset (for development)
      "db.reset": fn _ ->
        File.rm_rf!("data")
        File.mkdir_p!("data")
        Mix.shell().info("Database reset complete")
      end
    ]
  end

  defp releases do
    [
      merkle_db: [
        include_executables_for: [:windows, :unix],
        applications: [runtime_tools: :permanent],

        steps: [:assemble, :tar],

        # Windows-specific configuration
        rel_templates_path: "rel",

        # Cookie for distributed Erlang
        cookie: System.get_env("RELEASE_COOKIE") || generate_cookie()
      ]
    ]
  end

  defp dialyzer do
    [
      plt_file: {:no_warn, "priv/plts/dialyzer.plt"},
      plt_add_apps: [:mix, :ex_unit],
      flags: [
        :unmatched_returns,
        :error_handling,
        :no_opaque
      ],
      ignore_warnings: ".dialyzer_ignore.exs"
    ]
  end

  defp docs do
    [
      main: "readme",
      extras: ["README.md", "CHANGELOG.md"],
      source_ref: "v#{@version}",
      groups_for_modules: [
        "Core": [
          MerkleDb.KV,
          MerkleDb.Tree,
          MerkleDb.Query
        ],
        "Storage": [
          MerkleDb.WAL,
          MerkleDb.Persistence,
          MerkleDb.Storage
        ],
        "Consensus": [
          MerkleDb.Raft,
          MerkleDb.Raft.Machine,
          MerkleDb.Raft.Supervisor
        ],
        "Web": [
          MerkleDb.Web.Router,
          MerkleDb.Web.Auth
        ],
        "Security": [
          MerkleDb.ApiKeyStore,
          MerkleDb.RateLimiter,
          MerkleDb.Validator,
          MerkleDb.ConfigValidator
        ],
        "Analytics": [
          MerkleDb.Analytics,
          MerkleDb.TextAnalytics,
          MerkleDb.TelemetryAggregator
        ]
      ]
    ]
  end

  defp package do
    [
      name: "merkle_db",
      files: ~w(lib native priv .formatter.exs mix.exs README* LICENSE* CHANGELOG*),
      licenses: ["MIT"],
      links: %{
        "GitHub" => @source_url
      }
    ]
  end

  defp generate_cookie do
    :crypto.strong_rand_bytes(32) |> Base.encode64() |> binary_part(0, 32)
  end
end
