defmodule MerkleDb.FPManifestTest do
  use ExUnit.Case, async: true

  setup do
    MerkleDb.FPManifest.clear_cache()
    :ok
  end

  test "manifest covers all fp_ and get_ functions" do
    asm_functions =
      MerkleDb.ASM.__info__(:functions)
      |> Enum.map(fn {name, arity} -> {Atom.to_string(name), arity} end)
      |> Enum.filter(fn {name, _} ->
        String.starts_with?(name, "fp_") or String.starts_with?(name, "get_")
      end)
      |> MapSet.new()

    manifest_functions =
      MerkleDb.FPManifest.all()
      |> Enum.map(fn entry -> {entry.name, entry.arity} end)
      |> MapSet.new()

    assert manifest_functions == asm_functions
  end

  test "excluded entries are marked and carry reasons" do
    excluded = MerkleDb.FPManifest.excluded()

    assert Enum.all?(excluded, fn entry ->
             entry.allowed == false and is_binary(entry.reason) and entry.reason != ""
           end)
  end

  test "report writes to configured directory" do
    MerkleDb.FPManifest.clear_cache()
    dir = Path.join(System.tmp_dir!(), "merkle_db_manifest_test")
    _ = File.rm_rf(dir)
    System.put_env("MERKLE_DB_INVENTORY_DIR", dir)

    on_exit(fn ->
      System.delete_env("MERKLE_DB_INVENTORY_DIR")
      _ = File.rm_rf(dir)
      MerkleDb.FPManifest.clear_cache()
    end)

    report = MerkleDb.FPManifest.report()
    assert is_binary(report.report_path)
    assert File.exists?(report.report_path)
    assert String.starts_with?(report.report_path, Path.expand(dir))
  end
end
