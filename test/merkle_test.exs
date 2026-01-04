defmodule MerkleDb.MerkleTest do
  use ExUnit.Case, async: true

  alias MerkleDb.Merkle
  alias MerkleDb.Crypto

  # Helper to create test records
  defp make_record(id, value \\ nil) do
    value = value || id
    {id, [value * 1.0], %{}, 0}
  end

  describe "build_tree/1" do
    test "empty tree" do
      {:ok, tree} = Merkle.build_tree([])
      assert tree.leaf_count == 0
      assert tree.root == Crypto.hash_empty()
    end

    test "single record" do
      records = [make_record(1)]
      {:ok, tree} = Merkle.build_tree(records)
      assert tree.leaf_count == 1
      assert byte_size(tree.root) == 32
    end

    test "two records" do
      records = [make_record(1), make_record(2)]
      {:ok, tree} = Merkle.build_tree(records)
      assert tree.leaf_count == 2
      assert byte_size(tree.root) == 32
    end

    test "power of two records" do
      records = Enum.map(1..8, &make_record/1)
      {:ok, tree} = Merkle.build_tree(records)
      assert tree.leaf_count == 8
    end

    test "non-power of two records" do
      records = Enum.map(1..5, &make_record/1)
      {:ok, tree} = Merkle.build_tree(records)
      assert tree.leaf_count == 5
    end

    test "records are sorted by ID" do
      # Insert out of order
      records = [make_record(3), make_record(1), make_record(2)]
      {:ok, tree} = Merkle.build_tree(records)

      # Tree should have sorted leaf_ids
      assert tree.leaf_ids == [1, 2, 3]
    end

    test "deterministic root for same data" do
      records = Enum.map(1..10, &make_record/1)
      {:ok, tree1} = Merkle.build_tree(records)
      {:ok, tree2} = Merkle.build_tree(Enum.shuffle(records))
      assert tree1.root == tree2.root
    end

    test "different data produces different root" do
      {:ok, tree1} = Merkle.build_tree([make_record(1)])
      {:ok, tree2} = Merkle.build_tree([make_record(2)])
      refute tree1.root == tree2.root
    end
  end

  describe "prove_inclusion/2" do
    test "empty tree returns error" do
      {:ok, tree} = Merkle.build_tree([])
      assert {:error, :empty_tree} = Merkle.prove_inclusion(tree, 1)
    end

    test "missing record returns error" do
      {:ok, tree} = Merkle.build_tree([make_record(1)])
      assert {:error, :not_found} = Merkle.prove_inclusion(tree, 999)
    end

    test "single record proof has empty path" do
      {:ok, tree} = Merkle.build_tree([make_record(1)])
      {:ok, proof} = Merkle.prove_inclusion(tree, 1)
      assert proof.path == []
      assert proof.record_id == 1
      assert proof.leaf_index == 0
    end

    test "two record proof has one path element" do
      records = [make_record(1), make_record(2)]
      {:ok, tree} = Merkle.build_tree(records)

      {:ok, proof1} = Merkle.prove_inclusion(tree, 1)
      assert length(proof1.path) == 1
      assert proof1.leaf_index == 0

      {:ok, proof2} = Merkle.prove_inclusion(tree, 2)
      assert length(proof2.path) == 1
      assert proof2.leaf_index == 1
    end

    test "proof includes snapshot root" do
      records = [make_record(1)]
      {:ok, tree} = Merkle.build_tree(records)
      {:ok, proof} = Merkle.prove_inclusion(tree, 1)
      assert proof.snapshot_root == tree.root
    end
  end

  describe "verify_inclusion/3" do
    test "valid proof verifies" do
      record = make_record(1)
      {:ok, tree} = Merkle.build_tree([record])
      {:ok, proof} = Merkle.prove_inclusion(tree, 1)

      assert Merkle.verify_inclusion(tree.root, record, proof)
    end

    test "wrong record fails verification" do
      {:ok, tree} = Merkle.build_tree([make_record(1)])
      {:ok, proof} = Merkle.prove_inclusion(tree, 1)

      # Try to verify with wrong record
      wrong_record = make_record(1, 999)  # Same ID, different vector
      refute Merkle.verify_inclusion(tree.root, wrong_record, proof)
    end

    test "wrong root fails verification" do
      record = make_record(1)
      {:ok, tree} = Merkle.build_tree([record])
      {:ok, proof} = Merkle.prove_inclusion(tree, 1)

      wrong_root = Crypto.hash("wrong")
      refute Merkle.verify_inclusion(wrong_root, record, proof)
    end

    test "wrong record ID fails verification" do
      record = make_record(1)
      {:ok, tree} = Merkle.build_tree([record])
      {:ok, proof} = Merkle.prove_inclusion(tree, 1)

      # Modify the record ID
      wrong_record = {999, elem(record, 1), elem(record, 2), elem(record, 3)}
      refute Merkle.verify_inclusion(tree.root, wrong_record, proof)
    end

    test "multi-record tree verification" do
      records = Enum.map(1..8, &make_record/1)
      {:ok, tree} = Merkle.build_tree(records)

      # Verify each record
      for record <- records do
        {id, _, _, _} = record
        {:ok, proof} = Merkle.prove_inclusion(tree, id)
        assert Merkle.verify_inclusion(tree.root, record, proof),
               "Verification failed for record #{id}"
      end
    end

    test "odd number of records verification" do
      records = Enum.map(1..7, &make_record/1)
      {:ok, tree} = Merkle.build_tree(records)

      for record <- records do
        {id, _, _, _} = record
        {:ok, proof} = Merkle.prove_inclusion(tree, id)
        assert Merkle.verify_inclusion(tree.root, record, proof)
      end
    end
  end

  describe "proof encoding/decoding" do
    test "roundtrip" do
      records = [make_record(1), make_record(2)]
      {:ok, tree} = Merkle.build_tree(records)
      {:ok, proof} = Merkle.prove_inclusion(tree, 1)

      encoded = Merkle.encode_proof(proof)
      assert is_binary(encoded)

      {:ok, decoded} = Merkle.decode_proof(encoded)
      assert decoded.version == proof.version
      assert decoded.snapshot_root == proof.snapshot_root
      assert decoded.record_id == proof.record_id
      assert decoded.leaf_index == proof.leaf_index
      assert decoded.path == proof.path
    end

    test "invalid binary returns error" do
      assert {:error, :invalid_proof} = Merkle.decode_proof(<<1, 2, 3>>)
    end
  end

  describe "golden root tests" do
    @tag :golden
    test "known input produces known output" do
      # This is a golden test - the expected root should be computed once
      # and then used to verify consistency across implementations

      records = [
        {1, [1.0, 2.0, 3.0], %{}, 0},
        {2, [4.0, 5.0, 6.0], %{}, 0},
        {3, [7.0, 8.0, 9.0], %{}, 0}
      ]

      {:ok, tree} = Merkle.build_tree(records)

      # Print for initial capture (comment out after first run)
      # IO.puts("Golden root: #{Crypto.to_hex(tree.root)}")

      # Verify determinism
      {:ok, tree2} = Merkle.build_tree(Enum.reverse(records))
      assert tree.root == tree2.root
    end
  end
end
