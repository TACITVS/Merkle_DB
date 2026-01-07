defmodule MerkleDb.TreeUnitTest do
  use ExUnit.Case
  alias MerkleDb.Tree

  describe "MerkleDb.Tree.new/1" do
    test "initializes with correct defaults" do
      tree = Tree.new(dim: 128, precision: :f32)
      assert tree.dim == 128
      assert tree.precision == :f32
      assert tree.count == 0
      assert is_tuple(tree.columns)
      assert tuple_size(tree.columns) == 128
    end

    test "initializes with 0 dimensions" do
      tree = Tree.new()
      assert tree.dim == 0
      assert tree.columns == nil
    end
  end

  describe "MerkleDb.Tree.insert/4" do
    setup do
      {:ok, tree: Tree.new(dim: 4, precision: :f32)}
    end

    test "basic insertion", %{tree: tree} do
      vec = [1.0, 0.0, 0.0, 0.0]
      new_tree = Tree.insert(tree, "k1", vec, %{"meta" => "data"})
      assert new_tree.count == 1
      assert Map.has_key?(new_tree.keys, 0)
      assert new_tree.keys[0] == "k1"
      assert new_tree.key_index["k1"] == 0
      assert new_tree.metadata[0] == %{"meta" => "data"}
    end

    test "updates existing key (soft delete/append)", %{tree: tree} do
      v1 = [1.0, 0.0, 0.0, 0.0]
      v2 = [0.0, 1.0, 0.0, 0.0]
      
      tree1 = Tree.insert(tree, "key", v1)
      tree2 = Tree.insert(tree1, "key", v2)
      
      assert tree2.count == 2
      assert tree2.key_index["key"] == 1
      assert MapSet.member?(tree2.tombstones, 0)
    end

    test "error on dimension mismatch", %{tree: tree} do
      assert_raise ArgumentError, ~r/Dimension mismatch/, fn ->
        Tree.insert(tree, "bad", [1.0, 2.0])
      end
    end
  end

  describe "MerkleDb.Tree.insert_batch/2" do
    test "correctly handles duplicate keys within a batch" do
      tree = Tree.new(dim: 2, precision: :f32)
      batch = [
        {"k1", [1.0, 0.0], %{tag: 1}},
        {"k1", [0.0, 1.0], %{tag: 2}}
      ]
      
      new_tree = Tree.insert_batch(tree, batch)
      
      # Final state should point to second k1
      assert new_tree.key_index["k1"] == 1
      # First k1 should be tombstoned
      # Note: Currently Tree.insert_batch calculates updated_tombstones
      # but we need to verify if internal duplicates are handled.
      assert MapSet.member?(new_tree.tombstones, 0)
    end

    test "preserves metadata across batch" do
      tree = Tree.new(dim: 2, precision: :f32)
      batch = [
        {"k1", [1.0, 0.0], %{a: 1}},
        {"k2", [0.0, 1.0], %{b: 2}}
      ]
      new_tree = Tree.insert_batch(tree, batch)
      assert new_tree.metadata[0] == %{a: 1}
      assert new_tree.metadata[1] == %{b: 2}
    end
  end

  describe "MerkleDb.Tree.delete/2" do
    test "marks index as tombstoned" do
      tree = Tree.new(dim: 2) |> Tree.insert("k1", [1.0, 1.0])
      {:ok, idx} = Map.fetch(tree.key_index, "k1")
      
      new_tree = Tree.delete(tree, "k1")
      assert MapSet.member?(new_tree.tombstones, idx)
      refute Map.has_key?(new_tree.key_index, "k1")
    end

    test "returns error for missing key" do
      tree = Tree.new(dim: 2)
      assert Tree.delete(tree, "none") == {:error, :not_found}
    end
  end
end
