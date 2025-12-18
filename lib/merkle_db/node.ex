defmodule MerkleDb.Node do
  @type hash :: binary()
  @type vector :: binary()
  
  # A node in the Merkle DAG
  # type: :leaf | :branch
  # data: The vector (for leaf) or nil
  # children: Map of {index => hash} (for branch) or nil
  # hash: The BLAKE2s hash of the content
  defstruct [:type, :data, :children, :hash, :key]

  @hash_algo :blake2s

  def create_leaf(key, vector_data) when is_binary(vector_data) do
    # Leaf stores the key and the raw vector data
    # Hash = hash(key + vector_data)
    hash = :crypto.hash(@hash_algo, <<key::binary, vector_data::binary>>)
    
    %__MODULE__{
      type: :leaf,
      key: key,
      data: vector_data,
      children: nil,
      hash: hash
    }
  end

  def create_branch(children) when is_map(children) do
    # Branch stores pointers to children
    # Hash = hash(concat(sorted_child_hashes))
    
    # Sort by key (or index) for deterministic hashing
    sorted_hashes = 
      children
      |> Enum.sort_by(fn {k, _v} -> k end)
      |> Enum.map(fn {_k, h} -> h end)
      |> Enum.join(<<>>)

    hash = :crypto.hash(@hash_algo, sorted_hashes)

    %__MODULE__{
      type: :branch,
      key: nil,
      data: nil,
      children: children,
      hash: hash
    }
  end
  
  # Check if a node is valid
  def verify(%__MODULE__{type: :leaf, key: k, data: d, hash: h}) do
    h == :crypto.hash(@hash_algo, <<k::binary, d::binary>>)
  end
  
  def verify(%__MODULE__{type: :branch, children: c, hash: h}) do
    sorted_hashes = 
      c
      |> Enum.sort_by(fn {k, _v} -> k end)
      |> Enum.map(fn {_k, ch} -> ch end)
      |> Enum.join(<<>>)
      
    h == :crypto.hash(@hash_algo, sorted_hashes)
  end
end
