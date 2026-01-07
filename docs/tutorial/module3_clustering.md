# Module 3: Going Distributed with Raft

MerkleDb achieves high availability and strong consistency using the **Raft consensus algorithm**. In this module, we will set up a local 3-node cluster.

## 1. How Raft works in MerkleDb

In a Raft cluster:
1. One node is elected as the **Leader**.
2. All writes (`put`, `create_collection`, etc.) must go to the Leader.
3. The Leader replicates the operation to the **Followers**.
4. An operation is only committed once a **Quorum** (majority) of nodes have stored it.

---

## 2. Starting a 3-Node Cluster locally

We will simulate three nodes on your machine using different names.

### Node 1 (The Initial Node)
Open a terminal and run:
```bash
iex --name n1@127.0.0.1 -S mix
```

### Node 2
Open a second terminal:
```bash
iex --name n2@127.0.0.1 -S mix
```

### Node 3
Open a third terminal:
```bash
iex --name n3@127.0.0.1 -S mix
```

---

## 3. Forming the Cluster

Go to the console of **Node 2** and **Node 3** and tell them to join Node 1.

On **Node 2**:
```elixir
MerkleDb.Raft.join_cluster(:"n1@127.0.0.1")
```

On **Node 3**:
```elixir
MerkleDb.Raft.join_cluster(:"n1@127.0.0.1")
```

Wait a few seconds for the election to complete. You can check the cluster status on any node:
```elixir
:ra.overview()
```

---

## 4. Testing Fault Tolerance

### Step 4.1: Write to the cluster
On **Node 1**:
```elixir
MerkleDb.KV.create_collection("distributed_test", dim: 5)
MerkleDb.KV.put("distributed_test", "key1", <<1.0::f32, 0.0::f32, 0.0::f32, 0.0::f32, 0.0::f32>>, %{})
```

### Step 4.2: Kill the Leader
Find which node is the leader (using `:ra.overview()`) and close its terminal (or press Ctrl+C twice).

### Step 4.3: Verify Quorum
On any of the remaining nodes, try to read the data:
```elixir
MerkleDb.KV.snapshot("distributed_test").count
# Should still be 1!
```

Try a new write:
```elixir
MerkleDb.KV.put("distributed_test", "key2", <<0.0::f32, 1.0::f32, 0.0::f32, 0.0::f32, 0.0::f32>>, %{})
# This will succeed because 2 out of 3 nodes are still alive (Quorum reached).
```

---

## 5. Summary

By using Raft, MerkleDb ensures that:
- Data is never lost as long as a majority of nodes are alive.
- All clients see the same state (Strong Consistency).
- System recovered automatically from node failures.

In the final module, we will explore advanced indexing and semantic search.
