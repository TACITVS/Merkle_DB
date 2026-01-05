alias MerkleDb.{KV, Query}
{:ok, _} = Application.ensure_all_started(:merkle_db)
KV.reset()
v = for _ <- 1..64, into: <<>>, do: <<1.0::little-float-64>>
KV.put("k1", v)
IO.puts "Performing KNN..."
res = Query.execute(KV.snapshot(), [:knn, v, 1, 0.0])
IO.puts "Result: #{inspect(res)}"
