dims = 300
philosophy = ["philosophy"] ++ Enum.map(1..dims, fn _ -> "1.0" end)
logic = ["logic"] ++ Enum.map(1..dims, fn _ -> "2.0" end)
File.write!("data/glove_test.txt", Enum.join(philosophy, " ") <> "\n" <> Enum.join(logic, " ") <> "\n")
IO.puts "Created data/glove_test.txt"
