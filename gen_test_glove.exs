dims = 300
common_words = ["the", "was", "a", "of", "and", "in", "he", "it", "is"]

# Noise words have small random-ish values
vectors = 
  Enum.map(common_words, fn word ->
    [word | Enum.map(1..dims, fn _ -> "0.01" end)]
  end)

# Philosophy [1, 0, 0...]
philosophy_vec = ["1.0"] ++ Enum.map(1..(dims-1), fn _ -> "0.0" end)
philosophy = ["philosophy" | philosophy_vec]
filosofia = ["filosofia" | philosophy_vec]

# Logic [0, 1, 0...]
logic_vec = ["0.0", "1.0"] ++ Enum.map(1..(dims-2), fn _ -> "0.0" end)
logic = ["logic" | logic_vec]
logica = ["logica" | logic_vec]

all_data = [philosophy, filosofia, logic, logica | vectors]
File.write!("data/glove_test.txt", Enum.map(all_data, &Enum.join(&1, " ")) |> Enum.join("\n"))
IO.puts "Updated data/glove_test.txt with multilingual entries (English/Portuguese)."