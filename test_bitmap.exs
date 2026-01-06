# test_bitmap.exs
alias MerkleDb.Bitmap

IO.puts "1. Creating Bitmap..."
b1 = Bitmap.new(100)
IO.inspect(byte_size(b1), label: "Size (bytes)")

IO.puts "2. Setting bits..."
# Set bit 5
b2 = Bitmap.set(b1, 5)
# Set bit 99
b3 = Bitmap.set(b2, 99)

IO.puts "3. Testing bits..."
t1 = Bitmap.test(b3, 5)
t2 = Bitmap.test(b3, 99)
t3 = Bitmap.test(b3, 50) # Should be false

IO.puts "Bit 5: #{t1}"
IO.puts "Bit 99: #{t2}"
IO.puts "Bit 50: #{t3}"

if t1 and t2 and not t3 do
  IO.puts "✅ Bitmap Functional Test Passed"
else
  IO.puts "❌ Bitmap Test Failed"
  exit({:error, :failed})
end
