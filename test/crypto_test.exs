defmodule MerkleDb.CryptoTest do
  use ExUnit.Case, async: true

  alias MerkleDb.Crypto

  describe "hash/1" do
    test "produces 32-byte output" do
      hash = Crypto.hash("hello")
      assert byte_size(hash) == 32
    end

    test "is deterministic" do
      hash1 = Crypto.hash("test data")
      hash2 = Crypto.hash("test data")
      assert hash1 == hash2
    end

    test "different inputs produce different hashes" do
      hash1 = Crypto.hash("input1")
      hash2 = Crypto.hash("input2")
      refute hash1 == hash2
    end

    test "empty input works" do
      hash = Crypto.hash("")
      assert byte_size(hash) == 32
    end
  end

  describe "hash_tagged/2" do
    test "different tags produce different hashes for same data" do
      data = "same data"
      hash0 = Crypto.hash_tagged(0x00, data)
      hash1 = Crypto.hash_tagged(0x01, data)
      refute hash0 == hash1
    end

    test "domain separation prevents cross-context collisions" do
      # This is the key security property
      leaf_hash = Crypto.hash_leaf("record data")
      internal_hash = Crypto.hash_tagged(Crypto.tag_internal(), "record data")
      refute leaf_hash == internal_hash
    end
  end

  describe "hash_internal/2" do
    test "combines two hashes correctly" do
      left = Crypto.hash("left")
      right = Crypto.hash("right")
      combined = Crypto.hash_internal(left, right)
      assert byte_size(combined) == 32
    end

    test "order matters" do
      h1 = Crypto.hash("a")
      h2 = Crypto.hash("b")
      forward = Crypto.hash_internal(h1, h2)
      backward = Crypto.hash_internal(h2, h1)
      refute forward == backward
    end
  end

  describe "hash_empty/0" do
    test "produces consistent empty hash" do
      empty1 = Crypto.hash_empty()
      empty2 = Crypto.hash_empty()
      assert empty1 == empty2
      assert byte_size(empty1) == 32
    end
  end

  describe "to_hex/1 and from_hex/1" do
    test "roundtrip works" do
      original = Crypto.hash("test")
      hex = Crypto.to_hex(original)
      assert byte_size(hex) == 64
      assert {:ok, decoded} = Crypto.from_hex(hex)
      assert decoded == original
    end

    test "hex is lowercase" do
      hash = Crypto.hash("test")
      hex = Crypto.to_hex(hash)
      assert hex == String.downcase(hex)
    end

    test "invalid hex returns error" do
      assert {:error, :invalid_hex} = Crypto.from_hex("not valid hex")
      assert {:error, :invalid_hex} = Crypto.from_hex("abc")
    end
  end

  describe "secure_compare/2" do
    test "returns true for equal hashes" do
      hash = Crypto.hash("test")
      assert Crypto.secure_compare(hash, hash)
    end

    test "returns false for different hashes" do
      h1 = Crypto.hash("a")
      h2 = Crypto.hash("b")
      refute Crypto.secure_compare(h1, h2)
    end

    test "returns false for different sizes" do
      refute Crypto.secure_compare(<<1, 2, 3>>, <<1, 2, 3, 4>>)
    end
  end

  describe "tag constants" do
    test "tags are distinct" do
      tags = [
        Crypto.tag_leaf(),
        Crypto.tag_internal(),
        Crypto.tag_index(),
        Crypto.tag_manifest(),
        Crypto.tag_segment(),
        Crypto.tag_centroid(),
        Crypto.tag_posting(),
        Crypto.tag_empty()
      ]

      assert length(Enum.uniq(tags)) == length(tags)
    end
  end
end
