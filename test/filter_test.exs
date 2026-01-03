defmodule MerkleDb.FilterTest do
  use ExUnit.Case, async: true

  alias MerkleDb.Filter

  describe "matches?/2" do
    test "empty filter matches everything" do
      assert Filter.matches?(%{"any" => "value"}, [])
      assert Filter.matches?(%{}, [])
    end

    test "equality operator" do
      payload = %{"type" => "book", "price" => 25}

      assert Filter.matches?(payload, [{"type", "==", "book"}])
      refute Filter.matches?(payload, [{"type", "==", "electronics"}])
    end

    test "not equal operator" do
      payload = %{"type" => "book"}

      assert Filter.matches?(payload, [{"type", "!=", "electronics"}])
      refute Filter.matches?(payload, [{"type", "!=", "book"}])
    end

    test "greater than operator" do
      payload = %{"price" => 100}

      assert Filter.matches?(payload, [{"price", ">", 50}])
      refute Filter.matches?(payload, [{"price", ">", 100}])
      refute Filter.matches?(payload, [{"price", ">", 150}])
    end

    test "greater than or equal operator" do
      payload = %{"price" => 100}

      assert Filter.matches?(payload, [{"price", ">=", 50}])
      assert Filter.matches?(payload, [{"price", ">=", 100}])
      refute Filter.matches?(payload, [{"price", ">=", 150}])
    end

    test "less than operator" do
      payload = %{"price" => 100}

      assert Filter.matches?(payload, [{"price", "<", 150}])
      refute Filter.matches?(payload, [{"price", "<", 100}])
      refute Filter.matches?(payload, [{"price", "<", 50}])
    end

    test "less than or equal operator" do
      payload = %{"price" => 100}

      assert Filter.matches?(payload, [{"price", "<=", 150}])
      assert Filter.matches?(payload, [{"price", "<=", 100}])
      refute Filter.matches?(payload, [{"price", "<=", 50}])
    end

    test "in operator" do
      payload = %{"category" => "electronics"}

      assert Filter.matches?(payload, [{"category", "in", ["electronics", "books"]}])
      refute Filter.matches?(payload, [{"category", "in", ["books", "music"]}])
    end

    test "not_in operator" do
      payload = %{"category" => "electronics"}

      assert Filter.matches?(payload, [{"category", "not_in", ["books", "music"]}])
      refute Filter.matches?(payload, [{"category", "not_in", ["electronics", "books"]}])
    end

    test "contains operator" do
      payload = %{"title" => "The Great Gatsby"}

      assert Filter.matches?(payload, [{"title", "contains", "Great"}])
      refute Filter.matches?(payload, [{"title", "contains", "Terrible"}])
    end

    test "starts_with operator" do
      payload = %{"title" => "The Great Gatsby"}

      assert Filter.matches?(payload, [{"title", "starts_with", "The"}])
      refute Filter.matches?(payload, [{"title", "starts_with", "Great"}])
    end

    test "exists operator" do
      payload = %{"title" => "Test", "author" => nil}

      assert Filter.matches?(payload, [{"title", "exists", true}])
      refute Filter.matches?(payload, [{"missing", "exists", true}])
      assert Filter.matches?(payload, [{"missing", "exists", false}])
    end

    test "multiple conditions (AND)" do
      payload = %{"type" => "book", "price" => 25, "in_stock" => true}

      assert Filter.matches?(payload, [
        {"type", "==", "book"},
        {"price", "<", 50},
        {"in_stock", "==", true}
      ])

      # One condition fails
      refute Filter.matches?(payload, [
        {"type", "==", "book"},
        {"price", ">", 50}
      ])
    end

    test "nested field access" do
      payload = %{
        "author" => %{
          "name" => "F. Scott Fitzgerald",
          "country" => "USA"
        }
      }

      assert Filter.matches?(payload, [{"author.country", "==", "USA"}])
      refute Filter.matches?(payload, [{"author.country", "==", "UK"}])
    end

    test "boolean values" do
      payload = %{"active" => true, "deleted" => false}

      assert Filter.matches?(payload, [{"active", "==", true}])
      assert Filter.matches?(payload, [{"deleted", "==", false}])
      refute Filter.matches?(payload, [{"active", "==", false}])
    end

    test "handles missing fields gracefully" do
      payload = %{"name" => "test"}

      refute Filter.matches?(payload, [{"missing", "==", "value"}])
      refute Filter.matches?(payload, [{"missing", ">", 5}])
    end
  end

  describe "parse/1" do
    test "parses list of arrays" do
      input = [["price", ">=", 100], ["category", "==", "books"]]

      assert {:ok, conditions} = Filter.parse(input)
      assert conditions == [{"price", ">=", 100}, {"category", "==", "books"}]
    end

    test "parses list of tuples" do
      input = [{"price", ">=", 100}, {"category", "==", "books"}]

      assert {:ok, conditions} = Filter.parse(input)
      assert conditions == [{"price", ">=", 100}, {"category", "==", "books"}]
    end

    test "parses map as equality conditions" do
      input = %{"category" => "books", "in_stock" => true}

      assert {:ok, conditions} = Filter.parse(input)
      assert length(conditions) == 2
      assert {"category", "==", "books"} in conditions
      assert {"in_stock", "==", true} in conditions
    end

    test "parses nil as empty" do
      assert {:ok, []} = Filter.parse(nil)
    end

    test "parses empty list" do
      assert {:ok, []} = Filter.parse([])
    end

    test "returns error for invalid input" do
      assert {:error, _} = Filter.parse("invalid")
      assert {:error, _} = Filter.parse([["field"]])  # missing op and value
    end
  end

  describe "parse_query_param/1" do
    test "parses JSON filter string" do
      param = ~s|[["price",">=",100],["category","==","books"]]|

      assert {:ok, conditions} = Filter.parse_query_param(param)
      assert conditions == [{"price", ">=", 100}, {"category", "==", "books"}]
    end

    test "handles nil" do
      assert {:ok, []} = Filter.parse_query_param(nil)
    end

    test "handles empty string" do
      assert {:ok, []} = Filter.parse_query_param("")
    end

    test "returns error for invalid JSON" do
      assert {:error, "Invalid JSON in filter parameter"} = Filter.parse_query_param("not json")
    end
  end

  describe "validate/1" do
    test "validates empty filter" do
      assert :ok = Filter.validate([])
    end

    test "validates correct filter" do
      conditions = [
        {"field1", "==", "value"},
        {"field2", ">=", 100},
        {"field3", "in", ["a", "b"]}
      ]

      assert :ok = Filter.validate(conditions)
    end

    test "rejects invalid operator" do
      assert {:error, _} = Filter.validate([{"field", "~=", "value"}])
    end

    test "rejects invalid condition format" do
      assert {:error, _} = Filter.validate(["not a tuple"])
    end
  end

  describe "create_mask/2" do
    test "creates binary mask for matching records" do
      payloads = [
        %{"score" => 100},
        %{"score" => 50},
        %{"score" => 150},
        %{"score" => 75}
      ]

      mask = Filter.create_mask(payloads, [{"score", ">=", 100}])

      assert mask == <<1, 0, 1, 0>>
    end

    test "empty filter returns all ones" do
      payloads = [%{}, %{}, %{}]
      mask = Filter.create_mask(payloads, [])

      assert mask == <<1, 1, 1>>
    end
  end

  describe "apply_filter/2" do
    test "filters records by payload" do
      records = [
        {"id1", %{"category" => "books"}},
        {"id2", %{"category" => "electronics"}},
        {"id3", %{"category" => "books"}},
        {"id4", %{"category" => "music"}}
      ]

      result = Filter.apply_filter(records, [{"category", "==", "books"}])

      assert result == ["id1", "id3"]
    end

    test "empty filter returns all IDs" do
      records = [{"id1", %{}}, {"id2", %{}}, {"id3", %{}}]

      result = Filter.apply_filter(records, [])

      assert result == ["id1", "id2", "id3"]
    end
  end
end
