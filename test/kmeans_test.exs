defmodule MerkleDb.KmeansTest do
  use ExUnit.Case, async: false

  alias MerkleDb.FPDispatcher

  @moduledoc """
  Unit tests for fp_kmeans_f64 NIF function.

  Tests verify:
  - Basic clustering correctness
  - Deterministic results with same seed
  - Expected centroids for known datasets
  - Cluster assignment consistency
  """

  setup do
    # Ensure FPDispatcher is running
    case Process.whereis(FPDispatcher) do
      nil -> {:ok, _} = FPDispatcher.start_link(nil)
      _pid -> :ok
    end
    :ok
  end

  describe "fp_kmeans_f64 basic" do
    test "clusters 4 points into 2 clusters" do
      # Two clear clusters: (0,0), (1,1) and (10,10), (11,11)
      data = pack_f64([
        0.0, 0.0,
        1.0, 1.0,
        10.0, 10.0,
        11.0, 11.0
      ])

      n = 4  # 4 points
      d = 2  # 2 dimensions
      k = 2  # 2 clusters
      max_iter = 100
      tol = 1.0e-4
      seed = 42

      result = FPDispatcher.call(:fp_kmeans_f64, [data, n, d, k, max_iter, tol, seed])

      # Get results via accessors
      centroids = FPDispatcher.call(:get_KMeansResult_centroids, [result, k * d * 8])
      assignments_bin = FPDispatcher.call(:get_KMeansResult_assignments, [result, n * 4])
      converged = FPDispatcher.call(:get_KMeansResult_converged, [result])

      assert is_binary(centroids)
      assert is_binary(assignments_bin)
      assert converged in [0, 1]

      # Unpack centroids
      centroids_list = unpack_f64(centroids)
      assert length(centroids_list) == k * d

      # Verify centroids are near expected values
      c1 = {Enum.at(centroids_list, 0), Enum.at(centroids_list, 1)}
      c2 = {Enum.at(centroids_list, 2), Enum.at(centroids_list, 3)}

      # One centroid should be near (0.5, 0.5), the other near (10.5, 10.5)
      assert (near?(c1, {0.5, 0.5}) and near?(c2, {10.5, 10.5})) or
             (near?(c1, {10.5, 10.5}) and near?(c2, {0.5, 0.5}))
    end

    test "returns valid cluster assignments" do
      data = pack_f64([0.0, 1.0, 2.0, 3.0, 10.0, 11.0])
      n = 3
      d = 2
      k = 2

      result = FPDispatcher.call(:fp_kmeans_f64, [data, n, d, k, 100, 1.0e-4, 42])
      assignments_bin = FPDispatcher.call(:get_KMeansResult_assignments, [result, n * 4])

      assignments = for <<a::little-32 <- assignments_bin>>, do: a

      assert length(assignments) == n
      assert Enum.all?(assignments, fn a -> a >= 0 and a < k end)
    end

    test "deterministic with same seed" do
      data = pack_f64([
        1.0, 2.0,
        3.0, 4.0,
        5.0, 6.0,
        7.0, 8.0,
        100.0, 100.0,
        101.0, 101.0
      ])

      n = 6
      d = 2
      k = 2
      seed = 12345

      result1 = FPDispatcher.call(:fp_kmeans_f64, [data, n, d, k, 100, 1.0e-4, seed])
      result2 = FPDispatcher.call(:fp_kmeans_f64, [data, n, d, k, 100, 1.0e-4, seed])

      centroids1 = FPDispatcher.call(:get_KMeansResult_centroids, [result1, k * d * 8])
      centroids2 = FPDispatcher.call(:get_KMeansResult_centroids, [result2, k * d * 8])

      assignments1 = FPDispatcher.call(:get_KMeansResult_assignments, [result1, n * 4])
      assignments2 = FPDispatcher.call(:get_KMeansResult_assignments, [result2, n * 4])

      assert centroids1 == centroids2
      assert assignments1 == assignments2
    end
  end

  describe "fp_kmeans_f64 expected centroids" do
    test "single point per cluster gives exact centroids" do
      # 3 points, 3 clusters = each point is its own centroid
      data = pack_f64([
        1.0, 1.0,
        5.0, 5.0,
        9.0, 9.0
      ])

      n = 3
      d = 2
      k = 3

      result = FPDispatcher.call(:fp_kmeans_f64, [data, n, d, k, 100, 1.0e-4, 42])
      centroids_bin = FPDispatcher.call(:get_KMeansResult_centroids, [result, k * d * 8])
      centroids = unpack_f64(centroids_bin)

      # Each point should be a centroid (order may vary)
      expected_set = MapSet.new([{1.0, 1.0}, {5.0, 5.0}, {9.0, 9.0}])

      actual_set = MapSet.new([
        {Enum.at(centroids, 0), Enum.at(centroids, 1)},
        {Enum.at(centroids, 2), Enum.at(centroids, 3)},
        {Enum.at(centroids, 4), Enum.at(centroids, 5)}
      ])

      # Check all centroids match expected points (with tolerance)
      assert all_match_approx?(actual_set, expected_set, 0.01)
    end

    test "two points per cluster gives mean centroid" do
      # Cluster A: (0,0) and (2,2) -> centroid at (1,1)
      # Cluster B: (10,0) and (12,2) -> centroid at (11,1)
      data = pack_f64([
        0.0, 0.0,
        2.0, 2.0,
        10.0, 0.0,
        12.0, 2.0
      ])

      n = 4
      d = 2
      k = 2

      result = FPDispatcher.call(:fp_kmeans_f64, [data, n, d, k, 100, 1.0e-4, 42])
      centroids_bin = FPDispatcher.call(:get_KMeansResult_centroids, [result, k * d * 8])
      centroids = unpack_f64(centroids_bin)

      c1 = {Enum.at(centroids, 0), Enum.at(centroids, 1)}
      c2 = {Enum.at(centroids, 2), Enum.at(centroids, 3)}

      # One centroid near (1,1), other near (11,1)
      assert (near?(c1, {1.0, 1.0}) and near?(c2, {11.0, 1.0})) or
             (near?(c1, {11.0, 1.0}) and near?(c2, {1.0, 1.0}))
    end

    test "uniform line clusters correctly" do
      # Points on a line: 0, 10, 20, 30, 40, 50, 60, 70, 80, 90
      data = pack_f64(
        for i <- 0..9, do: i * 10.0
      )

      n = 10
      d = 1
      k = 2

      result = FPDispatcher.call(:fp_kmeans_f64, [data, n, d, k, 100, 1.0e-4, 42])
      centroids_bin = FPDispatcher.call(:get_KMeansResult_centroids, [result, k * d * 8])
      centroids = unpack_f64(centroids_bin)

      # Centroids should partition the line
      assert length(centroids) == 2

      [c1, c2] = Enum.sort(centroids)

      # First centroid should be in lower half, second in upper half
      assert c1 < 50.0
      assert c2 >= 50.0
    end
  end

  describe "fp_kmeans_f64 edge cases" do
    test "handles k=1 (all points in one cluster)" do
      data = pack_f64([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
      n = 3
      d = 2
      k = 1

      result = FPDispatcher.call(:fp_kmeans_f64, [data, n, d, k, 100, 1.0e-4, 42])

      assignments_bin = FPDispatcher.call(:get_KMeansResult_assignments, [result, n * 4])
      assignments = for <<a::little-32 <- assignments_bin>>, do: a

      assert length(assignments) == n
      assert Enum.all?(assignments, fn a -> a == 0 end)

      centroids_bin = FPDispatcher.call(:get_KMeansResult_centroids, [result, k * d * 8])
      centroids = unpack_f64(centroids_bin)

      # Centroid should be mean of all points: (1+3+5)/3 = 3, (2+4+6)/3 = 4
      assert length(centroids) == 2
      assert_in_delta Enum.at(centroids, 0), 3.0, 0.01
      assert_in_delta Enum.at(centroids, 1), 4.0, 0.01
    end

    test "handles max_iter=1" do
      data = pack_f64([0.0, 0.0, 10.0, 10.0])
      n = 2
      d = 2
      k = 2

      result = FPDispatcher.call(:fp_kmeans_f64, [data, n, d, k, 1, 1.0e-4, 42])

      # Should still produce valid centroids
      centroids_bin = FPDispatcher.call(:get_KMeansResult_centroids, [result, k * d * 8])
      centroids = unpack_f64(centroids_bin)

      assert length(centroids) == k * d

      # And valid assignments
      assignments_bin = FPDispatcher.call(:get_KMeansResult_assignments, [result, n * 4])
      assignments = for <<a::little-32 <- assignments_bin>>, do: a

      assert length(assignments) == n
    end

    test "handles high-dimensional data" do
      # 10 points in 50 dimensions
      d = 50
      n = 10
      k = 3

      :rand.seed(:exsss, {42, 42, 42})
      data = pack_f64(
        for _i <- 1..n, _j <- 1..d do
          :rand.uniform() * 100.0
        end
      )

      result = FPDispatcher.call(:fp_kmeans_f64, [data, n, d, k, 100, 1.0e-4, 42])

      centroids_bin = FPDispatcher.call(:get_KMeansResult_centroids, [result, k * d * 8])
      centroids = unpack_f64(centroids_bin)

      assignments_bin = FPDispatcher.call(:get_KMeansResult_assignments, [result, n * 4])
      assignments = for <<a::little-32 <- assignments_bin>>, do: a

      assert length(centroids) == k * d
      assert length(assignments) == n
    end
  end

  describe "fp_kmeans_f64 cluster quality" do
    test "well-separated clusters have correct assignments" do
      # Create 3 tight, well-separated clusters
      # Cluster A: around (0, 0)
      # Cluster B: around (100, 0)
      # Cluster C: around (50, 100)

      cluster_a = [{0.0, 0.0}, {0.1, 0.1}, {-0.1, 0.1}]
      cluster_b = [{100.0, 0.0}, {100.1, 0.1}, {99.9, 0.1}]
      cluster_c = [{50.0, 100.0}, {50.1, 100.1}, {49.9, 100.1}]

      points = cluster_a ++ cluster_b ++ cluster_c
      data = pack_f64(Enum.flat_map(points, fn {x, y} -> [x, y] end))

      n = 9
      d = 2
      k = 3

      result = FPDispatcher.call(:fp_kmeans_f64, [data, n, d, k, 100, 1.0e-4, 42])
      assignments_bin = FPDispatcher.call(:get_KMeansResult_assignments, [result, n * 4])
      assignments = for <<a::little-32 <- assignments_bin>>, do: a

      # Verify: points 0,1,2 share same cluster
      # points 3,4,5 share same cluster
      # points 6,7,8 share same cluster
      assert Enum.at(assignments, 0) == Enum.at(assignments, 1)
      assert Enum.at(assignments, 1) == Enum.at(assignments, 2)

      assert Enum.at(assignments, 3) == Enum.at(assignments, 4)
      assert Enum.at(assignments, 4) == Enum.at(assignments, 5)

      assert Enum.at(assignments, 6) == Enum.at(assignments, 7)
      assert Enum.at(assignments, 7) == Enum.at(assignments, 8)

      # And all three groups are different clusters
      cluster_ids = [
        Enum.at(assignments, 0),
        Enum.at(assignments, 3),
        Enum.at(assignments, 6)
      ]
      assert length(Enum.uniq(cluster_ids)) == 3
    end
  end

  # Helpers

  defp pack_f64(values) do
    values
    |> Enum.map(fn v -> <<v::little-float-size(64)>> end)
    |> IO.iodata_to_binary()
  end

  defp unpack_f64(binary) do
    for <<v::little-float-size(64) <- binary>>, do: v
  end

  defp near?({x1, y1}, {x2, y2}, tolerance \\ 0.5) do
    abs(x1 - x2) < tolerance and abs(y1 - y2) < tolerance
  end

  defp all_match_approx?(actual_set, expected_set, tolerance) do
    Enum.all?(actual_set, fn actual_point ->
      Enum.any?(expected_set, fn expected_point ->
        near?(actual_point, expected_point, tolerance)
      end)
    end)
  end
end
