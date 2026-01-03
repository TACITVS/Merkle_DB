defmodule MerkleDb.BuildInfo do
  @moduledoc false

  @git_commit (
    try do
      case System.cmd("git", ["rev-parse", "--short", "HEAD"]) do
        {hash, 0} -> String.trim(hash)
        _ -> "unknown"
      end
    rescue
      _ -> "unknown"
    end
  )

  @build_time DateTime.utc_now() |> DateTime.truncate(:second) |> DateTime.to_iso8601()

  def info do
    %{
      commit: @git_commit,
      build_time: @build_time
    }
  end
end
