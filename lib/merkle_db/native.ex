defmodule MerkleDb.Native do
  # @on_load :load_nifs

  # def load_nifs do
  #   path = :code.priv_dir(:merkle_db) |> Path.join("merkle_nif") |> String.to_charlist()
  #   :erlang.load_nif(path, 0)
  # end

  def exec_asm(_atom, _binary), do: :erlang.nif_error(:nif_not_loaded)
  def create_vector(_binary), do: :erlang.nif_error(:nif_not_loaded)
end