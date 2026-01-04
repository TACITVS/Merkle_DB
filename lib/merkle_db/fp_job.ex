defmodule MerkleDb.FP.Job do
  @moduledoc """
  Native job framework wrapper for long-running FP_ASM_LIB operations.
  """

  alias MerkleDb.ASM

  @type job :: term()

  def start(op_name, args, opts \\ []) do
    ASM.fp_job_start(op_name, args, opts)
  end

  def status(job) do
    ASM.fp_job_status(job)
  end

  def result(job) do
    ASM.fp_job_result(job)
  end

  def cancel(job) do
    ASM.fp_job_cancel(job)
  end

  def start_kmeans(data, n, d, k, max_iter, tol, seed, opts \\ []) do
    start(:fp_kmeans_f64, [data, n, d, k, max_iter, tol, seed], opts)
  end
end
