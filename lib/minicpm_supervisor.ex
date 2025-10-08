defmodule MiniCPMSupervisor do
  @moduledoc """
  Supervisor for MiniCPM Vision Server.
  Manages the lifecycle of the MiniCPM service.
  """

  use Supervisor

  def start_link(init_arg) do
    Supervisor.start_link(__MODULE__, init_arg, name: __MODULE__)
  end

  @impl true
  def init(_init_arg) do
    children = [
      {MiniCPMVisionService, []}
    ]

    Supervisor.init(children, strategy: :one_for_one, name: __MODULE__)
  end
end
