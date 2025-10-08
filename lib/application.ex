defmodule MinicpmVisionService.Application do
  # See https://hexdocs.pm/elixir/Application.html
  # for more information on OTP Applications
  @moduledoc false

  use Application
  require Logger

  @impl true
  def start(_type, _args) do
    # Configure distributed node settings
    configure_node()

    children = [
      # Starts the MiniCPM Vision Server
      MiniCPMSupervisor
    ]

    # See https://hexdocs.pm/elixir/Supervisor.html
    # for other strategies and supported options
    opts = [strategy: :one_for_one, name: MinicpmVisionService.Supervisor]
    Supervisor.start_link(children, opts)
  end

  # Configure the Erlang node for distributed communication
  defp configure_node do
    # Get node configuration from environment or defaults
    node_name = System.get_env("VISION_NODE_NAME", "vision_service@localhost")
    cookie = System.get_env("VISION_ERL_COOKIE", "vision_service_cookie")

    # Set the cookie for distributed communication
    if cookie do
      String.to_atom(cookie) |> Node.set_cookie()
    end

    # Start the node with distribution
    case Node.start(String.to_atom(node_name)) do
      {:ok, _pid} ->
        Logger.info("Vision service started as distributed node: #{node_name}")

        # Set up cookie-based authentication
        Node.set_cookie(Node.self(), String.to_atom(cookie || "vision_service_cookie"))

      {:error, reason} ->
        Logger.warning("Could not start as distributed node: #{inspect(reason)}. Running in local mode.")
    end
  end
end
