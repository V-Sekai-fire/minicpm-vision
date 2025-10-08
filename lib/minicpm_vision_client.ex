defmodule MinicpmVisionClient do
  @moduledoc """
  Distributed client for connecting to MiniCPM Vision Service nodes.

  Provides a clean interface for remote vision analysis calls using
  Elixir's node-to-node communication.
  """

  require Logger
  alias MinicpmVisionService.{SimpleDescription, LanguageAnalysis}

  @doc """
  Create a connection to a vision service node.

  ## Parameters
  - `node_name`: Remote node name (e.g., :"vision_service@192.168.1.100")
  - `cookie`: Authentication cookie (defaults to "vision_service_cookie")

  ## Returns
  - `{:ok, node}` if connection successful
  - `{:error, reason}` if connection fails
  """
  @spec connect(node(), String.t()) :: {:ok, node()} | {:error, term()}
  def connect(node_name, cookie \\ "vision_service_cookie") do
    # Set the authentication cookie
    Node.set_cookie(Node.self(), String.to_atom(cookie))

    # Attempt to connect to the remote node
    case Node.connect(node_name) do
      true ->
        Logger.info("Successfully connected to vision service node: #{node_name}")
        {:ok, node_name}

      false ->
        {:error, :cannot_connect}

      :ignored ->
        {:error, :cookie_mismatch}
    end
  end

  @doc """
  Analyze an image using Base64 data on a remote node.

  ## Parameters
  - `node`: Connected remote node
  - `base64_image`: Base64 encoded image data
  - `question`: Analysis question (optional)

  ## Returns
  Map with analysis results or error tuple.
  """
  @spec analyze_image_remote(node(), String.t(), String.t()) ::
          {:ok, map()} | {:error, term()}
  def analyze_image_remote(node, base64_image, question \\ "What do you see in this image?") do
    rpc_call(node, :analyze_image, [base64_image, question])
  end

  @doc """
  Get structured image description from remote node.

  ## Parameters
  - `node`: Connected remote node
  - `base64_image`: Base64 encoded image data

  ## Returns
  SimpleDescription struct or error tuple.
  """
  @spec analyze_simple_remote(node(), String.t()) ::
          {:ok, SimpleDescription.t()} | {:error, term()}
  def analyze_simple_remote(node, base64_image) do
    rpc_call(node, :analyze_simple, [base64_image])
  end

  @doc """
  Analyze text content on remote node.

  ## Parameters
  - `node`: Connected remote node
  - `text`: Text content to analyze

  ## Returns
  LanguageAnalysis struct or error tuple.
  """
  @spec analyze_language_remote(node(), String.t()) ::
          {:ok, LanguageAnalysis.t()} | {:error, term()}
  def analyze_language_remote(node, text) do
    rpc_call(node, :analyze_language, [text])
  end

  @doc """
  Get status from remote vision service node.

  ## Parameters
  - `node`: Connected remote node

  ## Returns
  Service status map or error tuple.
  """
  @spec status_remote(node()) :: {:ok, map()} | {:error, term()}
  def status_remote(node) do
    rpc_call(node, :status, [])
  end

  @doc """
  Create ImageInput from file path on remote node.

  ## Parameters
  - `node`: Connected remote node
  - `file_path`: Path to image file on remote node

  ## Returns
  ImageInput struct or error tuple.
  """
  @spec create_image_input_remote(node(), String.t()) ::
          {:ok, MiniCPMVisionService.ImageInput.t()} | {:error, term()}
  def create_image_input_remote(node, file_path) do
    rpc_call(node, :create_image_input, [file_path])
  end

  @doc """
  Get list of available vision service nodes with given prefix.

  ## Parameters
  - `node_prefix`: Node name prefix (e.g., "vision_service@")

  ## Returns
  List of available nodes.
  """
  @spec discover_nodes(String.t()) :: [node()]
  def discover_nodes(node_prefix) do
    # In a real scenario, you'd have service discovery here
    # For now, return all connected nodes with the prefix
    nodes = Node.list()
    Enum.filter(nodes, fn node ->
      String.starts_with?(Atom.to_string(node), node_prefix)
    end)
  end

  @doc """
  Batch analyze multiple images across available nodes.

  ## Parameters
  - `node_prefix`: Node name prefix for discovery
  - `images`: List of base64 image strings

  ## Returns
  List of analysis results.
  """
  @spec batch_analyze(String.t(), [String.t()]) :: [term()]
  def batch_analyze(node_prefix, images) do
    # Discover available nodes
    nodes = discover_nodes(node_prefix)

    if nodes == [] do
      Logger.warning("No vision service nodes found with prefix: #{node_prefix}")
      []
    else
      # Distribute workload across nodes
      {results, _} = Enum.map_reduce(images, 0, fn image, idx ->
        node = Enum.at(nodes, rem(idx, length(nodes)))
        result = analyze_simple_remote(node, image)
        {result, idx + 1}
      end)

      results
    end
  end

  # Private RPC helper function
  defp rpc_call(node, function, args) do
    case :rpc.call(node, MinicpmVisionService, function, args) do
      {:badrpc, reason} ->
        {:error, {:rpc_error, reason}}

      result ->
        result
    end
  rescue
    e ->
      {:error, {:connection_error, e}}
  end
end
