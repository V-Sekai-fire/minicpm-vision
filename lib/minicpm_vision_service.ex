defmodule MiniCPMVisionService do
  @moduledoc """
  MiniCPM Vision Service - Multi-modal analysis using MiniCPM.

  Provides image analysis (as subset of vision capabilities) and language/text interpretation.
  Images are one modality within a broader vision analysis framework.
  Uses GenServer for model lifecycle management.
  """

  use GenServer
  require Logger
  require EEx

  @name __MODULE__

  # Input Schema - Define what we send to the vision service
  defmodule ImageInput do
    use Ecto.Schema
    use Instructor

    @primary_key false
    embedded_schema do
      field(:content, :string)        # Base64 encoded image
      field(:format, :string, default: "jpeg")     # Image format (jpeg, png, etc.)
      field(:filename, :string)       # Original filename
      field(:metadata, :map, default: %{})        # Size, dimensions, etc.
    end

    @llm_doc """
    Image input for vision analysis containing:
    - content: Base64 encoded image data
    - format: Image format (jpeg, png, webp, gif)
    - filename: Original source filename
    - metadata: Image properties (width, height, file_size)
    """
  end

  # Output Schemas - Define structured results from vision service
  defmodule SimpleDescription do
    use Ecto.Schema
    use Instructor

    @primary_key false
    embedded_schema do
      field(:what_i_see, :string)
      field(:main_colors, {:array, :string}, default: [])
      field(:overall_feeling, :string)
    end

    @llm_doc """
    Simple description of an image containing:
    - what_i_see: A brief description of what's visible
    - main_colors: Primary colors observed
    - overall_feeling: General impression or mood
    """
  end

  defmodule LanguageAnalysis do
    use Ecto.Schema
    use Instructor

    @primary_key false
    embedded_schema do
      field(:summary, :string)              # Brief summary of the analysis
      field(:key_insights, {:array, :string}, default: [])      # Important observations
      field(:interpretation, :string)       # Creative or analytical interpretation
      field(:context_notes, {:array, :string}, default: [])     # Additional context or notes
    end

    @llm_doc """
    Language-based analysis and interpretation containing:
    - summary: Concise overview of the main findings
    - key_insights: Important observations and discoveries
    - interpretation: Creative or analytical interpretation
    - context_notes: Additional context or relevant notes
    """
  end



  # Client API

  @doc """
  Start the MiniCPM Vision Service.
  """
  def start_link(opts \\ []) do
    GenServer.start_link(__MODULE__, opts, name: @name)
  end

  @doc """
  Analyze an image with Base64 data and question.
  Returns basic map response.
  """
  @spec analyze_image(term(), String.t()) :: {:ok, map()} | {:error, String.t()}
  def analyze_image(binary_image, question) do
    GenServer.call(@name, {:analyze_image, binary_image, question}, 30_000)
  end

  @doc """
  Analyze an image with structured SimpleDescription output.
  Returns SimpleDescription struct.
  """
  @spec analyze_simple(String.t()) :: {:ok, %SimpleDescription{}} | {:error, String.t()}
  def analyze_simple(base64_image) do
    prompt = "Describe this image simply: what you see, main colors, and overall feeling."
    GenServer.call(@name, {:analyze_simple, base64_image, prompt}, 30_000)
  end

  @doc """
  Analyze text content with language interpretation.
  Returns LanguageAnalysis struct.
  """
  @spec analyze_language(String.t()) :: {:ok, %LanguageAnalysis{}} | {:error, String.t()}
  def analyze_language(text) do
    prompt = "Analyze this text content: provide a summary, key insights, interpretation, and context notes."
    GenServer.call(@name, {:analyze_language, text, prompt}, 30_000)
  end

  @doc """
  Get server status and model information.
  """
  @spec status() :: map()
  def status do
    GenServer.call(@name, :status)
  end

  # Server Callbacks

  @impl true
  def init(_opts) do
    Logger.info("Starting MiniCPM Vision Service...")

    # Initialize the pythonx runtime and load model
    case initialize_model() do
      {:ok, state} ->
        Logger.info("MiniCPM Vision Service started successfully")
        {:ok, state}

      {:error, reason} ->
        Logger.error("Failed to initialize MiniCPM: #{inspect(reason)}")
        {:stop, reason}
    end
  end

  @impl true
  def handle_call({:analyze_image, binary_image, question}, _from, state) do
    case run_analysis(binary_image, question, state) do
      {:ok, result, updated_state} -> {:reply, {:ok, result}, updated_state}
      {:error, reason} -> {:reply, {:error, reason}, state}
    end
  end

  @impl true
  def handle_call(:status, _from, state) do
    status = %{
      server: :running,
      model_loaded: true,
      device: state.device,
      model_path: state.model_path,
      uptime: :os.system_time(:millisecond) - state.start_time
    }
    {:reply, status, state}
  end

  @impl true
  def handle_info(msg, state) do
    Logger.warning("Unhandled message: #{inspect(msg)}")
    {:noreply, state}
  end

  # Private functions



  defp initialize_model do
    try do
      Logger.info("Initializing pythonx and loading MiniCPM model...")

      # Single consolidated Python initialization script using EEx template
      python_script = EEx.eval_file("lib/templates/initialization.eex", [])
      {result, python_globals} = Pythonx.eval(python_script, %{})

      # Verify initialization succeeded (decode from Pythonx.Object to string)
      result_string = Pythonx.decode(result)

      case result_string do
        "initialization_complete" ->
          Logger.info("MiniCPM Vision Service initialized successfully with Python globals")

          state = %{
            model_loaded_at: DateTime.utc_now(),
            start_time: :os.system_time(:millisecond),
            device: "cuda",
            model_path: "huihui-ai/Huihui-MiniCPM-V-4_5-abliterated",
            python_globals: python_globals
          }

          {:ok, state}

        _ ->
          Logger.error("Unexpected initialization result: #{inspect(result_string)}")
          {:error, "Model initialization failed"}
      end

    rescue
      e ->
        Logger.error("Failed to initialize model: #{inspect(e)}")
        {:error, "Model initialization error"}
    end
  end

  defp run_analysis(binary_image, question, state) do
    try do
      Logger.debug("Running image analysis with MiniCPM")

      # Use EEx template for dynamic Python code generation
      python_code = EEx.eval_file("lib/templates/image_analysis.eex", [])
      encoded_binary_image = Pythonx.encode!(binary_image)
      encoded_question = Pythonx.encode!(question)
      python_globals = state.python_globals
      python_globals = Map.put(python_globals, "question", encoded_question)
      python_globals = Map.put(python_globals, "image_bytes", encoded_binary_image)
      {result_string, updated_globals} = Pythonx.eval(python_code, python_globals)
      
      description = Pythonx.decode(result_string)

      analysis_response = %{
        "description" => description,
        "objects" => [],
        "colors" => [],
        "success" => true,
        "raw_response" => description,
        "analyzed_at" => DateTime.utc_now()
      }

      Logger.info("Image analysis completed successfully")
      {:ok, analysis_response, Map.put(state, :python_globals, updated_globals)}

    rescue
      e ->
        Logger.error("Image Analysis failed: #{inspect(e)}")
        {:error, "Image Analysis error: #{inspect(e)}"}
    end
  end
end
