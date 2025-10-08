defmodule MinicpmVision.Service do
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

    @primary_key false
    embedded_schema do
      field(:content, :binary)        # Raw image bytes
      field(:format, :string, default: "jpeg")     # Image format (jpeg, png, etc.)
      field(:filename, :string)       # Original filename
      field(:metadata, :map, default: %{})        # Size, dimensions, etc.
    end

    @llm_doc """
    Image input for vision analysis containing:
    - content: Raw image bytes
    - format: Image format (jpeg, png, webp, gif)
    - filename: Original source filename
    - metadata: Image properties (width, height, file_size)
    """
  end

  defmodule VideoInput do
    use Ecto.Schema

    @primary_key false
    embedded_schema do
      field(:content, :binary)        # Raw video bytes
      field(:format, :string, default: "mp4")     # Video format (mp4, avi, etc.)
      field(:filename, :string)       # Original filename
      field(:metadata, :map, default: %{})        # Duration, fps, file_size, etc.
    end
  end

  defmodule LanguageAnalysis do
    use Ecto.Schema

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
  Create an ImageInput struct from file path.
  """
  @spec create_image_input(String.t()) :: {:ok, %ImageInput{}} | {:error, String.t()}
  def create_image_input(image_path) do
    case File.read(image_path) do
      {:ok, binary} ->
        filename = Path.basename(image_path)
        format = Path.extname(image_path) |> String.trim_leading(".") |> String.downcase()

        metadata = %{
          file_size: byte_size(binary),
          filename: filename
        }

        image_input = %ImageInput{
          content: binary,
          format: format,
          filename: filename,
          metadata: metadata
        }

        {:ok, image_input}

      {:error, reason} ->
        {:error, "Could not read image file: #{reason}"}
    end
  end

  @doc """
  Analyze image content using MiniCPM.
  Accepts single image (%ImageInput{}) or list of images ([%ImageInput{}]).
  Returns single analysis response map or list of maps accordingly.
  """
  @spec analyze_image(%ImageInput{} | [%ImageInput{}], String.t()) :: {:ok, map() | [map()]} | {:error, String.t()}
  def analyze_image(image_input, question) do
    analyze_image(image_input, question, %{})
  end

  @spec analyze_image(%ImageInput{} | [%ImageInput{}], String.t(), map()) :: {:ok, map() | [map()]} | {:error, String.t()}
  def analyze_image(image_input, question, opts) do
    case image_input do
      %ImageInput{} ->
        # Single image
        GenServer.call(@name, {:analyze_single_image, image_input.content, question}, 30_000)
      [%ImageInput{} | _] = images ->
        batch_size = opts[:batch_size] || 10
        if length(images) <= batch_size do
          # Small batch - process all at once
          image_binaries = Enum.map(images, & &1.content)
          GenServer.call(@name, {:analyze_multiple_images, image_binaries, question}, 120_000)
        else
          # Large batch - process in chunks
          GenServer.call(@name, {:analyze_batch_images, images, question, batch_size}, 300_000 * (length(images) / batch_size))
        end
      _ ->
        {:error, "Input must be %ImageInput{} or [%ImageInput{}]"}
    end
  end

  @doc """
  Analyze text content with language interpretation.
  Returns analysis as any free-form content.
  """
  @spec analyze_language(String.t()) :: {:ok, any()} | {:error, String.t()}
  def analyze_language(text) do
    prompt = "Analyze this text content: provide a summary, key insights, interpretation, and context notes."
    GenServer.call(@name, {:analyze_language, text, prompt}, 30_000)
  end

  def analyze_video(binary_video, question, fps, force_packing) do
    GenServer.call(@name, {:analyze_video, binary_video, question, fps, force_packing}, 30_000)
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
  def handle_call({:analyze_single_image, binary_image, question}, _from, state) do
    case run_analysis(binary_image, question, state) do
      {:ok, result, updated_state} -> {:reply, {:ok, result}, updated_state}
      {:error, reason} -> {:reply, {:error, reason}, state}
    end
  end

  @impl true
  def handle_call({:analyze_multiple_images, image_binaries, question}, _from, state) do
    case run_multiple_analysis(image_binaries, question, state) do
      {:ok, results, updated_state} -> {:reply, {:ok, results}, updated_state}
      {:error, reason} -> {:reply, {:error, reason}, state}
    end
  end

  @impl true
  def handle_call({:analyze_batch_images, images, question, batch_size}, _from, state) do
    case run_batch_analysis(images, question, batch_size, state) do
      {:ok, results, updated_state} -> {:reply, {:ok, results}, updated_state}
      {:error, reason} -> {:reply, {:error, reason}, state}
    end
  end

  # Legacy support - redirect to new single image handler
  @impl true
  def handle_call({:analyze_image, binary_image, question}, _from, state) do
    case run_analysis(binary_image, question, state) do
      {:ok, result, updated_state} -> {:reply, {:ok, result}, updated_state}
      {:error, reason} -> {:reply, {:error, reason}, state}
    end
  end

  @impl true
  def handle_call({:analyze_video, binary_video, question, fps, force_packing}, _from, state) do
    case run_video_analysis(binary_video, question, fps, force_packing, state) do
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

  defp run_batch_analysis(images, question, batch_size, state) do
    try do
      total_images = length(images)
      num_batches = ceil(total_images / batch_size)

      Logger.info("Starting batch analysis: #{total_images} images in #{num_batches} batches of #{batch_size}")

      start_time = DateTime.utc_now()

      {results, final_state, errors} = Enum.reduce(1..num_batches, {[], state, []}, fn batch_num, {results_acc, state_acc, errors_acc} ->
        batch_start = (batch_num - 1) * batch_size
        batch_end = min(batch_num * batch_size, total_images)
        batch_images = Enum.slice(images, batch_start, batch_end - batch_start)

        Logger.info("Processing batch #{batch_num}/#{num_batches} (#{length(batch_images)} images)")

        image_binaries = Enum.map(batch_images, & &1.content)
        filenames = Enum.map(batch_images, & &1.filename)

        case run_multiple_analysis(image_binaries, question, state_acc) do
          {:ok, batch_result, updated_state} ->
            # Enhance batch result with filenames and position info
            enhanced_result = batch_result
            |> Map.put("batch_number", batch_num)
            |> Map.put("total_batches", num_batches)
            |> Map.put("filenames", filenames)
            |> Map.put("image_range", "#{batch_start + 1}-#{batch_end}")

            {results_acc ++ [enhanced_result], updated_state, errors_acc}

          {:error, reason} ->
            # Log the error but continue with other batches
            Logger.error("Batch #{batch_num} failed: #{reason}")
            error_result = %{
              "batch_number" => batch_num,
              "total_batches" => num_batches,
              "filenames" => filenames,
              "image_range" => "#{batch_start + 1}-#{batch_end}",
              "error" => reason,
              "success" => false,
              "analyzed_at" => DateTime.utc_now()
            }
            {results_acc ++ [error_result], state_acc, errors_acc ++ [batch_num]}
        end
      end)

      end_time = DateTime.utc_now()

      # Summary response
      summary = %{
        "batch_summary" => %{
          "total_images" => total_images,
          "num_batches" => num_batches,
          "batch_size" => batch_size,
          "successful_batches" => num_batches - length(errors),
          "failed_batches" => length(errors),
          "failed_batch_numbers" => errors,
          "start_time" => start_time,
          "end_time" => end_time,
          "total_time_seconds" => DateTime.diff(end_time, start_time, :microsecond) / 1_000_000
        },
        "results" => results,
        "success" => length(errors) == 0,
        "analyzed_at" => end_time
      }

      Logger.info("Batch analysis completed: #{total_images} images, #{num_batches - length(errors)} successful batches, #{length(errors)} failed batches")
      {:ok, summary, final_state}

    rescue
      e ->
        Logger.error("Batch analysis failed: #{inspect(e)}")
        {:error, "Batch analysis error: #{inspect(e)}"}
    end
  end

  defp run_multiple_analysis(image_binaries, question, state) do
    try do
      Logger.debug("Running multiple image analysis with MiniCPM")

      # Use EEx template for dynamic Python code generation with multiple images
      python_code = EEx.eval_file("lib/templates/multiple_image_analysis.eex", [])
      encoded_image_binaries = Enum.map(image_binaries, &Pythonx.encode!/1)
      encoded_question = Pythonx.encode!(question)
      python_globals = state.python_globals
      python_globals = Map.put(python_globals, "question", encoded_question)
      python_globals = Map.put(python_globals, "image_bytes_list", encoded_image_binaries)
      {result_string, updated_globals} = Pythonx.eval(python_code, python_globals)

      description = Pythonx.decode(result_string)

      analysis_response = %{
        "description" => description,
        "num_images" => length(image_binaries),
        "success" => true,
        "analyzed_at" => DateTime.utc_now()
      }

      Logger.info("Multiple image analysis completed successfully")
      {:ok, analysis_response, Map.put(state, :python_globals, updated_globals)}

    rescue
      e ->
        Logger.error("Multiple Image Analysis failed: #{inspect(e)}")
        {:error, "Multiple Image Analysis error: #{inspect(e)}"}
    end
  end

  defp run_video_analysis(binary_video, question, fps, force_packing, state) do
    try do
      Logger.debug("Running video analysis with MiniCPM")

      # Use EEx template for dynamic Python code generation
      python_code = EEx.eval_file("lib/templates/video_analysis.eex", [])
      encoded_binary_video = Pythonx.encode!(binary_video)
      encoded_question = Pythonx.encode!(question)
      encoded_fps = Pythonx.encode!(fps)
      encoded_force_packing = Pythonx.encode!(force_packing)
      python_globals = state.python_globals
      python_globals = Map.put(python_globals, "question", encoded_question)
      python_globals = Map.put(python_globals, "video_bytes", encoded_binary_video)
      python_globals = Map.put(python_globals, "fps", encoded_fps)
      python_globals = Map.put(python_globals, "force_packing", encoded_force_packing)
      {result_string, updated_globals} = Pythonx.eval(python_code, python_globals)

      description = Pythonx.decode(result_string)

      analysis_response = %{
        "description" => description,
        "success" => true,
        "analyzed_at" => DateTime.utc_now()
      }

      Logger.info("Video analysis completed successfully")
      {:ok, analysis_response, Map.put(state, :python_globals, updated_globals)}

    rescue
      e ->
        Logger.error("Video Analysis failed: #{inspect(e)}")
        {:error, "Video Analysis error: #{inspect(e)}"}
    end
  end

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
        "analyzed_at" => DateTime.utc_now()
      }

      Logger.info("Image analysis completed successfully")
      {:ok, analysis_response, Map.put(state, :python_globals, updated_globals)}

    rescue
      e ->
        Logger.error("Image Analysis failed: #{inspect(e)}")
        {:error, "Image Analysis error"}
    end
  end
end
