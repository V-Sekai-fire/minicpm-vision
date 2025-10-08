defmodule MiniCPMVisionService do
  @moduledoc """
  MiniCPM Vision Service - Multi-modal analysis using MiniCPM.

  Provides image analysis (as subset of vision capabilities) and language/text interpretation.
  Images are one modality within a broader vision analysis framework.
  Uses GenServer for model lifecycle management.
  """

  use GenServer
  require Logger

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
  @spec analyze_image(String.t(), String.t()) :: {:ok, map()} | {:error, String.t()}
  def analyze_image(base64_image, question) do
    GenServer.call(@name, {:analyze_image, base64_image, question}, 30_000)
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

  @doc """
  Create an ImageInput struct from file path.
  """
  @spec create_image_input(String.t()) :: {:ok, %ImageInput{}} | {:error, String.t()}
  def create_image_input(image_path) do
    case File.read(image_path) do
      {:ok, binary} ->
        base64_content = Base.encode64(binary)

        filename = Path.basename(image_path)
        format = Path.extname(image_path) |> String.trim_leading(".") |> String.downcase()

        metadata = %{
          file_size: byte_size(binary),
          filename: filename
        }

        image_input = %ImageInput{
          content: base64_content,
          format: format,
          filename: filename,
          metadata: metadata
        }

        {:ok, image_input}

      {:error, reason} ->
        {:error, "Could not read image file: #{reason}"}
    end
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
  def handle_call({:analyze_image, base64_image, question}, _from, state) do
    case run_analysis(base64_image, question, state) do
      {:ok, result} -> {:reply, {:ok, result}, state}
      {:error, reason} -> {:reply, {:error, reason}, state}
    end
  end

  @impl true
  def handle_call({:analyze_simple, base64_image, prompt}, _from, state) do
    case run_simple_analysis(base64_image, prompt, state) do
      {:ok, result} -> {:reply, {:ok, result}, state}
      {:error, reason} -> {:reply, {:error, reason}, state}
    end
  end

  @impl true
  def handle_call({:analyze_language, text, prompt}, _from, state) do
    case run_language_analysis(text, prompt, state) do
      {:ok, result} -> {:reply, {:ok, result}, state}
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

      # Test pythonx basic functionality
      {test_result, _} = Pythonx.eval("print('pythonx ready')", %{})
      Logger.info("Pythonx initialized: #{inspect(test_result)}")

      # Load MiniCPM model (this is expensive, done once)
      Logger.info("Loading MiniCPM model (this may take a minute)...")

      model_load_result = model_initialization_code()

      case model_load_result do
        {result, _globals} ->
          # Check if result indicates success
          result_str = to_string(result)
          if String.contains?(String.downcase(result_str), "loaded") or String.contains?(result_str, "GPU") do
            Logger.info("MiniCPM model loaded successfully")

            state = %{
              model_loaded_at: DateTime.utc_now(),
              start_time: :os.system_time(:millisecond),
              device: "cuda",
              model_path: "huihui-ai/Huihui-MiniCPM-V-4_5-abliterated"
            }

            {:ok, state}
          end
      end
    rescue
      e ->
        Logger.error("Failed to initialize model: #{inspect(e)}")
        {:error, "Model initialization error"}
    end
  end

  defp run_analysis(base64_image, question, _state) do
    try do
      Logger.debug("Running image analysis with MiniCPM")

      analysis_code = """
import base64
from PIL import Image
import io

# Decode base64 image in fresh python environment
image_bytes = base64.b64decode('#{String.replace(base64_image, "'", "\\'")}')
image = Image.open(io.BytesIO(image_bytes)).convert('RGB')

# Ask the loaded model directly
question = '#{String.replace(question, "'", "\\'")}'
msgs = [{'role': 'user', 'content': [image, question]}]

# Note: In persistent scope, model/tokenizer would be available from model_initialization_code
# For now, we reload in the same eval (trade-off between complexity and performance)
import torch
from transformers import AutoModel, AutoTokenizer

model_path = 'huihui-ai/Huihui-MiniCPM-V-4_5-abliterated'
model = AutoModel.from_pretrained(model_path, trust_remote_code=True, attn_implementation='sdpa', torch_dtype=torch.bfloat16)
model = model.eval().cuda()
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

answer = model.chat(msgs=msgs, image=image, tokenizer=tokenizer)
str(answer)
"""

      {result, _globals} = Pythonx.eval(analysis_code, %{})

      analysis_response = %{
        "description" => to_string(result),
        "objects" => [],
        "colors" => [],
        "success" => true,
        "raw_response" => to_string(result),
        "analyzed_at" => DateTime.utc_now()
      }

      Logger.info("Image analysis completed successfully")
      {:ok, analysis_response}

    rescue
      e ->
        Logger.error("Analysis failed: #{inspect(e)}")
        {:error, "Analysis error: #{inspect(e)}"}
    end
  end

  defp run_simple_analysis(base64_image, prompt, _state) do
    try do
      Logger.debug("Running simple structured analysis")

      # Get the raw MiniCPM response first
      case run_analysis(base64_image, prompt, %{}) do
        {:ok, basic_result} ->
          raw_response = basic_result["raw_response"]

          # Use Instructor with SimpleDescription schema
          case Instructor.chat_completion(%{
            messages: [
              %{
                role: "system",
                content: "Parse this vision analysis into a simple description with what you see, main colors, and overall feeling."
              },
              %{
                role: "user",
                content: raw_response
              }
            ]
          }, response_model: SimpleDescription, max_retries: 1) do
            {:ok, structured_result} ->
              Logger.info("Simple structured analysis complete")
              {:ok, structured_result}

            {:error, parse_error} ->
              Logger.warning("Simple analysis parsing failed: #{inspect(parse_error)}")
              {:error, "Failed to parse response: #{inspect(parse_error)}"}
          end

        error ->
          error
      end

    rescue
      e ->
        Logger.error("Simple analysis failed: #{inspect(e)}")
        {:error, "Simple analysis error: #{inspect(e)}"}
    end
  end

  defp run_language_analysis(text, prompt, _state) do
    try do
      Logger.debug("Running language analysis")

      # Use MiniCPM for text analysis (text-only, no image)
      analysis_code = """
# Use MiniCPM for text analysis
question = '#{String.replace(prompt, "'", "\\'")}'
text_content = '#{String.replace(text, "'", "\\'")}'
msgs = [{'role': 'user', 'content': text_content + '\\n\\n' + question}]

# Load model for text analysis
import torch
from transformers import AutoModel, AutoTokenizer

model_path = 'huihui-ai/Huihui-MiniCPM-V-4_5-abliterated'
model = AutoModel.from_pretrained(model_path, trust_remote_code=True, attn_implementation='sdpa', torch_dtype=torch.bfloat16)
model = model.eval().cuda()
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

answer = model.chat(msgs=msgs, tokenizer=tokenizer)
str(answer)
"""

      {result, _globals} = Pythonx.eval(analysis_code, %{})

      raw_response = to_string(result)

      # Use Instructor with LanguageAnalysis schema
      case Instructor.chat_completion(%{
        messages: [
          %{
            role: "system",
            content: "Parse this text analysis into summary, key insights, interpretation, and context notes."
          },
          %{
            role: "user",
            content: raw_response
          }
        ]
      }, response_model: LanguageAnalysis, max_retries: 1) do
        {:ok, structured_result} ->
          Logger.info("Language analysis complete")
          {:ok, structured_result}

        {:error, parse_error} ->
          Logger.warning("Language analysis parsing failed: #{inspect(parse_error)}")
          {:error, "Failed to parse language response: #{inspect(parse_error)}"}
      end

    rescue
      e ->
        Logger.error("Language analysis failed: #{inspect(e)}")
        {:error, "Language analysis error: #{inspect(e)}"}
    end
  end

  defp model_initialization_code do
    """
import torch
from PIL import Image
from transformers import AutoModel, AutoTokenizer
import base64
import io

# Initialize MiniCPM exactly as in the docs
model_path = 'huihui-ai/Huihui-MiniCPM-V-4_5-abliterated'
model = AutoModel.from_pretrained(model_path, trust_remote_code=True, attn_implementation='sdpa', torch_dtype=torch.bfloat16)
model = model.eval().cuda()
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

'MiniCPM model loaded on GPU'
"""
    |> Pythonx.eval(%{})
  end
end
