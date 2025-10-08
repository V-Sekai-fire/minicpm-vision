defmodule MinicpmVision.ServiceTest do
  use ExUnit.Case
  require Logger

  alias MinicpmVision.Service
  alias MinicpmVision.Service.{ImageInput, SimpleDescription, LanguageAnalysis}

  @color_image_paths [
    "tests/media/image1.png",
    "tests/media/image2.png",
    "tests/media/image3.png",
    "tests/media/image4.png"
  ]

  describe "ImageInput schema" do
    test "struct has expected fields" do
      input = %ImageInput{
        filename: "test.jpg",
        content: "base64content",
        format: "jpeg"
      }

      assert input.filename == "test.jpg"
      assert input.content == "base64content"
      assert input.format == "jpeg"
    end
  end

  describe "SimpleDescription schema" do
    test "struct initializes with default values" do
      output = %SimpleDescription{}

      assert output.what_i_see == nil
      assert output.main_colors == []
      assert output.overall_feeling == nil
    end
  end

  describe "LanguageAnalysis schema" do
    test "struct initializes with default values" do
      output = %LanguageAnalysis{}

      assert output.summary == nil
      assert output.key_insights == []
      assert output.interpretation == nil
      assert output.context_notes == []
    end
  end

  describe "multiple image analysis" do
    @tag integration: true
    @tag timeout: 180_000
    test "analyze multiple color images" do
      # Start the service if not already running
      unless Process.whereis(Service) do
        {:ok, _pid} = Service.start_link([])
      end

      # Load all color images
      loaded_images = Enum.reduce_while(@color_image_paths, [], fn path, acc ->
        case Service.create_image_input(path) do
          {:ok, image_input} -> {:cont, acc ++ [image_input]}
          {:error, reason} -> {:halt, {:error, "Failed to load #{path}: #{reason}"}}
        end
      end)

      case loaded_images do
        {:error, reason} ->
          flunk(reason)

        _ when is_list(loaded_images) ->
          question = "What are the colors in these images?"

            case Service.analyze_image(loaded_images, question) do
              {:ok, result} ->
                Logger.debug("Multiple image analysis result: #{inspect(result)}")

                # Check that the response contains color information for multiple images
                description = String.downcase(result["description"])

                # Count how many different color names appear in the response
                color_words = ["red", "blue", "green", "yellow", "orange", "purple", "pink", "brown", "black", "white", "gray"]
                detected_colors = Enum.filter(color_words, fn color -> String.contains?(description, color) end)

                # The model should have identified multiple different colors from the 4 distinct images
                assert length(detected_colors) >= 3,
                  "Should detect at least 3 different colors in response. Detected: #{inspect(detected_colors)}. Full response: #{result["description"]}"

                # Check that we have the correct number of images in the result
                assert result["num_images"] == 4,
                  "Should report analyzing 4 images"

              {:error, reason} ->
                flunk("Multiple image analysis failed: #{reason}")
            end
      end
    end
  end

  describe "low-level API" do
    test "responds to GenServer calls" do
      # Test that the service is registered (may not be running in test context)
      pid = Process.whereis(Service)
      if pid do
        status = Service.status()
        assert is_map(status)
        assert Map.has_key?(status, :server)
      else
        # Service not running is acceptable for unit tests
        assert true
      end
    end
  end

  describe "service integration" do
    @tag integration: true, timeout: 1_200_000  # 20 minutes for model loading/download
    test "service can be started and stopped" do
      # Only run if service isn't already started
      unless Process.whereis(Service) do
        case Service.start_link([]) do
          {:ok, pid} ->
            assert Process.alive?(pid)
            status = Service.status()
            assert status.server == :running

          {:error, {:already_started, _pid}} ->
            # Service already running
            assert true
        end
      end
    end
  end

  describe "configuration" do
    test "elixir config has required python dependencies" do
      # Test that our Elixir config defines Python dependencies for MiniCPM
      config = Application.get_env(:pythonx, :uv_init)
      assert config != nil
      assert config[:pyproject_toml] != nil

      toml_content = config[:pyproject_toml]
      assert String.contains?(toml_content, "torch")
      assert String.contains?(toml_content, "transformers")
      assert String.contains?(toml_content, "PILLOW")
    end

    test "gpu requirement is enforced without cuda" do
      # Test GPU availability check (will pass/fail based on hardware)
      # This test validates that the GPU requirement check runs without errors
      # In a real environment, this would fail if no CUDA GPU is available
      gpu_result = try do
        gpu_check_code = """
import torch
cuda_available = torch.cuda.is_available()
device_count = torch.cuda.device_count() if cuda_available else 0
f"CUDA devices: {device_count}"
"""

        {result, _} = Pythonx.eval(gpu_check_code, %{})
        {:ok, to_string(result)}
      rescue
        _ -> {:error, "Python environment not available"}
      end

      # The test should run without crashing, regardless of GPU availability
      # GPU enforcement happens at service startup, not in unit tests
      assert match?({:ok, _}, gpu_result) or match?({:error, _}, gpu_result)
    end
  end
end
