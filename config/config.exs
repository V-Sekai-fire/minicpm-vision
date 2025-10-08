import Config

# pythonx configuration for MiniCPM dependencies
config :pythonx, :uv_init,
  pyproject_toml: """
[project]
name = "minicpm-vision"
version = "0.0.0"
requires-python = "==3.11.*"
dependencies = [
  "torch>=2.0.0",
  "torchvision",
  "decord",
  "transformers>=4.36.0",
  "PILLOW>=10.0.0",
  "accelerate",
  "bitsandbytes",
  "scipy",
  "protobuf"
]
"""