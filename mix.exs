defmodule MinicpmVision.MixProject do
  use Mix.Project

  def project do
    [
      app: :minicpm_vision,
      version: "0.1.0",
      elixir: "~> 1.14",
      start_permanent: Mix.env() == :prod,
      deps: deps()
    ]
  end

  def application do
    [
      extra_applications: [:logger]
    ]
  end

  defp deps do
    [
      {:ecto_sql, "~> 3.10"},
      {:jaxon, "~> 2.0"},
      {:pythonx, "~> 0.4.7"},
    ]
  end
end
