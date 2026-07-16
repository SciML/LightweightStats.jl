using SciMLTesting, LightweightStats, JET, Test

run_qa(LightweightStats; explicit_imports = true, api_docs_kwargs = (; rendered = true))
