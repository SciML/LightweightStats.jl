using SciMLTesting, LightweightStats, JET

run_qa(LightweightStats; jet_kwargs = (; mode = :basic))
