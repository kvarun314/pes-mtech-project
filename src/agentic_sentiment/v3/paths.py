"""All model and artifact paths for Phase 2 v3 (repo-root relative)."""
D = "v2_reflective/e23/"
MLX_BASE = "v2_reflective/mlx_model"
ADAPTER_R = D + "p1/adapter_R"
INSTRUCT = ".scratch/mlx-llama3-8b-instruct-q4"
AGENTS = D + "agents/"            # joblib files written by persist_agents.py (gitignored)
EVIDENCE_JOBLIB = AGENTS + "e6b.joblib"
PROBE_JOBLIB = AGENTS + "p3.joblib"
REFLECTOR_JOBLIB = AGENTS + "reflector.joblib"
PROBE_PREFIX = "evid7"            # becomes "evid8" only if gate A1 passes (LOG.md)
B2 = D + "b2/"                    # rationales and packs, gitignored
JUDGE2 = ".scratch/mlx-qwen2.5-7b-instruct-q4"   # Q3 second judge (LOG.md 2026-10-28)
