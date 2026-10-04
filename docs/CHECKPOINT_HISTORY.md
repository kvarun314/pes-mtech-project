# Checkpoint history

Every training run (Phase 1 LoRA, XLNet baseline) that produced a Drive run
directory, appended automatically by the Colab notebooks right after each
run finishes (`RunDir.history_row()`). Checkpoints themselves live on
Google Drive at the path shown — this table is the durable pointer to them,
not a copy of the weights.

| run_id | type | best | config | Drive path |
|---|---|---|---|---|
| phase1 | phase1 | step 1300, eval_loss=0.04432910308241844 | model.model_name_or_path=meta-llama/Meta-Llama-3-8B, model.use_fast_tokenizer=True, model.trust_remote_code=True, model. | `/content/drive/MyDrive/pes-mtech-project/runs/phase1` |
| xlnet | xlnet | step 4000, eval_loss=1.1575580835342407 | model=xlnet-large-cased, num_samples=16000, seed=42 | `/content/drive/MyDrive/pes-mtech-project/runs/xlnet` |
| xlnet | xlnet | step 12000, eval_loss=1.0068880319595337 | model=xlnet-large-cased, num_samples=16000, seed=42 | `/content/drive/MyDrive/pes-mtech-project/runs/xlnet` |
