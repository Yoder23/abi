# ABI Phase 3 BPE Core V38 Attempt 1 Failure

Status: **NO SCIENTIFIC RUN — FAILED BEFORE OPTIMIZER STEP 1**

The first V38 training process wrote no candidate directory or checkpoint and
completed zero optimizer steps. PyTorch deterministic execution rejected the
first CUDA cuBLAS operation because `CUBLAS_WORKSPACE_CONFIG` was absent.

Runtime Repair V39 authorizes one retry with
`CUBLAS_WORKSPACE_CONFIG=:4096:8`. No model, data, seed, sampler, optimizer,
budget, evaluation gate, or final-test policy changes.
