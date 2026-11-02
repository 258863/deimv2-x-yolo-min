# Remote acceptance — 2026-11-02

Status: User accepted the measured one-hit FP16 regression and authorized publishing the tools with this record. This is not acceptance of the engine as a production replacement.

Host: Windows RTX3090 Ti, torch2.7.0+cu126, TensorRT10.13.3.9.
Job: deimv2-repo-check-20261102-095915.
Independent remote directory: F:/remote-training/deimv2-repo-check-20261102-095915.

- All 15 repository tests passed.
- Layout rewrite passed four fixed input ORT equivalence checks.
- Rewritten ONNX SHA256 exactly matches the previously validated graph: bc85972a6aa7a64f4804c4759eabc9b3951188b375210f63e0fd62b5faea1ea5.
- CLI FP16 engine build and input-tensor inference checks completed.
- FP32-protected layer list exactly matches all 121 previous entries.
- Full 387-image evaluation completed: 286/368 image/class hits (77.7174%), compared with previous FP16 engine 287/368. Confidence 0.5, no IoU requirement; empty labels excluded from denominator.
- Only image 0243 changed predicted class set: lx score 0.5020751953125 -> 0.4993896782398224, crossing the fixed threshold.
- New engine SHA256: 483484b202f2a6d9e6036278a9987aa5c2e9d18931b5ad64d4108052cfff880f.
- Build took 221.14 seconds. Pipeline measured 14.19ms / 70.49FPS; a single run, not evidence of a stable speedup over older benchmarks.

The validation runner deliberately failed its >=287 hit acceptance assertion after evaluation. No engine was deployed and no threshold was changed. The user subsequently accepted this result and authorized pushing the repository tools with the limitation documented. The graph and precision policy match; the exact source of the rebuilt engine's numerical difference is not yet established. It must not be described as proven FP16 rounding or a proven code migration defect.

Checkpoint export and strict FP32 CLI build were not exercised in this remote run. The existing validated artifacts remain intact.
