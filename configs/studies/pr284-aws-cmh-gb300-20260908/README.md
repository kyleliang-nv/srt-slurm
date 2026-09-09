# InferenceMAX PR #284 AWS-CMH GB300 port

This study ports the complete 18-point performance matrix from InferenceMAX
PR #284 commit `7ae649d194874b4570533ea2c809a4c0811b1dea` to AWS-CMH. The runtime is
pinned to srt-slurm main commit `d50ee7280c33d469df8708e363e23be2456e94fb`,
which is the exact revision used by the corresponding PR workflow.

AWS-CMH currently exposes GB300 NVL72 compute rather than GB200. These points
therefore accelerate the three-arm runtime comparison but are not substitute
GB200 measurements. The only intentional platform adaptations are the GB300
hardware label, AWS-CMH Slurm settings, AWS NIC names, and shared filesystem
paths.

Each arm contains the same six one-hour AgentX points:

- TP4 resident KV at concurrency 1 and 10
- TP4 SimpleCPU KV offload at concurrency 15, 25, and 30
- TP8 resident KV at concurrency 1

The three arms are:

- `fix1-v028-flashinfer`: vLLM 0.28.0, Dynamo 1.5.0.dev20260906,
  FlashInfer draft attention
- `fix2-v0271-flashattn`: vLLM 0.27.1, Dynamo 1.3.1,
  FlashAttention draft attention
- `nightly-native`: vLLM nightly 9ea8f3ffc354, Dynamo
  1.5.0.dev20260908, native frontend/worker parser placement, FlashInfer draft
  attention

All points preserve aggregate serving, EAGLE3 K=3, `FULL_AND_PIECEWISE`, the
same AgentX trace and warmup/profile controls, and synthetic acceptance length
2.78 used by PR #284's performance comparison. The retained `router-mode=kv`
is explicitly labeled as a PR #284 routing control; with one aggregate worker
it does not select among workers.

Regenerate the files from an immutable InferenceMAX checkout with:

```bash
python3 scripts/generate_pr284_aws_cmh_gb300.py \
  --inference-max-root /path/to/InferenceMAX-at-7ae649d
```
