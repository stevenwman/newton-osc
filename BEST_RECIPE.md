# Best FlashSAC peg-insertion recipe (UTD sweep result)

Task SOLVED at 100% success. The dominant lever is UTD (gradient updates per
env-step); parallel envs and buffer size are NOT levers (convergence is grad-step
bound, buffer already phase-saturated at ~17 episodes).

## BEST recipe — UTD 64 (best sample efficiency)
```
PYTHONPATH=. .venv/bin/python train_peg_flashsac.py \
    --num-envs 128 --total-steps 2500000 --episode-length 450 \
    --batch-size 2048 --grad-updates 64 --gamma 0.99 --min-buffer 10000 \
    --action-mode delta --outdir runs/<name>
```
- obs 42-d (joint state + peg pose + EE linvel/angvel + last/prev action)
- UTD = grad/N = 64/128 = 0.5 ; replay ratio = grad*batch/N = 1024
- **90% success at ~448k env-steps** (~1.85x better than jax_rl's ~830k), 100% by ~0.5-0.6M
- deterministic eval ep_ret ~2860-3000

## UTD sweep (N=128, batch 2048, obs 42-d) — env-steps to 90% succ
| UTD knob (grad) | UTD=grad/N | ratio | env-steps to 90% | grad-steps to 90% | env-sps |
|---|---|---|---|---|---|
| 16 (P2)  | 0.125 | 256  | 1.47M | 184k | 960 |
| 32 (P4)  | 0.25  | 512  | 704k  | 176k | 625 |
| 64 (P5)  | 0.5   | 1024 | **448k** | 224k | 390 |

## Negative results (don't repeat)
- N=512 batch 8192 grad 16 (P3): UNDERTRAINED (732 ep_ret, 0% succ) — fixed grad at
  high N => 4x fewer optimizer steps. Convergence is grad-step bound.
- N=256 grad 64 (P6, UTD 0.25 matched P4): ~neutral (768k vs 704k), small wall gain only.
- N=256 grad 128 (P7, UTD 0.5 matched P5): WORSE per grad-step (775 ep_ret at 192k
  grad-steps vs P5's ~2300) — scaling envs+grad together regresses; bigger-N cohorts
  make big grad batches more correlated/redundant.
- Wall-time to 90% is ~flat (~17-19 min) across UTD because it's grad-step bound;
  higher UTD buys SAMPLE efficiency (fewer env-steps), not wall-time.
