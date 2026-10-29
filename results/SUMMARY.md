# MixSched results

Trace replay of the public Acme job logs (Hu et al., NSDI '24) on a simulated cluster: real job sizes, types, submit times and runtimes; simulated placement, queueing, sharing and preemption. The inference fleet is synthetic (Acme has none): a diurnal demand of 115-384 GPUs on its own 48-node pool. Effective utilization counts GPUs doing work: training/eval/debug plus inference replicas that serve actual demand (warm idle replicas excluded). See docs/DESIGN.md for the model.

## Seren, days 112-126: 164,525 GPU jobs

| general nodes | policy | GPU allocation | effective utilization | fragmentation | eval queue p50 / p99 | training queue p50 / p99 | training JCT slowdown (mean) | inference SLO | GPU-h lost to preemption |
|---|---|---|---|---|---|---|---|---|---|
| 286 | fifo | 68.4% | 63.4% | 0.067 | 0 s / 0 s | 0 s / 0 s | 1.00 | 100.00% | 0 |
| 286 | binpack | 68.4% | 63.4% | 0.054 | 0 s / 0 s | 0 s / 0 s | 1.00 | 100.00% | 0 |
| 286 | kueue | 63.4% | 63.3% | 0.560 | 0 s / 3.1 min | 0 s / 13.3 min | 3.23 | 99.65% | 0 |
| 286 | mixsched | 63.0% | 62.9% | 0.053 | 0 s / 0 s | 0 s / 0 s | 1.00 | 100.00% | 0 |
| 240 | fifo | 79.4% | 73.5% | 0.170 | 0 s / 15.1 min | 0 s / 1.6 min | 1.24 | 100.00% | 0 |
| 240 | binpack | 79.4% | 73.5% | 0.135 | 0 s / 0 s | 0 s / 2 s | 1.07 | 100.00% | 0 |
| 240 | kueue | 72.7% | 72.7% | 0.677 | 0 s / 17.3 min | 0 s / 40.7 min | 6.04 | 99.65% | 47 |
| 240 | mixsched | 73.0% | 73.0% | 0.098 | 0 s / 0 s | 0 s / 0 s | 1.02 | 100.00% | 0 |
| 200 | fifo | 88.1% | 81.3% | 0.523 | 5.6 h / 15.6 h | 7.4 min / 15.8 h | 425.02 | 100.00% | 0 |
| 200 | binpack | 89.3% | 82.5% | 0.388 | 18 s / 2.9 h | 0 s / 3.0 h | 32.39 | 100.00% | 0 |
| 200 | kueue | 81.0% | 80.9% | 0.754 | 0 s / 2.6 h | 0 s / 2.8 h | 41.95 | 99.65% | 280 |
| 200 | mixsched | 82.0% | 82.0% | 0.238 | 0 s / 1.7 h | 0 s / 1.6 h | 17.93 | 100.00% | 11 |

### Ablations, Seren, days 112-126, 200 general nodes: one MixSched mechanism changed at a time

| variant | mechanism changed | eval queue p99 | training queue p99 | training JCT slowdown | fragmentation | inference SLO | preemptions | GPU-h lost |
|---|---|---|---|---|---|---|---|---|
| mixsched | (none) | 1.7 h | 1.6 h | 17.93 | 0.238 | 100.00% | 26 | 11 |
| mixsched-nofrag | fragmentation-aware placement (spread instead) | 1.8 h | 5.0 h | 33.84 | 0.771 | 100.00% | 2 | 0 |
| mixsched+resv | (adds EASY reservation for blocked training gangs) | 1.7 h | 3.2 h | 20.80 | 0.244 | 100.00% | 25 | 20 |
| mixsched-noshare | eval GPU sharing (MPS) | 1.5 h | 4.3 h | 48.83 | 0.227 | 100.00% | 28 | 21 |
| mixsched-nobatch | adaptive eval batching (one model load per batch) | 2.1 h | 6.3 h | 26.51 | 0.249 | 100.00% | 38 | 9 |
| mixsched-lendall | lend only to evals (lends to all work) | 1.9 h | 1.4 h | 16.46 | 0.262 | 100.00% | 27 | 264 |
| mixsched-newest | checkpoint-aware victims (newest first) | 1.8 h | 1.6 h | 20.11 | 0.247 | 100.00% | 20 | 10 |
| mixsched-nopredict | predictive reclaim | 2.0 h | 5.7 h | 27.48 | 0.252 | 99.65% | 18 | 15 |
| mixsched-nolease | lease-aware lending | 1.9 h | 4.3 h | 26.07 | 0.252 | 99.99% | 7255 | 2171 |
| mixsched-noevalfirst | eval-first ordering | 2.9 h | 3.3 h | 23.73 | 0.260 | 100.00% | 21 | 20 |

## Seren, days 84-98: 50,963 GPU jobs

| general nodes | policy | GPU allocation | effective utilization | fragmentation | eval queue p50 / p99 | training queue p50 / p99 | training JCT slowdown (mean) | inference SLO | GPU-h lost to preemption |
|---|---|---|---|---|---|---|---|---|---|
| 286 | fifo | 58.3% | 53.3% | 0.017 | 0 s / 0 s | 0 s / 0 s | 1.00 | 100.00% | 0 |
| 286 | binpack | 58.3% | 53.3% | 0.014 | 0 s / 0 s | 0 s / 0 s | 1.00 | 100.00% | 0 |
| 286 | kueue | 53.3% | 53.3% | 0.172 | 0 s / 0 s | 0 s / 0 s | 1.00 | 99.63% | 0 |
| 286 | mixsched | 53.4% | 53.3% | 0.015 | 0 s / 0 s | 0 s / 0 s | 1.00 | 100.00% | 0 |
| 240 | fifo | 67.7% | 61.8% | 0.034 | 0 s / 21 s | 0 s / 0 s | 1.05 | 100.00% | 0 |
| 240 | binpack | 67.7% | 61.8% | 0.029 | 0 s / 0 s | 0 s / 0 s | 1.05 | 100.00% | 0 |
| 240 | kueue | 61.8% | 61.8% | 0.269 | 0 s / 2.4 min | 0 s / 1.6 min | 1.11 | 99.63% | 0 |
| 240 | mixsched | 61.9% | 61.8% | 0.026 | 0 s / 0 s | 0 s / 0 s | 1.07 | 100.00% | 0 |
| 200 | fifo | 78.6% | 71.8% | 0.248 | 0 s / 6.3 h | 0 s / 1.8 h | 23.17 | 100.00% | 0 |
| 200 | binpack | 78.6% | 71.8% | 0.233 | 0 s / 2.6 h | 0 s / 47.6 min | 9.71 | 100.00% | 0 |
| 200 | kueue | 71.8% | 71.8% | 0.487 | 0 s / 6.0 h | 0 s / 48.7 min | 12.43 | 99.63% | 0 |
| 200 | mixsched | 71.9% | 71.8% | 0.104 | 0 s / 1.3 h | 0 s / 33.3 min | 6.17 | 100.00% | 3 |

### Ablations, Seren, days 84-98, 200 general nodes: one MixSched mechanism changed at a time

| variant | mechanism changed | eval queue p99 | training queue p99 | training JCT slowdown | fragmentation | inference SLO | preemptions | GPU-h lost |
|---|---|---|---|---|---|---|---|---|
| mixsched | (none) | 1.3 h | 33.3 min | 6.17 | 0.104 | 100.00% | 4 | 3 |
| mixsched-nofrag | fragmentation-aware placement (spread instead) | 3.0 h | 1.4 h | 15.32 | 0.485 | 100.00% | 0 | 0 |
| mixsched+resv | (adds EASY reservation for blocked training gangs) | 1.4 h | 37.7 min | 6.55 | 0.098 | 100.00% | 5 | 4 |
| mixsched-noshare | eval GPU sharing (MPS) | 1.3 h | 33.6 min | 6.57 | 0.106 | 100.00% | 4 | 3 |
| mixsched-nobatch | adaptive eval batching (one model load per batch) | 1.3 h | 33.3 min | 6.17 | 0.104 | 100.00% | 4 | 3 |
| mixsched-lendall | lend only to evals (lends to all work) | 1.3 h | 27.1 min | 3.54 | 0.127 | 100.00% | 9 | 9 |
| mixsched-newest | checkpoint-aware victims (newest first) | 1.3 h | 33.3 min | 6.17 | 0.104 | 100.00% | 4 | 3 |
| mixsched-nopredict | predictive reclaim | 1.3 h | 33.5 min | 6.33 | 0.105 | 99.63% | 6 | 3 |
| mixsched-nolease | lease-aware lending | 1.2 h | 31.1 min | 6.18 | 0.105 | 100.00% | 15 | 1191 |
| mixsched-noevalfirst | eval-first ordering | 1.1 h | 26.4 min | 5.18 | 0.109 | 100.00% | 3 | 3 |

## Kalos, days 46-60: 7,081 GPU jobs

| general nodes | policy | GPU allocation | effective utilization | fragmentation | eval queue p50 / p99 | training queue p50 / p99 | training JCT slowdown (mean) | inference SLO | GPU-h lost to preemption |
|---|---|---|---|---|---|---|---|---|---|
| 302 | fifo | 45.4% | 40.6% | 0.001 | 0 s / 0 s | 0 s / 0 s | 1.00 | 99.99% | 0 |
| 302 | binpack | 45.4% | 40.6% | 0.001 | 0 s / 0 s | 0 s / 0 s | 1.00 | 99.99% | 0 |
| 302 | kueue | 40.6% | 40.6% | 0.086 | 0 s / 0 s | 0 s / 0 s | 1.00 | 99.61% | 0 |
| 302 | mixsched | 40.6% | 40.6% | 0.003 | 0 s / 0 s | 0 s / 0 s | 1.00 | 99.99% | 0 |
| 150 | fifo | 80.2% | 71.8% | 0.005 | 0 s / 0 s | 0 s / 18 s | 1.00 | 99.99% | 0 |
| 150 | binpack | 80.2% | 71.8% | 0.005 | 0 s / 0 s | 0 s / 18 s | 1.00 | 99.99% | 0 |
| 150 | kueue | 71.8% | 71.7% | 0.352 | 0 s / 0 s | 0 s / 18 s | 1.00 | 99.61% | 0 |
| 150 | mixsched | 71.8% | 71.7% | 0.016 | 0 s / 0 s | 0 s / 18 s | 1.00 | 99.99% | 0 |

## Fragmentation on the Alibaba GPU trace v2023 (FGD's workload-inflation experiment)

Tasks sampled with replacement from `openb_pod_list_default` (8,152 tasks: shared-GPU, whole-GPU and CPU-only) and placed one at a time on the 1,213 GPU nodes of `openb_node_list_gpu_node` with no departures; a task that does not fit is dropped. Mean of 10 sampling seeds. Unallocated GPU capacity at a given arrived demand is what fragmentation (and CPU, memory and GPU-model constraints) wasted.

| policy | unallocated at 80% arrived | at 90% | at 100% | at 120% |
|---|---|---|---|---|
| Random | 21.5% | 14.5% | 13.4% | 12.6% |
| Dot-product | 23.0% | 13.5% | 7.9% | 6.0% |
| GPU clustering | 20.0% | 14.8% | 14.7% | 14.6% |
| GPU packing | 20.0% | 10.6% | 10.5% | 10.3% |
| Best-fit | 20.0% | 10.3% | 10.2% | 10.0% |
| MixSched whole-GPU frag score | 20.0% | 10.6% | 10.5% | 10.4% |
| FGD (re-implemented) | 20.0% | 10.1% | 5.0% | 4.9% |

At 100% arrived, FGD leaves 5.0% unallocated against 7.9% for the best baseline (Dot-product): 36% less capacity wasted.

## Real control plane: kube-scheduler with the MixSched plugin on KWOK fake GPU nodes

kube-apiserver and kube-scheduler v1.34.2, etcd 3.6.4 and KWOK 0.7.0, all built from source; no controller-manager (the benchmark creates the namespace's service account and removes the not-ready taint that the node lifecycle controller would). Nodes advertise 8 `nvidia.com/gpu` each. Each run fills the cluster to 90% of its GPUs with one job mix (55% single-GPU evals, small training jobs, 1-8-node training gangs of 8-GPU pods). Everything, including the benchmark client, shares one 2-vCPU VM. Attempt latency is kube-scheduler's `scheduling_attempt_duration_seconds` for successful attempts: from the start of the attempt through filtering, scoring, Permit (including a gang member's wait for the rest of its gang) and the bind call. Score p99 is the scoring extension point alone, all score plugins of the profile. pods/s counts bound pods over the time to the last bind. Mean over the workload seeds shown.

| nodes | scheduler | seeds | pods/s | attempt p50 ms | attempt p99 ms | Score p99 ms | GPU allocation | fragmentation | whole 8-GPU jobs that still fit | unscheduled pods |
|---|---|---|---|---|---|---|---|---|---|---|
| 1,000 | default | 3 | 168 | 28.6 | 524.1 | 8.35 | 55.4% | 1.000 | 0 | 347 |
| 1,000 | binpack | 3 | 235 | 55.4 | 689.3 | 7.82 | 90.1% | 0.004 | 99 | 0 |
| 1,000 | mixsched | 3 | 252 | 70.9 | 705.6 | 6.11 | 90.1% | 0.004 | 99 | 0 |
| 5,000 | default | 1 | 70 | 44.2 | 510.0 | 11.46 | 57.1% | 1.000 | 0 | 1645 |
| 5,000 | binpack | 1 | 121 | 55.5 | 809.5 | 9.79 | 84.9% | 1.000 | 0 | 255 |
| 5,000 | mixsched | 1 | 160 | 92.3 | 1022.9 | 9.67 | 90.0% | 0.974 | 13 | 0 |

### Fidelity: the simulator's placement vs the real scheduler on the same job sequence

| nodes | policy | simulator fragmentation | KWOK fragmentation | simulator whole-node fits | KWOK whole-node fits |
|---|---|---|---|---|---|
| 1,000 | default / spread | 1.000 | 1.000 | 0 | 0 |
| 1,000 | binpack / binpack | 0.004 | 0.004 | 99 | 99 |
| 1,000 | mixsched / frag | 0.004 | 0.004 | 99 | 99 |
| 5,000 | default / spread | 1.000 | 1.000 | 0 | 0 |
| 5,000 | binpack / binpack | 0.000 | 1.000 | 500 | 0 |
| 5,000 | mixsched / frag | 0.000 | 0.974 | 500 | 13 |

