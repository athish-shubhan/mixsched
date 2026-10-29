#!/usr/bin/env bash
# KWOK benchmark: for 1,000 and 5,000 fake 8-GPU nodes, fill the cluster to
# 90% of its GPUs with the same job mix under each scheduler configuration
# (default kube-scheduler, GPU bin packing, MixSched), 3 workload seeds.
# Writes results/k8s.jsonl. Needs build.sh and fetch.sh first.
set -euo pipefail
cd "$(dirname "$0")/../.."
BIN=${BIN:-$PWD/bin/k8s}
W=${W:-/var/lib/mixsched-k8s}
OUT=${OUT:-results/k8s.jsonl}
NODES=${NODES:-"1000 5000"}
SEEDS=${SEEDS:-"1 2 3"}
CONFIGS=${CONFIGS:-"default binpack mixsched"}
: > "$OUT"
for n in $NODES; do
  bench/k8s/cluster.sh up
  "$BIN/mixsched-bench" -kubeconfig "$W/kubeconfig" -setup -nodes "$n"
  # wait for KWOK to report every node Ready
  for i in $(seq 120); do
    r=$("$BIN/kubectl" --kubeconfig "$W/kubeconfig" get nodes --no-headers 2>/dev/null | grep -c " Ready" || true)
    [[ $r -ge $n ]] && break
    sleep 2
  done
  echo "nodes ready: $r / $n"
  for seed in $SEEDS; do
    for cfg in $CONFIGS; do
      "$BIN/mixsched-bench" -kubeconfig "$W/kubeconfig" -clean
      bench/k8s/cluster.sh scheduler "k8s/config/$cfg.yaml"
      "$BIN/mixsched-bench" -kubeconfig "$W/kubeconfig" -nodes "$n" -fill 0.9 -seed "$seed" -label "$cfg" -out "$OUT"
      bench/k8s/cluster.sh stop-scheduler
    done
  done
  bench/k8s/cluster.sh down
done
# the simulator side of the fidelity check: same job sequence, same policies
for n in $NODES; do
  ./bin/mixsched fill -nodes "$n" -fill 0.9 -seeds 3 > "results/fill-$n.jsonl"
done
