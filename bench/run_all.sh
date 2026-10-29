#!/usr/bin/env bash
# Reproduces results/: four policies across cluster sizes on two public Acme
# windows, plus ablations of every MixSched mechanism. About 40 minutes on 2 vCPUs.
# Traces: git clone https://github.com/InternLM/AcmeTrace and copy
# data/job_trace/trace_{seren,kalos}.csv into data/ (see data/README.md).
set -euo pipefail
cd "$(dirname "$0")/.."
go build -o bin/mixsched ./cmd/mixsched
OUT=results/runs.jsonl
: > "$OUT"
P=fifo,binpack,kueue,mixsched
# Seren, two windows: days 112-126 (the most evaluation-heavy two weeks,
# 135k eval jobs) and days 84-98. 286 nodes is the real cluster; fewer nodes
# raise the offered load.
for from in 112 84; do
  for n in 286 240 200; do
    ./bin/mixsched -trace data/trace_seren.csv -from-day $from -days 14 -nodes $n -policies $P -out "$OUT"
  done
  ./bin/mixsched -trace data/trace_seren.csv -from-day $from -days 14 -nodes 200 -out "$OUT" \
    -policies mixsched-nofrag,mixsched+resv,mixsched-noshare,mixsched-nobatch,mixsched-lendall,mixsched-newest,mixsched-nopredict,mixsched-nolease,mixsched-noevalfirst
done
# Kalos, days 46-60, its busiest two weeks: a control. Its GPU time is 95%
# pretraining gangs, and no size tested produced queueing.
for n in 302 150; do
  ./bin/mixsched -trace data/trace_kalos.csv -from-day 46 -days 14 -nodes $n -policies $P -out "$OUT"
done
# Alibaba GPU trace v2023 (data/alibaba, from github.com/alibaba/clusterdata):
# FGD's workload-inflation experiment, 10 sampling seeds per policy.
: > results/fgd.jsonl
./bin/mixsched fgd -seeds 10 -out results/fgd.jsonl
# The real control plane (needs bench/k8s/fetch.sh and build.sh): bench/k8s/run.sh
python3 bench/plot.py "$OUT" results
