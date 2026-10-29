#!/usr/bin/env python3
"""Figures and results/SUMMARY.md for MixSched."""
import json
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

src, out = sys.argv[1], sys.argv[2]
rows = [json.loads(l) for l in open(src) if l.strip()]
for r in rows:
    r["trace_name"] = "Seren" if "seren" in r["trace"] else "Kalos"
    r["nodes"] = (r["gpus"] - 384) // 8  # general-pool nodes (the inference pool is 48 nodes)
    r["window"] = f"{r['trace_name']}, days {int(r['from_day'])}-{int(r['from_day'] + 14)}"

POL = ["fifo", "binpack", "kueue", "mixsched"]
COL = {"fifo": "#9aa5b1", "binpack": "#5b8def", "kueue": "#e8a33d", "mixsched": "#2bb3a3"}
WHAT = {"mixsched": "(none)", "mixsched-nofrag": "fragmentation-aware placement (spread instead)",
        "mixsched+resv": "(adds EASY reservation for blocked training gangs)", "mixsched-noshare": "eval GPU sharing (MPS)",
        "mixsched-nobatch": "adaptive eval batching (one model load per batch)", "mixsched-lendall": "lend only to evals (lends to all work)",
        "mixsched-newest": "checkpoint-aware victims (newest first)", "mixsched-nopredict": "predictive reclaim",
        "mixsched-nolease": "lease-aware lending", "mixsched-noevalfirst": "eval-first ordering"}

md = ["# MixSched results", "",
      "Trace replay of the public Acme job logs (Hu et al., NSDI '24) on a simulated cluster: real job sizes, types, "
      "submit times and runtimes; simulated placement, queueing, sharing and preemption. The inference fleet is synthetic "
      "(Acme has none): a diurnal demand of 115-384 GPUs on its own 48-node pool. Effective utilization counts GPUs doing "
      "work: training/eval/debug plus inference replicas that serve actual demand (warm idle replicas excluded). "
      "See docs/DESIGN.md for the model.", ""]


def fmt_s(x):
    if x >= 3600:
        return f"{x/3600:.1f} h"
    if x >= 60:
        return f"{x/60:.1f} min"
    return f"{x:.0f} s"


windows = []
for r in rows:
    if r["window"] not in windows:
        windows.append(r["window"])

for w in windows:
    rs = [r for r in rows if r["window"] == w and r["policy"] in POL]
    if not rs:
        continue
    md += [f"## {w}: {rs[0]['jobs']:,} GPU jobs", "",
           "| general nodes | policy | GPU allocation | effective utilization | fragmentation | eval queue p50 / p99 | training queue p50 / p99 | training JCT slowdown (mean) | inference SLO | GPU-h lost to preemption |",
           "|---|---|---|---|---|---|---|---|---|---|"]
    for n in sorted({r["nodes"] for r in rs}, reverse=True):
        for p in POL:
            for r in rs:
                if r["nodes"] == n and r["policy"] == p:
                    md.append(f"| {n} | {p} | {100*r['gpu_allocation_ratio']:.1f}% | {100*r['effective_utilization']:.1f}% | {r['node_fragmentation_ratio']:.3f} | "
                              f"{fmt_s(r['eval_queue_p50_s'])} / {fmt_s(r['eval_queue_p99_s'])} | {fmt_s(r['train_queue_p50_s'])} / {fmt_s(r['train_queue_p99_s'])} | "
                              f"{r['train_jct_slowdown_mean']:.2f} | {100*r['inference_slo_attainment']:.2f}% | {r['gpu_hours_lost_to_preemption']:.0f} |")
    md.append("")
    if rs[0]["trace_name"] == "Seren":
        fig, axes = plt.subplots(1, 2, figsize=(11, 4))
        for p in POL:
            pts = sorted([r for r in rs if r["policy"] == p], key=lambda r: r["effective_utilization"])
            if not pts:
                continue
            x = [100 * r["effective_utilization"] for r in pts]
            axes[0].plot(x, [max(r["eval_queue_p99_s"], 1) / 60 for r in pts], "-o", c=COL[p], label=p)
            axes[1].plot(x, [max(r["train_jct_slowdown_mean"], 1) for r in pts], "-o", c=COL[p], label=p)
        axes[0].set_title(f"{w}: eval queueing p99")
        axes[0].set_ylabel("minutes (log)")
        axes[1].set_title(f"{w}: training JCT slowdown (mean)")
        axes[1].set_ylabel("slowdown (log)")
        for ax in axes:
            ax.set_yscale("log")
            ax.set_xlabel("effective GPU utilization (%), 3 cluster sizes")
            ax.grid(alpha=0.3)
        axes[0].legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(os.path.join(out, f"seren_days{int(rs[0]['from_day'])}.png"), dpi=140)

    ab = [r for r in rows if r["window"] == w and r["nodes"] == 200 and r["policy"].startswith("mixsched")]
    if len(ab) > 1:
        md += [f"### Ablations, {w}, 200 general nodes: one MixSched mechanism changed at a time", "",
               "| variant | mechanism changed | eval queue p99 | training queue p99 | training JCT slowdown | fragmentation | inference SLO | preemptions | GPU-h lost |",
               "|---|---|---|---|---|---|---|---|---|"]
        for p in WHAT:
            for r in ab:
                if r["policy"] == p:
                    md.append(f"| {p} | {WHAT[p]} | {fmt_s(r['eval_queue_p99_s'])} | {fmt_s(r['train_queue_p99_s'])} | "
                              f"{r['train_jct_slowdown_mean']:.2f} | {r['node_fragmentation_ratio']:.3f} | {100*r['inference_slo_attainment']:.2f}% | {r['preemptions']} | {r['gpu_hours_lost_to_preemption']:.0f} |")
        md.append("")

# ---------------------------------------------------------------- Alibaba GPU trace v2023: FGD workload inflation
fg = os.path.join(out, "fgd.jsonl")
if os.path.exists(fg):
    import statistics as st
    rs = [json.loads(l) for l in open(fg) if l.strip()]
    pols = ["random", "dotprod", "clustering", "packing", "bestfit", "mixsched", "fgd"]
    label = {"random": "Random", "dotprod": "Dot-product", "clustering": "GPU clustering", "packing": "GPU packing", "bestfit": "Best-fit",
             "mixsched": "MixSched whole-GPU frag score", "fgd": "FGD (re-implemented)"}
    curves = {}
    for p in pols:
        prs = [r for r in rs if r["policy"] == p]
        if not prs:
            continue
        xs = [pt["arrived_pct"] for pt in prs[0]["points"]]
        ys = [st.mean([r["points"][i]["unallocated_pct"] for r in prs if i < len(r["points"])]) for i in range(len(xs))]
        curves[p] = (xs, ys, len(prs))
    if curves:
        fig, ax = plt.subplots(figsize=(7, 4))
        for p, (xs, ys, n) in curves.items():
            ax.plot(xs, ys, label=label[p], lw=2 if p in ("fgd", "mixsched") else 1)
        ax.set_xlabel("arrived GPU demand (% of cluster capacity)")
        ax.set_ylabel("GPU capacity unallocated (%)")
        ax.set_xlim(60, 130)
        ax.set_ylim(0, 45)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
        ax.set_title("Alibaba GPU trace v2023: workload inflation (1,213 nodes, 6,212 GPUs)")
        fig.tight_layout()
        fig.savefig(os.path.join(out, "fgd_inflation.png"), dpi=140)

        def at(p, x):
            xs, ys, _ = curves[p]
            return min(zip(xs, ys), key=lambda t: abs(t[0] - x))[1]
        n = list(curves.values())[0][2]
        md += ["## Fragmentation on the Alibaba GPU trace v2023 (FGD's workload-inflation experiment)", "",
               f"Tasks sampled with replacement from `openb_pod_list_default` (8,152 tasks: shared-GPU, whole-GPU and CPU-only) and placed one at a time on the "
               f"1,213 GPU nodes of `openb_node_list_gpu_node` with no departures; a task that does not fit is dropped. Mean of {n} sampling seeds. "
               "Unallocated GPU capacity at a given arrived demand is what fragmentation (and CPU, memory and GPU-model constraints) wasted.", "",
               "| policy | unallocated at 80% arrived | at 90% | at 100% | at 120% |", "|---|---|---|---|---|"]
        for p in curves:
            md.append(f"| {label[p]} | {at(p,80):.1f}% | {at(p,90):.1f}% | {at(p,100):.1f}% | {at(p,120):.1f}% |")
        if "fgd" in curves:
            best = min((at(p, 100), p) for p in curves if p != "fgd")
            md += ["", f"At 100% arrived, FGD leaves {at('fgd',100):.1f}% unallocated against {best[0]:.1f}% for the best baseline ({label[best[1]]}): "
                   f"{100*(1-at('fgd',100)/best[0]):.0f}% less capacity wasted.", ""]

# ---------------------------------------------------------------- KWOK: real kube-scheduler
kw = os.path.join(out, "k8s.jsonl")
if os.path.exists(kw):
    import statistics as st
    rs = [json.loads(l) for l in open(kw) if l.strip()]
    md += ["## Real control plane: kube-scheduler with the MixSched plugin on KWOK fake GPU nodes", "",
           "kube-apiserver and kube-scheduler v1.34.2, etcd 3.6.4 and KWOK 0.7.0, all built from source; no controller-manager (the benchmark creates the namespace's service account and removes the not-ready taint that the node lifecycle controller would). Nodes advertise 8 `nvidia.com/gpu` each. "
           "Each run fills the cluster to 90% of its GPUs with one job mix (55% single-GPU evals, small training jobs, 1-8-node training gangs of 8-GPU pods). "
           "Everything, including the benchmark client, shares one 2-vCPU VM. Attempt latency is kube-scheduler's `scheduling_attempt_duration_seconds` "
           "for successful attempts: from the start of the attempt through filtering, scoring, Permit (including a gang member's wait for the rest of its gang) "
           "and the bind call. Score p99 is the scoring extension point alone, all score plugins of the profile. pods/s counts bound pods over the time to the last bind. "
           "Mean over the workload seeds shown.", "",
           "| nodes | scheduler | seeds | pods/s | attempt p50 ms | attempt p99 ms | Score p99 ms | GPU allocation | fragmentation | whole 8-GPU jobs that still fit | unscheduled pods |",
           "|---|---|---|---|---|---|---|---|---|---|---|"]
    for n in sorted({r["nodes"] for r in rs}):
        for cfg in ["default", "binpack", "mixsched"]:
            g = [r for r in rs if r["nodes"] == n and r["scheduler"] == cfg]
            if not g:
                continue
            m = lambda f: st.mean([f(r) for r in g])
            md.append(f"| {n:,} | {cfg} | {len(g)} | {m(lambda r: r['pods_per_sec']):.0f} | {m(lambda r: (r.get('scheduling_attempt_ms') or {}).get('p50', 0)):.1f} | "
                      f"{m(lambda r: (r.get('scheduling_attempt_ms') or {}).get('p99', 0)):.1f} | {m(lambda r: (r.get('score_extension_ms') or {}).get('p99', 0)):.2f} | "
                      f"{100*m(lambda r: r['gpu_allocation_ratio']):.1f}% | {m(lambda r: r['node_fragmentation_ratio']):.3f} | {m(lambda r: r['whole_node_jobs_fit_after']):.0f} | "
                      f"{m(lambda r: r['unschedulable_pods']):.0f} |")
    md.append("")
    fills = []
    for n in sorted({r["nodes"] for r in rs}):
        f = os.path.join(out, f"fill-{n}.jsonl")
        if os.path.exists(f):
            fills += [json.loads(l) for l in open(f) if l.strip()]
    if fills:
        md += ["### Fidelity: the simulator's placement vs the real scheduler on the same job sequence", "",
               "| nodes | policy | simulator fragmentation | KWOK fragmentation | simulator whole-node fits | KWOK whole-node fits |", "|---|---|---|---|---|---|"]
        pair = {"spread": "default", "binpack": "binpack", "frag": "mixsched"}
        for n in sorted({r["nodes"] for r in fills}):
            for sp, kc in pair.items():
                a = [r for r in fills if r["nodes"] == n and r["policy"] == sp]
                b = [r for r in rs if r["nodes"] == n and r["scheduler"] == kc]
                if a and b:
                    md.append(f"| {n:,} | {kc} / {sp} | {st.mean([r['node_fragmentation_ratio'] for r in a]):.3f} | {st.mean([r['node_fragmentation_ratio'] for r in b]):.3f} | "
                              f"{st.mean([r['whole_node_jobs_fit_after'] for r in a]):.0f} | {st.mean([r['whole_node_jobs_fit_after'] for r in b]):.0f} |")
        md.append("")

open(os.path.join(out, "SUMMARY.md"), "w").write("\n".join(md) + "\n")
print("\n".join(md))
