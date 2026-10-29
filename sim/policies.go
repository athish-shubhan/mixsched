package sim

import "fmt"

// Policies compared in results/. Each is a fixed set of Features.
var Policies = map[string]Features{
	// Strict FIFO, first-fit, dedicated inference pool: the simplest baseline.
	"fifo": {StrictFIFO: true, Placement: "firstfit", DedicatedInfer: true, Preempt: "newest"},
	// Best-effort FIFO (a blocked job does not block later ones) with
	// best-fit bin packing, dedicated inference pool.
	"binpack": {Placement: "binpack", DedicatedInfer: true, Preempt: "newest"},
	// Kueue-style: best-effort FIFO, default kube-scheduler spreading
	// (LeastAllocated), the inference pool lends idle GPUs to anyone and
	// reclaims by preempting the newest borrowers.
	"kueue": {Placement: "spread", LendTo: "all", Preempt: "newest"},
	// MixSched: evaluation jobs first, fragmentation-aware placement, eval
	// GPU sharing and adaptive batching, lending idle inference GPUs only to
	// evaluation jobs that will finish before demand returns, checkpoint-aware
	// victim choice, predictive reclaim. EASY reservation for blocked training
	// gangs is available but off: it lengthened training queues on both Seren
	// windows tested (see results/SUMMARY.md, "mixsched+resv").
	"mixsched": {Placement: "frag", LendTo: "eval", Reservation: false, ShareEval: true, BatchEval: true,
		Preempt: "checkpoint", PredictiveReclaim: true, LeaseAware: true, EvalFirst: true},
}

// Ablations of mixsched: one mechanism off at a time.
func Ablation(name string) (Features, error) {
	f := Policies["mixsched"]
	switch name {
	case "mixsched-nofrag":
		f.Placement = "spread"
	case "mixsched+resv":
		f.Reservation = true
	case "mixsched-noshare":
		f.ShareEval = false
	case "mixsched-nobatch":
		f.BatchEval = false
	case "mixsched-lendall":
		f.LendTo, f.Preempt = "all", "checkpoint"
	case "mixsched-newest":
		f.Preempt = "newest"
	case "mixsched-nopredict":
		f.PredictiveReclaim = false
	case "mixsched-nolease":
		f.LeaseAware = false
	case "mixsched-noevalfirst":
		f.EvalFirst = false
	default:
		p, ok := Policies[name]
		if !ok {
			return f, fmt.Errorf("unknown policy %q", name)
		}
		return p, nil
	}
	return f, nil
}
