package sim

import (
	"testing"

	"mixsched/trace"
)

func synthAli() ([]trace.AliNode, []trace.AliPod) {
	var ns []trace.AliNode
	for i := 0; i < 60; i++ {
		ns = append(ns, trace.AliNode{Name: "n", CPUMilli: 96000, MemMiB: 393216, GPUs: 8, Model: "G2"})
	}
	ps := []trace.AliPod{
		{CPUMilli: 4000, MemMiB: 8192, NumGPU: 1, GPUMilli: 300},
		{CPUMilli: 4000, MemMiB: 8192, NumGPU: 1, GPUMilli: 500},
		{CPUMilli: 4000, MemMiB: 8192, NumGPU: 1, GPUMilli: 700},
		{CPUMilli: 8000, MemMiB: 16384, NumGPU: 1, GPUMilli: 1000},
		{CPUMilli: 32000, MemMiB: 65536, NumGPU: 4, GPUMilli: 1000},
		{CPUMilli: 64000, MemMiB: 131072, NumGPU: 8, GPUMilli: 1000},
	}
	return ns, ps
}

func TestInflationRunsAndFGDHelps(t *testing.T) {
	ns, ps := synthAli()
	var fgd, rnd float64
	for seed := uint64(1); seed <= 3; seed++ {
		f := Inflate("fgd", ns, ps, seed, 1.0)
		r := Inflate("random", ns, ps, seed, 1.0)
		if len(f.Points) == 0 || f.Placed == 0 {
			t.Fatalf("no progress: %+v", f)
		}
		// unallocated never increases as work arrives
		for i := 1; i < len(f.Points); i++ {
			if f.Points[i].Unallocated > f.Points[i-1].Unallocated+1e-9 {
				t.Fatalf("unallocated went up: %+v", f.Points)
			}
		}
		fgd += f.Points[len(f.Points)-1].Unallocated
		rnd += r.Points[len(r.Points)-1].Unallocated
	}
	if fgd >= rnd {
		t.Fatalf("FGD left %.1f%% unallocated, random %.1f%%", fgd/3, rnd/3)
	}
	t.Logf("unallocated at 100%% arrived: fgd %.1f%%, random %.1f%%", fgd/3, rnd/3)
}

func TestFillPolicies(t *testing.T) {
	sp, fr := Fill("spread", 200, 0.9, 1), Fill("frag", 200, 0.9, 1)
	if fr.Fragmentation >= sp.Fragmentation {
		t.Fatalf("frag-aware placement fragments more than spreading: %.3f vs %.3f", fr.Fragmentation, sp.Fragmentation)
	}
	if fr.WholeNodeFit < sp.WholeNodeFit {
		t.Fatalf("whole-node fits: frag %d < spread %d", fr.WholeNodeFit, sp.WholeNodeFit)
	}
}
