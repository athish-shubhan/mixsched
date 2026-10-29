package sim

import (
	"math/rand/v2"
)

// Static fill: the simulator side of the simulator-vs-Kubernetes fidelity
// check. It places the same job sequence the KWOK benchmark creates
// (k8s/bench: same size mix, same random stream for a given seed) on N empty
// 8-GPU nodes with no departures, until the requested GPUs reach a share of
// capacity, and reports the same placement metrics. Policies mirror the three
// kube-scheduler configurations: spread (default: least allocated, which on
// CPU/memory with equal requests means fewest pods), binpack (most GPUs
// allocated) and frag (MixSched's fragDelta score). Ties are broken at
// random, as kube-scheduler does.

type FillJob struct {
	Class string
	GPUs  int // per pod
	Pods  int // gang members
}

// SampleFillJob is k8s/bench's sample(); keep the two identical.
func SampleFillJob(r *rand.Rand) FillJob {
	x := r.Float64()
	switch {
	case x < 0.55:
		return FillJob{"eval", 1, 1}
	case x < 0.70:
		return FillJob{"train", 1, 1}
	case x < 0.80:
		return FillJob{"train", 2, 1}
	case x < 0.88:
		return FillJob{"train", 4, 1}
	case x < 0.94:
		return FillJob{"infer", 1, 1}
	default:
		return FillJob{"train", 8, []int{1, 2, 2, 4, 4, 8}[r.IntN(6)]}
	}
}

type FillResult struct {
	Policy        string  `json:"policy"`
	Nodes         int     `json:"nodes"`
	Pods          int     `json:"pods"`
	Placed        int     `json:"pods_bound"`
	AllocRatio    float64 `json:"gpu_allocation_ratio"`
	Fragmentation float64 `json:"node_fragmentation_ratio"`
	FreeNodes     int     `json:"fully_free_nodes"`
	WholeNodeFit  int     `json:"whole_node_jobs_fit_after"`
}

func Fill(policy string, nodes int, fill float64, seed uint64) FillResult {
	const per = 8
	free := make([]int, nodes)
	pods := make([]int, nodes)
	for i := range free {
		free[i] = per
	}
	r := rand.New(rand.NewPCG(seed, 7))
	tie := rand.New(rand.NewPCG(seed, 99))
	var jobs []FillJob
	var demand int
	for float64(demand) < float64(nodes*per)*fill {
		j := SampleFillJob(r)
		jobs = append(jobs, j)
		demand += j.GPUs * j.Pods
	}
	res := FillResult{Policy: policy, Nodes: nodes}
	place := func(g int) int {
		best, bestS, n := -1, 0.0, 0
		for i := range free {
			if free[i] < g {
				continue
			}
			var s float64 // higher is better
			switch policy {
			case "spread":
				s = -float64(pods[i])
			case "binpack":
				s = float64(per - free[i])
			case "frag":
				fd := FragDeltaK8s(free[i], g)
				s = -fd
			}
			if best < 0 || s > bestS {
				best, bestS, n = i, s, 1
			} else if s == bestS {
				n++
				if tie.IntN(n) == 0 {
					best = i
				}
			}
		}
		return best
	}
	for _, j := range jobs {
		res.Pods += j.Pods
		for m := 0; m < j.Pods; m++ {
			if i := place(j.GPUs); i >= 0 {
				free[i] -= j.GPUs
				pods[i]++
				res.Placed++
			}
		}
	}
	var alloc, fr, stranded int
	for _, f := range free {
		alloc += per - f
		fr += f
		if f > 0 && f < per {
			stranded += f
		}
		if f == per {
			res.FreeNodes++
		}
		res.WholeNodeFit += f / per
	}
	res.AllocRatio = float64(alloc) / float64(nodes*per)
	if fr > 0 {
		res.Fragmentation = float64(stranded) / float64(fr)
	}
	return res
}

// FragDeltaK8s is the plugin's integer-score version of fragDelta: the
// plugin maps FragDelta to int64 scores in [0, 100] (50 - 8*delta), and the
// rounding matters for ties, so the fill uses the same rounding.
func FragDeltaK8s(free, take int) float64 {
	s := 50 - 8*fragDeltaFor(free, take)
	return -float64(int64(min(max(s, 0), 100)))
}

func fragDeltaFor(free, take int) float64 {
	frag := func(f int) float64 {
		var x float64
		for _, m := range sizeMix {
			if f < m.gpus {
				x += m.p * float64(f)
			}
		}
		return x
	}
	return frag(free-take) - frag(free)
}
