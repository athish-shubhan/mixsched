package sim

import (
	"math"
	"math/rand/v2"
	"sort"

	"mixsched/trace"
)

// Workload inflation on the Alibaba GPU trace v2023, the experiment of the
// FGD paper (Weng et al., ATC '23, section 6): tasks are sampled from the
// trace with replacement and placed one at a time on the 1,213-node cluster,
// with no departures, until the arrived GPU demand reaches 130% of capacity.
// A task that cannot be placed is dropped. The metric is the share of GPU
// capacity left unallocated as the arrived demand grows: anything above zero
// at 100% is capacity lost to fragmentation (or to CPU, memory and GPU-model
// constraints).
//
// GPUs are tracked in milli-GPUs: a task asks for one GPU share (1-1000
// milli, GPU sharing) or for k whole GPUs.
//
// FGD: for a node n and the target workload M (task classes with their
// popularity), the fragment of n for class m is the idle GPU capacity m
// cannot use there: all of it if m does not fit on n at all, otherwise the
// GPUs whose idle share is below what m needs (for whole-GPU tasks, every
// partially used GPU). F_n(M) = sum_m p_m F_n(m). Each task goes to the
// node and GPU that increase F_n(M) the least.

type aliNode struct {
	cpu, mem       int64 // free
	cpuCap, memCap int64
	gpu            []int64 // free milli per GPU
	model          string
	classes        map[int]int // tasks of each class placed here (clustering policy)
}

type aliClass struct {
	gpus  int   // whole GPUs (0 for shared)
	milli int64 // share of one GPU for shared tasks
	cpu   int64
	mem   int64
	p     float64
}

func (n *aliNode) idle() int64 {
	var s int64
	for _, g := range n.gpu {
		s += g
	}
	return s
}

func (n *aliNode) fullyFree() int {
	c := 0
	for _, g := range n.gpu {
		if g == 1000 {
			c++
		}
	}
	return c
}

func modelOK(spec []string, model string) bool {
	if len(spec) == 0 {
		return true
	}
	for _, s := range spec {
		if s == model {
			return true
		}
	}
	return false
}

// fragFor is F_n(m) in milli-GPUs.
func fragFor(cpu, mem int64, gpu []int64, m aliClass) float64 {
	var idle int64
	for _, g := range gpu {
		idle += g
	}
	if cpu < m.cpu || mem < m.mem {
		return float64(idle)
	}
	if m.gpus == 0 { // shared
		var fr int64
		fits := false
		for _, g := range gpu {
			if g < m.milli {
				fr += g
			} else {
				fits = true
			}
		}
		if !fits {
			return float64(idle)
		}
		return float64(fr)
	}
	full := 0
	var part int64
	for _, g := range gpu {
		if g == 1000 {
			full++
		} else {
			part += g
		}
	}
	if full < m.gpus {
		return float64(idle)
	}
	return float64(part)
}

func nodeFrag(cpu, mem int64, gpu []int64, M []aliClass) float64 {
	var f float64
	for _, m := range M {
		f += m.p * fragFor(cpu, mem, gpu, m)
	}
	return f
}

// classes groups the trace's GPU tasks by GPU demand (whole-GPU count, or a
// shared GPU in 100-milli bins), with the median CPU and memory of each group.
func classes(pods []trace.AliPod) ([]aliClass, func(trace.AliPod) int) {
	key := func(p trace.AliPod) [2]int64 {
		if p.NumGPU == 1 && p.GPUMilli < 1000 {
			return [2]int64{0, (p.GPUMilli + 99) / 100 * 100}
		}
		return [2]int64{int64(p.NumGPU), 1000}
	}
	group := map[[2]int64][]trace.AliPod{}
	n := 0
	for _, p := range pods {
		if p.NumGPU == 0 {
			continue
		}
		k := key(p)
		group[k] = append(group[k], p)
		n++
	}
	keys := make([][2]int64, 0, len(group))
	for k := range group {
		keys = append(keys, k)
	}
	sort.Slice(keys, func(i, j int) bool { return keys[i][0] < keys[j][0] || (keys[i][0] == keys[j][0] && keys[i][1] < keys[j][1]) })
	idx := map[[2]int64]int{}
	var out []aliClass
	for i, k := range keys {
		g := group[k]
		cpus, mems := make([]int64, len(g)), make([]int64, len(g))
		for j, p := range g {
			cpus[j], mems[j] = p.CPUMilli, p.MemMiB
		}
		sort.Slice(cpus, func(a, b int) bool { return cpus[a] < cpus[b] })
		sort.Slice(mems, func(a, b int) bool { return mems[a] < mems[b] })
		c := aliClass{gpus: int(k[0]), milli: k[1], cpu: cpus[len(cpus)/2], mem: mems[len(mems)/2], p: float64(len(g)) / float64(n)}
		out = append(out, c)
		idx[k] = i
	}
	return out, func(p trace.AliPod) int {
		if p.NumGPU == 0 {
			return -1
		}
		return idx[key(p)]
	}
}

type InflationPoint struct {
	Arrived     float64 `json:"arrived_pct"`     // arrived GPU demand, % of capacity
	Unallocated float64 `json:"unallocated_pct"` // GPU capacity not allocated, %
	Failed      float64 `json:"failed_pct"`      // GPU demand of dropped tasks, % of capacity
}

type InflationResult struct {
	Policy  string           `json:"policy"`
	Seed    uint64           `json:"seed"`
	GPUs    int              `json:"gpus"`
	Points  []InflationPoint `json:"points"`
	Placed  int              `json:"tasks_placed"`
	Dropped int              `json:"tasks_dropped"`
}

// Inflate runs one inflation experiment. Policies: random, bestfit,
// dotprod, packing, clustering, fgd, and mixsched (the whole-GPU
// fragmentation score from MixSched's placement plus packing of shared
// tasks, to show what a whole-GPU model misses on fractional work).
func Inflate(policy string, nodesIn []trace.AliNode, pods []trace.AliPod, seed uint64, maxArrived float64) InflationResult {
	r := rand.New(rand.NewPCG(seed, 0x6a09e667))
	M, classOf := classes(pods)
	nodes := make([]*aliNode, len(nodesIn))
	var capMilli int64
	for i, n := range nodesIn {
		g := make([]int64, n.GPUs)
		for j := range g {
			g[j] = 1000
		}
		nodes[i] = &aliNode{cpu: n.CPUMilli, mem: n.MemMiB, cpuCap: n.CPUMilli, memCap: n.MemMiB, gpu: g, model: n.Model, classes: map[int]int{}}
		capMilli += int64(n.GPUs) * 1000
	}
	frag := make([]float64, len(nodes))
	for i, n := range nodes {
		frag[i] = nodeFrag(n.cpu, n.mem, n.gpu, M)
	}
	res := InflationResult{Policy: policy, Seed: seed, GPUs: int(capMilli / 1000)}
	var arrived, allocated, failed int64
	next := 0.05
	scratch := make([]int64, 8)
	for float64(arrived)/float64(capMilli) < maxArrived {
		p := pods[r.IntN(len(pods))]
		demand := int64(p.NumGPU) * 1000
		if p.NumGPU == 1 {
			demand = p.GPUMilli
		}
		arrived += demand
		ci := classOf(p)
		best, bestGPU := -1, -1
		bestScore := math.Inf(1)
		ties := 0
		consider := func(i, g int, score float64) {
			if score < bestScore-1e-9 {
				best, bestGPU, bestScore, ties = i, g, score, 1
			} else if math.Abs(score-bestScore) <= 1e-9 {
				ties++
				if r.IntN(ties) == 0 {
					best, bestGPU = i, g
				}
			}
		}
		for i, n := range nodes {
			if n.cpu < p.CPUMilli || n.mem < p.MemMiB || !modelOK(p.Spec, n.model) {
				continue
			}
			if p.NumGPU == 0 {
				var s float64
				switch policy {
				case "fgd":
					s = nodeFrag(n.cpu-p.CPUMilli, n.mem-p.MemMiB, n.gpu, M) - frag[i]
				case "random":
					s = 0
				case "dotprod":
					s = -(float64(p.CPUMilli)*float64(n.cpu)/float64(n.cpuCap) + float64(p.MemMiB)*float64(n.mem)/float64(n.memCap))
				default:
					s = float64(n.cpu-p.CPUMilli)/float64(n.cpuCap) + float64(n.mem-p.MemMiB)/float64(n.memCap)
				}
				consider(i, -1, s)
				continue
			}
			if p.NumGPU == 1 && p.GPUMilli < 1000 {
				// shared task: choose a GPU on this node
				seen := map[int64]bool{}
				for g, free := range n.gpu {
					if free < p.GPUMilli || seen[free] {
						continue
					}
					seen[free] = true // GPUs with the same free share are equivalent
					var s float64
					switch policy {
					case "fgd":
						copy(scratch, n.gpu)
						sg := scratch[:len(n.gpu)]
						sg[g] -= p.GPUMilli
						s = nodeFrag(n.cpu-p.CPUMilli, n.mem-p.MemMiB, sg, M) - frag[i]
					case "random":
						s = 0
					case "bestfit":
						s = float64(n.cpu-p.CPUMilli)/float64(n.cpuCap) + float64(n.mem-p.MemMiB)/float64(n.memCap) + float64(n.idle()-p.GPUMilli)/float64(1000*len(n.gpu))
					case "dotprod":
						s = -(float64(p.CPUMilli)*float64(n.cpu)/float64(n.cpuCap) + float64(p.MemMiB)*float64(n.mem)/float64(n.memCap) + float64(p.GPUMilli)*float64(n.idle())/float64(1000*len(n.gpu)))
					case "packing", "mixsched":
						// prefer a GPU already in use, the fullest that still fits; then the fullest node
						s = float64(free-p.GPUMilli) + 1e-4*float64(n.idle())
						if free == 1000 {
							s += 2000
							if policy == "mixsched" {
								s += 100 * FragDeltaMilli(n.fullyFree(), 1)
							}
						}
					case "clustering":
						s = -float64(n.classes[ci])*1e6 + float64(free-p.GPUMilli)
					}
					consider(i, g, s)
				}
				continue
			}
			if n.fullyFree() < p.NumGPU {
				continue
			}
			var s float64
			switch policy {
			case "fgd":
				copy(scratch, n.gpu)
				sg := scratch[:len(n.gpu)]
				take := p.NumGPU
				for g := range sg {
					if take > 0 && sg[g] == 1000 {
						sg[g] = 0
						take--
					}
				}
				s = nodeFrag(n.cpu-p.CPUMilli, n.mem-p.MemMiB, sg, M) - frag[i]
			case "random":
				s = 0
			case "bestfit":
				s = float64(n.cpu-p.CPUMilli)/float64(n.cpuCap) + float64(n.mem-p.MemMiB)/float64(n.memCap) + float64(n.idle()-demand)/float64(1000*len(n.gpu))
			case "dotprod":
				s = -(float64(p.CPUMilli)*float64(n.cpu)/float64(n.cpuCap) + float64(p.MemMiB)*float64(n.mem)/float64(n.memCap) + float64(demand)*float64(n.idle())/float64(1000*len(n.gpu)))
			case "packing":
				s = float64(n.fullyFree() - p.NumGPU) // fewest whole GPUs left
			case "mixsched":
				s = FragDeltaMilli(n.fullyFree(), p.NumGPU)
			case "clustering":
				s = -float64(n.classes[ci])*1e6 + float64(n.fullyFree()-p.NumGPU)
			}
			consider(i, -1, s)
		}
		if best < 0 {
			failed += demand
			res.Dropped++
		} else {
			n := nodes[best]
			n.cpu -= p.CPUMilli
			n.mem -= p.MemMiB
			if p.NumGPU == 1 && p.GPUMilli < 1000 {
				n.gpu[bestGPU] -= p.GPUMilli
			} else {
				take := p.NumGPU
				for g := range n.gpu {
					if take > 0 && n.gpu[g] == 1000 {
						n.gpu[g] = 0
						take--
					}
				}
			}
			allocated += demand
			n.classes[ci]++
			frag[best] = nodeFrag(n.cpu, n.mem, n.gpu, M)
			res.Placed++
		}
		for float64(arrived)/float64(capMilli) >= next-1e-12 && next <= maxArrived+1e-9 {
			res.Points = append(res.Points, InflationPoint{Arrived: 100 * next, Unallocated: 100 * (1 - float64(allocated)/float64(capMilli)),
				Failed: 100 * float64(failed) / float64(capMilli)})
			next += 0.05
		}
	}
	return res
}

// FragDeltaMilli is MixSched's whole-GPU fragDelta on a node's count of
// fully free GPUs (nodes in this trace have 1, 2, 4 or 8 GPUs).
func FragDeltaMilli(fullyFree, take int) float64 { return fragDeltaFor(fullyFree, take) }
