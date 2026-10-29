// Package sim is a trace-driven simulator of a shared GPU cluster running
// training, evaluation, debug and inference work under a pluggable policy.
package sim

// GPU slot states.
const (
	free      = 0
	exclusive = -1
)

// Node is one 8-GPU server. gpu[i] is free (0), exclusive (-1), or the
// number of evaluation jobs sharing it (MPS-style, 1..ShareMax).
type Node struct {
	ID   int
	Rack int
	gpu  []int
	Pool string // "general" or "inference" (dedicated pool for static policies)
}

func (n *Node) FreeGPUs() int {
	c := 0
	for _, g := range n.gpu {
		if g == free {
			c++
		}
	}
	return c
}

func (n *Node) UsedGPUs() int { return len(n.gpu) - n.FreeGPUs() }

// SharedSlot returns a GPU already shared by fewer than max eval jobs.
func (n *Node) SharedSlot(max int) int {
	for i, g := range n.gpu {
		if g > 0 && g < max {
			return i
		}
	}
	return -1
}

type Cluster struct {
	Nodes    []*Node
	GPUsPer  int
	ShareMax int
}

func NewCluster(general, inference, gpusPer, rackSize, shareMax int) *Cluster {
	c := &Cluster{GPUsPer: gpusPer, ShareMax: shareMax}
	for i := 0; i < general+inference; i++ {
		pool := "general"
		if i >= general {
			pool = "inference"
		}
		c.Nodes = append(c.Nodes, &Node{ID: i, Rack: i / rackSize, gpu: make([]int, gpusPer), Pool: pool})
	}
	return c
}

func (c *Cluster) TotalGPUs() int { return len(c.Nodes) * c.GPUsPer }

func (c *Cluster) AllocatedGPUs() int {
	t := 0
	for _, n := range c.Nodes {
		t += n.UsedGPUs()
	}
	return t
}

// Fragmentation is the share of free GPUs stranded on partially used nodes,
// i.e. free capacity that an 8-GPU (whole-node) job cannot use.
func (c *Cluster) Fragmentation() float64 {
	var freeAll, stranded int
	for _, n := range c.Nodes {
		f := n.FreeGPUs()
		freeAll += f
		if f > 0 && f < c.GPUsPer {
			stranded += f
		}
	}
	if freeAll == 0 {
		return 0
	}
	return float64(stranded) / float64(freeAll)
}

// Placement is where a job's GPUs went.
type Placement struct {
	Node int
	GPU  int
}
