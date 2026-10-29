// Package plugin is MixSched's placement and ordering as a kube-scheduler
// framework plugin. It ports the simulator's mechanisms that belong in the
// scheduler (sim/sim.go: place, fragDelta; sim/policies.go):
//
//   - QueueSort: evaluation pods first, then by priority and arrival
//     (eval-first ordering; quota and borrowing belong to Kueue).
//   - Score: fragmentation-aware placement for GPU pods. The score is minus
//     the increase in "stranded" GPUs a placement causes, where a free GPU
//     is stranded for a job size it cannot fit, weighted by how much GPU
//     demand comes in that size (FGD's idea, for whole GPUs). Breaking a
//     fully free node costs the most; filling a node exactly costs nothing.
//   - PreFilter + Permit: gang scheduling. Pods with the same
//     mixsched.io/gang label wait at Permit until mixsched.io/gang-size of
//     them hold reservations, then all bind; on timeout every waiting member
//     is rejected and its reservation released. PreFilter turns a gang away
//     early when the cluster does not have enough free GPUs left for the
//     members that are not placed yet, so partial gangs do not sit on GPUs.
//
// Pods without a GPU request are left to the default plugins (score 0 here).
package plugin

import (
	"context"
	"fmt"
	"strconv"
	"sync"
	"time"

	v1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/labels"
	"k8s.io/apimachinery/pkg/runtime"
	corelisters "k8s.io/client-go/listers/core/v1"
	corev1helpers "k8s.io/component-helpers/scheduling/corev1"
	fwk "k8s.io/kube-scheduler/framework"
	"k8s.io/kubernetes/pkg/scheduler/framework"
)

const (
	Name          = "MixSched"
	LabelClass    = "mixsched.io/class" // train | eval | infer
	LabelGang     = "mixsched.io/gang"
	LabelGangSize = "mixsched.io/gang-size"
	GPU           = v1.ResourceName("nvidia.com/gpu")
	gangTimeout   = 30 * time.Second
)

// SizeMix is the share of GPU demand by job size (from the Acme traces:
// most GPU time is in multi-node training on 8-GPU nodes).
var SizeMix = []struct {
	GPUs int64
	P    float64
}{{1, 0.15}, {2, 0.05}, {4, 0.05}, {8, 0.75}}

type MixSched struct {
	h    framework.Handle
	pods corelisters.PodLister
	mu   sync.Mutex
}

var (
	_ framework.QueueSortPlugin = &MixSched{}
	_ framework.PreFilterPlugin = &MixSched{}
	_ framework.ScorePlugin     = &MixSched{}
	_ framework.PermitPlugin    = &MixSched{}
)

func New(_ context.Context, _ runtime.Object, h framework.Handle) (framework.Plugin, error) {
	return &MixSched{h: h, pods: h.SharedInformerFactory().Core().V1().Pods().Lister()}, nil
}

func (m *MixSched) Name() string { return Name }

func gpuRequest(p *v1.Pod) int64 {
	var n int64
	for _, c := range p.Spec.Containers {
		if q, ok := c.Resources.Requests[GPU]; ok {
			n += q.Value()
		} else if q, ok := c.Resources.Limits[GPU]; ok {
			n += q.Value()
		}
	}
	return n
}

func classRank(p *v1.Pod) int {
	switch p.Labels[LabelClass] {
	case "eval":
		return 0
	case "infer":
		return 1
	}
	return 2
}

// Less: evaluation first, then priority, then arrival.
func (m *MixSched) Less(a, b fwk.QueuedPodInfo) bool {
	pa, pb := a.GetPodInfo().GetPod(), b.GetPodInfo().GetPod()
	if ra, rb := classRank(pa), classRank(pb); ra != rb {
		return ra < rb
	}
	p1, p2 := corev1helpers.PodPriority(pa), corev1helpers.PodPriority(pb)
	return p1 > p2 || (p1 == p2 && a.GetTimestamp().Before(b.GetTimestamp()))
}

// stranded is how many free GPUs on a node are useless to each job size,
// weighted by that size's share of demand.
func stranded(free int64) float64 {
	var x float64
	for _, s := range SizeMix {
		if free < s.GPUs {
			x += s.P * float64(free)
		}
	}
	return x
}

// FragDelta is the increase in stranded GPUs when take GPUs are placed on a
// node with free GPUs (the simulator's fragDelta).
func FragDelta(free, take int64) float64 { return stranded(free-take) - stranded(free) }

func freeGPUs(n fwk.NodeInfo) int64 {
	return n.GetAllocatable().GetScalarResources()[GPU] - n.GetRequested().GetScalarResources()[GPU]
}

// Score: higher is better. FragDelta ranges over roughly [-4, +6] for 8-GPU
// nodes; map it into [0, 100] with 0 delta at 50.
func (m *MixSched) Score(_ context.Context, _ fwk.CycleState, p *v1.Pod, n fwk.NodeInfo) (int64, *fwk.Status) {
	g := gpuRequest(p)
	if g == 0 {
		return 0, nil
	}
	f := freeGPUs(n)
	if f < g {
		return 0, nil
	}
	s := 50 - 8*FragDelta(f, g)
	return int64(min(max(s, 0), 100)), nil
}

func (m *MixSched) ScoreExtensions() framework.ScoreExtensions { return nil }

func gangOf(p *v1.Pod) (string, int, bool) {
	g := p.Labels[LabelGang]
	if g == "" {
		return "", 0, false
	}
	n, err := strconv.Atoi(p.Labels[LabelGangSize])
	if err != nil || n < 1 {
		return "", 0, false
	}
	return p.Namespace + "/" + g, n, true
}

// members counts gang pods already bound and waiting at Permit.
func (m *MixSched) members(key, ns, gang string) (bound, waiting int) {
	sel := labels.SelectorFromSet(labels.Set{LabelGang: gang})
	if pods, err := m.pods.Pods(ns).List(sel); err == nil {
		for _, p := range pods {
			if p.Spec.NodeName != "" && p.DeletionTimestamp == nil {
				bound++
			}
		}
	}
	m.h.IterateOverWaitingPods(func(wp framework.WaitingPod) {
		if k, _, ok := gangOf(wp.GetPod()); ok && k == key {
			waiting++
		}
	})
	return
}

// PreFilter rejects a gang member early if the free GPUs in the whole
// cluster cannot cover the members that still need a place.
func (m *MixSched) PreFilter(_ context.Context, _ fwk.CycleState, p *v1.Pod, nodes []fwk.NodeInfo) (*framework.PreFilterResult, *fwk.Status) {
	key, size, ok := gangOf(p)
	if !ok {
		return nil, nil
	}
	bound, waiting := m.members(key, p.Namespace, p.Labels[LabelGang])
	need := int64(size-bound-waiting) * gpuRequest(p)
	var free int64
	for _, n := range nodes {
		free += freeGPUs(n)
	}
	if free < need {
		return nil, fwk.NewStatus(fwk.Unschedulable, fmt.Sprintf("gang %s needs %d GPUs for %d members, %d free", key, need, size-bound-waiting, free))
	}
	return nil, nil
}

func (m *MixSched) PreFilterExtensions() framework.PreFilterExtensions { return nil }

// Permit holds gang members until the whole gang has reservations.
func (m *MixSched) Permit(_ context.Context, _ fwk.CycleState, p *v1.Pod, _ string) (*fwk.Status, time.Duration) {
	key, size, ok := gangOf(p)
	if !ok {
		return nil, 0
	}
	m.mu.Lock()
	defer m.mu.Unlock()
	bound, waiting := m.members(key, p.Namespace, p.Labels[LabelGang])
	if bound+waiting+1 >= size {
		m.h.IterateOverWaitingPods(func(wp framework.WaitingPod) {
			if k, _, ok := gangOf(wp.GetPod()); ok && k == key {
				wp.Allow(Name)
			}
		})
		return nil, 0
	}
	return fwk.NewStatus(fwk.Wait, ""), gangTimeout
}
