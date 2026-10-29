package sim

import (
	"container/heap"
	"math"
	"sort"

	"mixsched/trace"
)

// Features select the mechanisms a policy uses. Named policies are fixed
// combinations (see policies.go), so ablations are one flag away.
type Features struct {
	StrictFIFO        bool   // stop at the first pending job that does not fit (head-of-line blocking)
	Placement         string // firstfit | spread | binpack | frag
	DedicatedInfer    bool   // inference runs only on its own nodes, which nobody borrows
	LendTo            string // elastic inference pool lends idle GPUs to: "all" | "eval" | "none"
	Reservation       bool   // EASY backfill: reserve GPUs for the oldest blocked training gang
	ShareEval         bool   // 1-GPU eval jobs may share a GPU (MPS), ShareMax per GPU
	BatchEval         bool   // same-user eval jobs within a window run as one batch, loading the model once
	Preempt           string // victim order when inference reclaims: "newest" | "checkpoint"
	PredictiveReclaim bool   // size inference for demand one model-load time ahead
	LeaseAware        bool   // lend an idle inference GPU only to work expected to finish before demand returns
	EvalFirst         bool   // evaluation jobs are ordered ahead of other pending work (the reservation still protects training gangs)
}

type Config struct {
	GPUsPer         int
	RackSize        int
	ShareMax        int
	SharePenalty    float64 // runtime multiplier while sharing a GPU
	CkptInterval    float64 // training checkpoint period (s)
	RestartOverhead float64 // s to reload state after a preemption
	EvalLoadFrac    float64 // share of an eval job spent loading the model (Acme: ~29.5%)
	BatchWindow     float64 // s a forming eval batch waits for more members
	BatchMax        int
	BatchJobMax     float64 // only eval jobs up to this runtime are batched (s)
	BatchWorkMax    float64 // cap on a batch's total sequential runtime (s)
	InferPeakGPUs   int     // peak inference demand (GPUs = replicas)
	InferMinFrac    float64 // trough demand as a share of peak
	InferLoad       float64 // s for a new replica to load its model before serving
	ScanLimit       int     // pending runs examined per scheduling pass
}

func DefaultConfig() Config {
	return Config{GPUsPer: 8, RackSize: 16, ShareMax: 2, SharePenalty: 1.3, CkptInterval: 1800, RestartOverhead: 300,
		EvalLoadFrac: 0.295, BatchWindow: 60, BatchMax: 8, BatchJobMax: 600, BatchWorkMax: 1800, InferPeakGPUs: 384, InferMinFrac: 0.3, InferLoad: 120, ScanLimit: 3000}
}

// run is a schedulable unit: a trace job, an eval batch, or an inference replica.
type run struct {
	id        int
	class     trace.Class
	jobs      []*trace.Job
	gpus      int
	nodes     int
	work      float64 // remaining seconds of work at full speed
	submit    float64
	readyAt   float64
	start     float64 // current start (after a preemption: the restart)
	first     float64 // first start, -1 before
	lastCkpt  float64
	place     []Placement
	shared    bool
	slowed    bool // joined an occupied GPU: runs SharePenalty slower
	epoch     int
	running   bool
	done      bool
	pending   bool
	borrowing bool // placed on inference-pool nodes
	finishAt  float64
	preempts  int
}

type event struct {
	t     float64
	kind  int // 0 submit, 1 finish, 2 inference tick, 3 batch ready
	r     *run
	job   *trace.Job
	epoch int
}

type evq []event

func (q evq) Len() int { return len(q) }
func (q evq) Less(i, j int) bool {
	if q[i].t != q[j].t {
		return q[i].t < q[j].t
	}
	return q[i].kind < q[j].kind
}
func (q evq) Swap(i, j int) { q[i], q[j] = q[j], q[i] }
func (q *evq) Push(x any)   { *q = append(*q, x.(event)) }
func (q *evq) Pop() any {
	o := *q
	e := o[len(o)-1]
	*q = o[:len(o)-1]
	return e
}

type Sim struct {
	Cfg     Config
	F       Features
	Name    string
	C       *Cluster
	now     float64
	q       evq
	pend    []*run
	active  map[*run]struct{} // running runs
	batch   map[string]*run
	horizon float64
	nextID  int

	inferTarget int
	inferRuns   []*run
	lendAt      float64
	lendVal     float64

	// metrics
	lastT       float64
	allocGPUs   float64 // integral of allocated GPUs over time
	effGPUs     float64 // integral of GPUs doing useful work: non-inference + inference replicas serving demand
	busyGPUs    float64 // integral excluding inference replicas
	fragInt     float64
	inferShort  float64 // replica-seconds of unmet inference demand
	inferDemand float64
	lostGPUSec  float64
	preemptions int
	evalDelays  []float64
	trainDelays []float64
	trainSlow   []float64
	debugDelays []float64
	end         float64
	finished    int
}

type Result struct {
	Policy            string  `json:"policy"`
	Jobs              int     `json:"jobs"`
	Finished          int     `json:"finished"`
	DaysSimulated     float64 `json:"days_simulated"`
	GPUs              int     `json:"gpus"`
	AllocRatio        float64 `json:"gpu_allocation_ratio"`     // allocated GPU-time / capacity, all classes
	NonInferAlloc     float64 `json:"non_inference_allocation"` // allocated to training/eval/debug / capacity
	EffectiveUtil     float64 `json:"effective_utilization"`    // non-inference allocation + inference replicas actually needed
	Fragmentation     float64 `json:"node_fragmentation_ratio"` // time-avg share of free GPUs stranded on partial nodes
	EvalQueueP50      float64 `json:"eval_queue_p50_s"`
	EvalQueueP99      float64 `json:"eval_queue_p99_s"`
	EvalQueueMean     float64 `json:"eval_queue_mean_s"`
	TrainQueueP50     float64 `json:"train_queue_p50_s"`
	TrainQueueP99     float64 `json:"train_queue_p99_s"`
	TrainSlowdownMean float64 `json:"train_jct_slowdown_mean"`
	DebugQueueP99     float64 `json:"debug_queue_p99_s"`
	InferSLO          float64 `json:"inference_slo_attainment"` // 1 - unmet replica-seconds / demanded
	PreemptLostGPUh   float64 `json:"gpu_hours_lost_to_preemption"`
	Preemptions       int     `json:"preemptions"`
	MakespanDays      float64 `json:"makespan_days"`
}

func New(cfg Config, f Features, name string, generalNodes int) *Sim {
	inferNodes := int(math.Ceil(float64(cfg.InferPeakGPUs) / float64(cfg.GPUsPer)))
	s := &Sim{Cfg: cfg, F: f, Name: name, C: NewCluster(generalNodes, inferNodes, cfg.GPUsPer, cfg.RackSize, cfg.ShareMax), batch: map[string]*run{}, active: map[*run]struct{}{}}
	return s
}

// inferDemand is a diurnal curve between InferMinFrac and 1 of the peak.
func (s *Sim) inferDemandAt(t float64) int {
	x := 0.5 - 0.5*math.Cos(2*math.Pi*math.Mod(t, 86400)/86400)
	f := s.Cfg.InferMinFrac + (1-s.Cfg.InferMinFrac)*x
	return int(math.Round(f * float64(s.Cfg.InferPeakGPUs)))
}

func (s *Sim) push(e event) { heap.Push(&s.q, e) }

// Run replays the jobs. Utilization metrics cover [0, horizon]; jobs still
// queued or running at the horizon are simulated to completion so their
// delays count.
func (s *Sim) Run(jobs []trace.Job, horizon float64) Result {
	s.horizon = horizon
	for i := range jobs {
		s.push(event{t: jobs[i].Submit, kind: 0, job: &jobs[i]})
	}
	for t := 0.0; t <= horizon; t += 300 {
		s.push(event{t: t, kind: 2})
	}
	for s.q.Len() > 0 {
		e := heap.Pop(&s.q).(event)
		s.advance(e.t)
		switch e.kind {
		case 0:
			s.submit(e.job)
		case 1:
			if e.r.epoch == e.epoch && e.r.running {
				s.finish(e.r)
			}
		case 2:
			s.inferTarget = s.inferDemandAt(s.now)
			if s.F.DedicatedInfer {
				// a dedicated pool keeps every replica warm: no reload on the
				// morning ramp, idle GPUs at night
				s.inferTarget = s.Cfg.InferPeakGPUs
			}
			if s.F.PredictiveReclaim {
				if n := s.inferDemandAt(s.now + s.Cfg.InferLoad + 300); n > s.inferTarget {
					s.inferTarget = n
				}
			}
		}
		// drain events at the same instant before scheduling
		if s.q.Len() > 0 && s.q[0].t == e.t {
			continue
		}
		s.schedule()
	}
	return s.result(len(jobs), horizon)
}

func (s *Sim) advance(t float64) {
	from, to := s.lastT, math.Min(t, s.horizon)
	if dt := to - from; dt > 0 {
		alloc := float64(s.C.AllocatedGPUs())
		inf, ready := 0, 0
		for _, r := range s.inferRuns {
			inf += r.gpus
			if from >= r.start+s.Cfg.InferLoad {
				ready += r.gpus // a replica serves only after loading its model
			}
		}
		s.allocGPUs += alloc * dt
		s.busyGPUs += (alloc - float64(inf)) * dt
		served := ready
		if d := s.inferDemandAt(from); d < served {
			served = d
		}
		s.effGPUs += (alloc - float64(inf) + float64(served)) * dt // idle warm replicas do not count
		s.fragInt += s.C.Fragmentation() * dt
		d := s.inferDemandAt(from)
		if d > ready {
			s.inferShort += float64(d-ready) * dt
		}
		s.inferDemand += float64(d) * dt
		s.end = to
	}
	if t > s.lastT {
		s.lastT = t
	}
	s.now = t
}

func (s *Sim) newRun(class trace.Class, gpus, nodes int, work, submit float64) *run {
	s.nextID++
	return &run{id: s.nextID, class: class, gpus: gpus, nodes: nodes, work: work, submit: submit, readyAt: submit, first: -1, pending: true}
}

func (s *Sim) submit(j *trace.Job) {
	// Batching is for short evaluations only: members run back to back, so a
	// long member would delay everyone behind it. Runtimes of eval jobs are
	// taken as known (the same assumption backfill makes).
	// It is also adaptive: when a GPU is free right now, the job just runs;
	// batches only form when evaluations would queue anyway.
	if j.Class == trace.Eval && s.F.BatchEval && j.GPUs == 1 && j.Duration <= s.Cfg.BatchJobMax && !s.evalCanStartNow() {
		key := j.User
		if b, ok := s.batch[key]; ok && b.pending && !b.running && s.now < b.readyAt && len(b.jobs) < s.Cfg.BatchMax &&
			b.work+j.Duration*(1-s.Cfg.EvalLoadFrac) <= s.Cfg.BatchWorkMax {
			b.jobs = append(b.jobs, j)
			b.work += j.Duration * (1 - s.Cfg.EvalLoadFrac) // the model is already loaded
			return
		}
		b := s.newRun(trace.Eval, 1, 1, j.Duration, j.Submit)
		b.jobs = []*trace.Job{j}
		b.readyAt = s.now + s.Cfg.BatchWindow
		s.batch[key] = b
		s.pend = append(s.pend, b)
		s.push(event{t: b.readyAt, kind: 3})
		return
	}
	r := s.newRun(j.Class, j.GPUs, j.Nodes, j.Duration, j.Submit)
	r.jobs = []*trace.Job{j}
	s.pend = append(s.pend, r)
}

// evalCanStartNow reports whether a 1-GPU eval would find a GPU (or a
// shareable slot) without waiting.
func (s *Sim) evalCanStartNow() bool {
	probe := &run{class: trace.Eval, gpus: 1, nodes: 1, work: 600}
	for _, n := range s.C.Nodes {
		if !s.eligibleNode(n, probe) {
			continue
		}
		if n.FreeGPUs() > 0 || (s.F.ShareEval && n.SharedSlot(s.C.ShareMax) >= 0) {
			return true
		}
	}
	return false
}

func (s *Sim) finish(r *run) {
	r.running, r.done = false, true
	delete(s.active, r)
	s.release(r)
	s.finished++
	if r.class == trace.Training {
		for _, j := range r.jobs {
			s.trainSlow = append(s.trainSlow, (s.now-j.Submit)/j.Duration)
		}
	}
	if r.class == trace.Eval && len(r.jobs) > 1 {
		s.finished += len(r.jobs) - 1
	}
}

func (s *Sim) release(r *run) {
	for _, p := range r.place {
		g := &s.C.Nodes[p.Node].gpu[p.GPU]
		if r.shared {
			*g--
		} else {
			*g = free
		}
	}
	r.place = nil
	if r.class == trace.Inference {
		for i, x := range s.inferRuns {
			if x == r {
				s.inferRuns = append(s.inferRuns[:i], s.inferRuns[i+1:]...)
				break
			}
		}
	}
}

func (s *Sim) startRun(r *run, place []Placement, shared bool) {
	joined := false
	for _, p := range place {
		g := &s.C.Nodes[p.Node].gpu[p.GPU]
		if shared {
			if *g > 0 {
				joined = true
			}
			*g++
		} else {
			*g = exclusive
		}
	}
	r.place, r.shared, r.running, r.pending = place, shared, true, false
	s.active[r] = struct{}{}
	r.start = s.now
	r.lastCkpt = s.now
	r.borrowing = false
	for _, p := range place {
		if s.C.Nodes[p.Node].Pool == "inference" && r.class != trace.Inference {
			r.borrowing = true
		}
	}
	if r.first < 0 {
		r.first = s.now
		off := 0.0
		for i, j := range r.jobs {
			d := s.now - j.Submit
			switch r.class {
			case trace.Eval:
				// batch members run back to back in one process after one
				// model load: member i starts when the earlier ones are done
				pen := 1.0
				if joined {
					pen = s.Cfg.SharePenalty
				}
				if i > 0 {
					prev := r.jobs[i-1].Duration
					if i > 1 {
						prev *= 1 - s.Cfg.EvalLoadFrac
					}
					off += pen * prev
				}
				s.evalDelays = append(s.evalDelays, d+off)
			case trace.Training:
				s.trainDelays = append(s.trainDelays, d)
			case trace.Debug:
				s.debugDelays = append(s.debugDelays, d)
			}
		}
	}
	// Interference is charged to the job that joins an occupied GPU (an
	// approximation: the incumbent's slowdown is not modelled).
	dur := r.work
	r.slowed = joined
	if joined {
		dur *= s.Cfg.SharePenalty
	}
	r.finishAt = s.now + dur
	r.epoch++
	if r.class != trace.Inference {
		s.push(event{t: r.finishAt, kind: 1, r: r, epoch: r.epoch})
	}
}

// preempt stops a running job; work since its last checkpoint is lost.
func (s *Sim) preempt(r *run) {
	elapsed := s.now - r.start
	speed := 1.0
	if r.slowed {
		speed = 1 / s.Cfg.SharePenalty
	}
	var lost float64
	switch r.class {
	case trace.Training:
		ck := math.Floor(elapsed/s.Cfg.CkptInterval) * s.Cfg.CkptInterval
		lost = elapsed - ck
		r.work -= ck * speed
		r.work += s.Cfg.RestartOverhead
		lost += s.Cfg.RestartOverhead
	default:
		lost = elapsed // eval/debug restart from scratch
	}
	s.lostGPUSec += lost * float64(r.gpus)
	s.preemptions++
	r.preempts++
	s.release(r)
	delete(s.active, r)
	r.running, r.pending = false, true
	r.epoch++
	s.pend = append(s.pend, r)
}

func (s *Sim) eligibleNode(n *Node, r *run) bool {
	if n.Pool == "inference" {
		if r.class == trace.Inference {
			return true
		}
		if s.F.DedicatedInfer {
			return false
		}
		switch s.F.LendTo {
		case "none":
			return false
		case "eval":
			if r.class != trace.Eval { // short, restartable; debug sessions are long-lived
				return false
			}
		}
		if s.F.LeaseAware {
			// The serving demand curve is forecastable: only lend for as long
			// as the pool will not need these GPUs back.
			return r.work*s.Cfg.SharePenalty <= s.lendHorizon()
		}
		return true
	}
	return r.class != trace.Inference // replicas live in the inference pool, which may lend out idle GPUs
}

// lendHorizon is how long the inference pool can lend its currently idle
// GPUs before forecast demand needs them back (cached per instant).
func (s *Sim) lendHorizon() float64 {
	if s.lendAt == s.now && s.lendAt > 0 {
		return s.lendVal
	}
	pool := 0
	for _, n := range s.C.Nodes {
		if n.Pool == "inference" {
			pool += len(n.gpu)
		}
	}
	borrowed := 0
	for r := range s.active {
		if r.borrowing {
			borrowed += r.gpus
		}
	}
	h := 0.0
	for ; h < 6*3600; h += 300 {
		if s.inferDemandAt(s.now+h+s.Cfg.InferLoad)+borrowed+1 > pool {
			break
		}
	}
	s.lendAt, s.lendVal = s.now, h
	return h
}

// place finds GPUs for r under the placement policy, or nil.
func (s *Sim) place(r *run, blocked map[int]bool) ([]Placement, bool) {
	per := s.Cfg.GPUsPer
	// shared eval slot first: join a GPU another 1-GPU eval already uses
	if s.F.ShareEval && r.class == trace.Eval && r.gpus == 1 {
		for _, n := range s.C.Nodes {
			if blocked[n.ID] || !s.eligibleNode(n, r) {
				continue
			}
			if g := n.SharedSlot(s.C.ShareMax); g >= 0 {
				return []Placement{{n.ID, g}}, true
			}
		}
	}
	shareable := s.F.ShareEval && r.class == trace.Eval && r.gpus == 1
	if r.gpus >= per || r.nodes > 1 {
		// gang over whole nodes
		need := int(math.Ceil(float64(r.gpus) / float64(per)))
		if r.nodes > need {
			need = r.nodes
		}
		perNode := int(math.Ceil(float64(r.gpus) / float64(need)))
		var cands []*Node
		for _, n := range s.C.Nodes {
			if !blocked[n.ID] && s.eligibleNode(n, r) && n.FreeGPUs() >= perNode {
				cands = append(cands, n)
			}
		}
		if len(cands) < need {
			return nil, false
		}
		// prefer fully free nodes in the same rack (topology), then fewest racks
		sort.SliceStable(cands, func(i, j int) bool {
			if s.F.Placement == "frag" || s.F.Placement == "binpack" {
				fi, fj := cands[i].FreeGPUs(), cands[j].FreeGPUs()
				if fi != fj {
					return fi < fj // use partially free nodes that still fit first
				}
				return cands[i].Rack < cands[j].Rack
			}
			return cands[i].ID < cands[j].ID
		})
		var out []Placement
		for _, n := range cands[:need] {
			c := 0
			for g, st := range n.gpu {
				if st == free && c < perNode {
					out = append(out, Placement{n.ID, g})
					c++
				}
			}
		}
		return out, false
	}
	// small job on one node
	best, bestScore := -1, math.Inf(1)
	for _, n := range s.C.Nodes {
		if blocked[n.ID] || !s.eligibleNode(n, r) {
			continue
		}
		f := n.FreeGPUs()
		if f < r.gpus {
			continue
		}
		var sc float64
		switch s.F.Placement {
		case "firstfit":
			sc = float64(n.ID)
		case "spread":
			sc = -float64(f) // least allocated (kube-scheduler default)
		case "binpack":
			sc = float64(f)
		case "frag":
			// FGD-style: minimise the increase in stranded GPUs, weighted by how
			// likely each job size is; breaking a full node costs the most.
			sc = s.fragDelta(f, r.gpus)
			if n.Pool == "inference" {
				sc += 0.5 // borrow inference nodes only when it is no worse
			}
		}
		if sc < bestScore {
			best, bestScore = n.ID, sc
		}
	}
	if best < 0 {
		return nil, false
	}
	var out []Placement
	for g, st := range s.C.Nodes[best].gpu {
		if st == free && len(out) < r.gpus {
			out = append(out, Placement{best, g})
		}
	}
	if shareable {
		// opens a shareable slot (occupancy 1, full speed); a later eval may join it
		return out, true
	}
	return out, false
}

// Job-size mix used by the fragmentation score: share of GPU demand that comes
// in whole-node gangs vs small jobs (from the Acme traces: most GPU time is
// in multi-node training).
var sizeMix = []struct {
	gpus int
	p    float64
}{{1, 0.15}, {2, 0.05}, {4, 0.05}, {8, 0.75}}

func (s *Sim) fragDelta(free, take int) float64 {
	frag := func(f int) float64 {
		var x float64
		for _, m := range sizeMix {
			if f < m.gpus {
				x += m.p * float64(f) // free GPUs a job of this size cannot use
			}
		}
		return x
	}
	return frag(free-take) - frag(free)
}

// schedule is one pass: inference first, then pending work in submit order.
func (s *Sim) schedule() {
	s.scheduleInference()
	// pending order: by submit time (oldest first); eval batches wait for readyAt
	sort.SliceStable(s.pend, func(i, j int) bool {
		if s.F.EvalFirst {
			ei, ej := s.pend[i].class == trace.Eval, s.pend[j].class == trace.Eval
			if ei != ej {
				return ei
			}
		}
		return s.pend[i].submit < s.pend[j].submit
	})
	blocked := map[int]bool{}
	freeAll := 0
	for _, n := range s.C.Nodes {
		freeAll += n.FreeGPUs()
	}
	var resvT float64 = -1
	var resvNodes map[int]bool
	kept := s.pend[:0]
	scanned := 0
	stop := false
	for _, r := range s.pend {
		if !r.pending || r.done {
			continue
		}
		if stop || scanned >= s.Cfg.ScanLimit || s.now < r.readyAt {
			kept = append(kept, r)
			continue
		}
		scanned++
		if r.gpus > freeAll && !(s.F.ShareEval && r.class == trace.Eval && r.gpus == 1) {
			kept = append(kept, r) // cannot fit anywhere right now
			if s.F.StrictFIFO {
				stop = true
			}
			if s.F.Reservation && resvNodes == nil && r.class == trace.Training && r.gpus >= s.Cfg.GPUsPer {
				resvT, resvNodes = s.reserve(r)
			}
			continue
		}
		// EASY backfill: a job may use reserved nodes only if it ends before the reservation.
		bl := blocked
		if resvNodes != nil && !(s.now+r.work*s.Cfg.SharePenalty <= resvT) {
			bl = resvNodes
		}
		pl, shared := s.place(r, bl)
		if pl != nil {
			s.startRun(r, pl, shared)
			if !shared {
				freeAll -= len(pl)
			}
			if r.class == trace.Eval && len(r.jobs) > 0 && s.batch[r.jobs[0].User] == r {
				delete(s.batch, r.jobs[0].User)
			}
			continue
		}
		kept = append(kept, r)
		if s.F.StrictFIFO {
			stop = true
			continue
		}
		if s.F.Reservation && resvNodes == nil && r.class == trace.Training && r.gpus >= s.Cfg.GPUsPer {
			resvT, resvNodes = s.reserve(r)
		}
	}
	s.pend = kept
}

// reserve finds when the gang r could start if nobody else started, and
// which nodes it would use, from the running jobs' end times.
func (s *Sim) reserve(r *run) (float64, map[int]bool) {
	per := s.Cfg.GPUsPer
	need := int(math.Ceil(float64(r.gpus) / float64(per)))
	freeAt := make([]float64, len(s.C.Nodes)) // when each node becomes fully free
	for _, n := range s.C.Nodes {
		if !s.eligibleNode(n, r) {
			freeAt[n.ID] = math.Inf(1)
		}
	}
	for x := range s.active {
		end := x.finishAt
		if x.class == trace.Inference {
			end = math.Inf(1)
		}
		for _, p := range x.place {
			if end > freeAt[p.Node] {
				freeAt[p.Node] = end
			}
		}
	}
	idx := make([]int, len(freeAt))
	for i := range idx {
		idx[i] = i
	}
	sort.Slice(idx, func(a, b int) bool { return freeAt[idx[a]] < freeAt[idx[b]] })
	if need > len(idx) || math.IsInf(freeAt[idx[need-1]], 1) {
		return -1, nil
	}
	nodes := map[int]bool{}
	for _, i := range idx[:need] {
		nodes[i] = true
	}
	return freeAt[idx[need-1]], nodes
}

func (s *Sim) scheduleInference() {
	target := s.inferTarget
	cur := 0
	for _, r := range s.inferRuns {
		cur += r.gpus
	}
	// scale down: newest replicas first
	for cur > target && len(s.inferRuns) > 0 {
		r := s.inferRuns[len(s.inferRuns)-1]
		r.running, r.done = false, true
		delete(s.active, r)
		s.release(r)
		cur--
	}
	for cur < target {
		r := s.newRun(trace.Inference, 1, 1, math.Inf(1), s.now)
		pl, _ := s.place(r, nil)
		if pl == nil && !s.F.DedicatedInfer {
			if !s.reclaim(1) {
				r.done = true
				break
			}
			pl, _ = s.place(r, nil)
		}
		if pl == nil {
			r.done = true
			break
		}
		r.jobs = nil
		s.startRun(r, pl, false)
		s.inferRuns = append(s.inferRuns, r)
		cur++
	}
}

// reclaim preempts borrowers on inference-pool nodes until n GPUs are free there.
func (s *Sim) reclaim(n int) bool {
	var victims []*run
	for r := range s.active {
		if r.borrowing {
			victims = append(victims, r)
		}
	}
	sort.Slice(victims, func(i, j int) bool { return victims[i].id < victims[j].id }) // determinism
	if len(victims) == 0 {
		return false
	}
	lost := func(r *run) float64 {
		el := s.now - r.start
		if r.class == trace.Training {
			el = el - math.Floor(el/s.Cfg.CkptInterval)*s.Cfg.CkptInterval + s.Cfg.RestartOverhead
		}
		return el * float64(r.gpus)
	}
	switch s.F.Preempt {
	case "checkpoint":
		sort.Slice(victims, func(i, j int) bool { return lost(victims[i]) < lost(victims[j]) })
	default: // newest first
		sort.Slice(victims, func(i, j int) bool { return victims[i].start > victims[j].start })
	}
	freed := 0
	for _, v := range victims {
		if freed >= n {
			break
		}
		for _, p := range v.place {
			if s.C.Nodes[p.Node].Pool == "inference" {
				freed++
			}
		}
		s.preempt(v)
	}
	return freed > 0
}

func pct(xs []float64, q float64) float64 {
	if len(xs) == 0 {
		return 0
	}
	sort.Float64s(xs)
	return xs[int(q*float64(len(xs)-1))]
}

func mean(xs []float64) float64 {
	if len(xs) == 0 {
		return 0
	}
	var t float64
	for _, x := range xs {
		t += x
	}
	return t / float64(len(xs))
}

func (s *Sim) result(jobs int, horizon float64) Result {
	span := s.end
	if span <= 0 {
		span = 1
	}
	cap := float64(s.C.TotalGPUs()) * span
	r := Result{Policy: s.Name, Jobs: jobs, Finished: s.finished, DaysSimulated: horizon / 86400, GPUs: s.C.TotalGPUs(),
		AllocRatio: s.allocGPUs / cap, NonInferAlloc: s.busyGPUs / cap, EffectiveUtil: s.effGPUs / cap, Fragmentation: s.fragInt / span,
		EvalQueueP50: pct(s.evalDelays, 0.5), EvalQueueP99: pct(s.evalDelays, 0.99), EvalQueueMean: mean(s.evalDelays),
		TrainQueueP50: pct(s.trainDelays, 0.5), TrainQueueP99: pct(s.trainDelays, 0.99), TrainSlowdownMean: mean(s.trainSlow),
		DebugQueueP99: pct(s.debugDelays, 0.99), PreemptLostGPUh: s.lostGPUSec / 3600, Preemptions: s.preemptions,
		MakespanDays: span / 86400}
	if s.inferDemand > 0 {
		r.InferSLO = 1 - s.inferShort/s.inferDemand
	}
	return r
}
