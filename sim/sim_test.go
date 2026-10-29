package sim

import (
	"testing"

	"mixsched/trace"
)

func cfgNoInfer() Config {
	c := DefaultConfig()
	c.InferPeakGPUs = 8 // one inference node
	c.InferMinFrac = 1
	return c
}

func job(id int, class trace.Class, submit float64, gpus int, dur float64, user string) trace.Job {
	n := 1
	if gpus > 8 {
		n = gpus / 8
	}
	return trace.Job{ID: id, Class: class, Submit: submit, GPUs: gpus, Nodes: n, Duration: dur, User: user}
}

// Every job finishes, and with an idle cluster nobody waits.
func TestIdleClusterNoQueueing(t *testing.T) {
	for name := range Policies {
		jobs := []trace.Job{job(0, trace.Training, 0, 16, 3600, "a"), job(1, trace.Eval, 10, 1, 100, "b"), job(2, trace.Debug, 20, 2, 50, "c")}
		f, _ := Ablation(name)
		r := New(cfgNoInfer(), f, name, 4).Run(jobs, 4000)
		if r.Finished != 3 || r.TrainQueueP99 != 0 || r.EvalQueueP99 != 0 {
			t.Fatalf("%s: %+v", name, r)
		}
	}
}

// Strict FIFO blocks small jobs behind a gang that does not fit; best-effort
// FIFO lets them through.
func TestHeadOfLineBlocking(t *testing.T) {
	jobs := []trace.Job{
		job(0, trace.Training, 0, 8, 1000, "a"),  // one of the two nodes
		job(1, trace.Training, 1, 16, 1000, "a"), // needs both: blocked until 1000
		job(2, trace.Eval, 2, 1, 10, "b"),        // fits on the free node right away
	}
	jobs = append(jobs, job(3, trace.Eval, 1500, 1, 10, "b"))
	fs := New(cfgNoInfer(), Policies["fifo"], "fifo", 2)
	if r := fs.Run(append([]trace.Job(nil), jobs...), 3000); r.Finished != 4 {
		t.Fatalf("fifo did not finish all jobs: %+v", r)
	}
	// strict FIFO: the eval submitted at t=2 waits for job 1, which runs 1000-2000
	if d := maxOf(fs.evalDelays); d < 1990 {
		t.Fatalf("strict FIFO should block the eval behind the gang until t=2000, max delay %v", d)
	}
	bp := New(cfgNoInfer(), Policies["binpack"], "binpack", 2)
	bp.Run(append([]trace.Job(nil), jobs...), 3000)
	// best-effort FIFO: the blocked gang does not hold the eval back
	if d := bp.evalDelays[0]; d != 0 {
		t.Fatalf("best-effort FIFO should start the eval immediately, delay %v", d)
	}
}

func maxOf(xs []float64) float64 {
	m := 0.0
	for _, x := range xs {
		if x > m {
			m = x
		}
	}
	return m
}

// Preempting a training job loses only the work since its last checkpoint
// plus the restart overhead.
func TestCheckpointLoss(t *testing.T) {
	s := New(cfgNoInfer(), Policies["mixsched"], "t", 1)
	r := s.newRun(trace.Training, 8, 1, 10000, 0)
	pl, _ := s.place(r, nil)
	s.startRun(r, pl, false)
	s.now = 4000 // 2 checkpoints (1800 s) done, 400 s since the last
	s.preempt(r)
	want := (400 + s.Cfg.RestartOverhead) * 8
	if s.lostGPUSec != want {
		t.Fatalf("lost %v GPU-s, want %v", s.lostGPUSec, want)
	}
	if r.work != 10000-3600+s.Cfg.RestartOverhead {
		t.Fatalf("remaining work %v", r.work)
	}
}

// Batched evals share one model load: total GPU time drops by the load share
// of every member after the first, and members start back to back.
func TestEvalBatchOffsets(t *testing.T) {
	c := cfgNoInfer()
	f := Policies["mixsched"]
	s := New(c, f, "t", 1)
	// fill the only general node so that evals must queue and batch
	jobs := []trace.Job{job(0, trace.Training, 0, 8, 500, "a")}
	for i := 1; i <= 3; i++ {
		jobs = append(jobs, job(i, trace.Eval, float64(i), 1, 100, "u"))
	}
	f.LendTo = "none" // keep evals off the inference node for this test
	s = New(c, f, "t", 1)
	r := s.Run(jobs, 2000)
	if r.Finished != 4 {
		t.Fatalf("%+v", r)
	}
	// batch starts at 500; members start at 500, 600, 600+70.5
	want := []float64{500 - 1, 600 - 2, 670.5 - 3}
	for i, d := range want {
		if diff := s.evalDelays[i] - d; diff > 1e-6 || diff < -1e-6 {
			t.Fatalf("member %d delay %v, want %v (all %v)", i, s.evalDelays[i], d, s.evalDelays)
		}
	}
}

func TestDeterministic(t *testing.T) {
	var jobs []trace.Job
	for i := 0; i < 300; i++ {
		c := trace.Eval
		g := 1
		if i%7 == 0 {
			c, g = trace.Training, 16
		}
		jobs = append(jobs, job(i, c, float64(i*20), g, float64(100+i%13*200), "u"+string(rune('a'+i%5))))
	}
	a := New(DefaultConfig(), Policies["mixsched"], "m", 3).Run(append([]trace.Job(nil), jobs...), 20000)
	b := New(DefaultConfig(), Policies["mixsched"], "m", 3).Run(append([]trace.Job(nil), jobs...), 20000)
	if a != b {
		t.Fatalf("non-deterministic:\n%+v\n%+v", a, b)
	}
}
