// Package trace loads public GPU-cluster job traces into a common job model.
//
// Acme (Hu et al., NSDI '24, github.com/InternLM/AcmeTrace, CC-BY-4.0): job
// logs of Shanghai AI Lab's Seren and Kalos LLM clusters, with each job's
// type (Pretrain, SFT, MLLM, Eval, Debug, Other), GPU count, node count,
// submit/start/end times and final state.
package trace

import (
	"encoding/csv"
	"fmt"
	"io"
	"os"
	"sort"
	"strconv"
	"time"
)

type Class uint8

const (
	Training  Class = iota // Pretrain, SFT, MLLM, Other: gang-scheduled, long, checkpointable
	Eval                   // short, mostly 1 GPU, tied to checkpoints, latency-sensitive for researchers
	Debug                  // interactive development
	Inference              // SLO-bound serving replicas (synthetic: Acme has no serving fleet)
)

func (c Class) String() string { return [...]string{"training", "eval", "debug", "inference"}[c] }

type Job struct {
	ID       int
	Name     string
	User     string
	Type     string // original Acme type
	Class    Class
	Submit   float64 // seconds since trace start
	GPUs     int
	Nodes    int
	Duration float64 // seconds of execution in the original cluster
	State    string
	// QueuedOrig is how long the job waited in the original cluster (start - submit).
	QueuedOrig float64
}

func classOf(t string) Class {
	switch t {
	case "Eval":
		return Eval
	case "Debug":
		return Debug
	}
	return Training
}

func parseTime(s string) (time.Time, error) {
	for _, layout := range []string{"2006-01-02 15:04:05-07:00", "2006-01-02 15:04:05Z07:00"} {
		if t, err := time.Parse(layout, s); err == nil {
			return t, nil
		}
	}
	return time.Time{}, fmt.Errorf("bad time %q", s)
}

// LoadAcme returns the GPU jobs of one Acme trace, sorted by submit time,
// with times relative to the first submission. Jobs that never ran (no start
// time) or ran for 0 s are dropped.
func LoadAcme(path string) ([]Job, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()
	r := csv.NewReader(f)
	head, err := r.Read()
	if err != nil {
		return nil, err
	}
	col := map[string]int{}
	for i, h := range head {
		col[h] = i
	}
	for _, need := range []string{"job_id", "user", "node_num", "gpu_num", "type", "state", "submit_time", "start_time", "duration"} {
		if _, ok := col[need]; !ok {
			return nil, fmt.Errorf("%s: missing column %s", path, need)
		}
	}
	var jobs []Job
	var t0 time.Time
	for {
		rec, err := r.Read()
		if err == io.EOF {
			break
		}
		if err != nil {
			return nil, err
		}
		g, _ := strconv.ParseFloat(rec[col["gpu_num"]], 64)
		if g < 1 {
			continue
		}
		dur, _ := strconv.ParseFloat(rec[col["duration"]], 64)
		if dur <= 0 || rec[col["start_time"]] == "" {
			continue
		}
		sub, err := parseTime(rec[col["submit_time"]])
		if err != nil {
			return nil, err
		}
		st, err := parseTime(rec[col["start_time"]])
		if err != nil {
			return nil, err
		}
		n, _ := strconv.Atoi(rec[col["node_num"]])
		if n < 1 {
			n = 1
		}
		j := Job{Name: rec[col["job_id"]], User: rec[col["user"]], Type: rec[col["type"]], Class: classOf(rec[col["type"]]),
			GPUs: int(g), Nodes: n, Duration: dur, State: rec[col["state"]], QueuedOrig: st.Sub(sub).Seconds()}
		j.Submit = float64(sub.Unix())
		if t0.IsZero() || sub.Before(t0) {
			t0 = sub
		}
		jobs = append(jobs, j)
	}
	for i := range jobs {
		jobs[i].Submit -= float64(t0.Unix())
	}
	sort.SliceStable(jobs, func(a, b int) bool { return jobs[a].Submit < jobs[b].Submit })
	for i := range jobs {
		jobs[i].ID = i
	}
	return jobs, nil
}

// Window keeps jobs submitted in [from, from+span) days, re-based to 0.
func Window(jobs []Job, fromDay, days float64) []Job {
	lo, hi := fromDay*86400, (fromDay+days)*86400
	var out []Job
	for _, j := range jobs {
		if j.Submit >= lo && j.Submit < hi {
			j.Submit -= lo
			j.ID = len(out)
			out = append(out, j)
		}
	}
	return out
}
