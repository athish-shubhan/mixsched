// Command mixsched replays a public GPU-cluster trace under scheduling
// policies and prints the metrics as JSON lines.
//
//	mixsched -trace data/trace_kalos.csv -nodes 302 -policies fifo,binpack,kueue,mixsched
package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"log"
	"os"
	"runtime"
	"strings"
	"sync"
	"time"

	"mixsched/sim"
	"mixsched/trace"
)

func main() {
	if len(os.Args) > 1 && os.Args[1] == "fill" {
		fill(os.Args[2:])
		return
	}
	if len(os.Args) > 1 && os.Args[1] == "fgd" {
		fgd(os.Args[2:])
		return
	}
	path := flag.String("trace", "data/trace_kalos.csv", "Acme job trace CSV")
	nodes := flag.Int("nodes", 302, "general-pool 8-GPU nodes (Kalos had 302, Seren 286)")
	from := flag.Float64("from-day", 0, "first day of the replay window")
	days := flag.Float64("days", 0, "days to replay (0 = whole trace)")
	policies := flag.String("policies", "fifo,binpack,kueue,mixsched", "comma list (ablations: mixsched-nofrag, mixsched+resv, -noshare, -nobatch, -lendall, -newest, -nopredict, -nolease, -noevalfirst)")
	inferPeak := flag.Int("infer-peak", 384, "peak inference demand in GPUs (synthetic diurnal serving load)")
	out := flag.String("out", "", "append JSON lines here (default stdout)")
	flag.Parse()

	jobs, err := trace.LoadAcme(*path)
	if err != nil {
		log.Fatal(err)
	}
	if *days > 0 {
		jobs = trace.Window(jobs, *from, *days)
	}
	horizon := 0.0
	for _, j := range jobs {
		if j.Submit > horizon {
			horizon = j.Submit
		}
	}
	w := os.Stdout
	if *out != "" {
		f, err := os.OpenFile(*out, os.O_CREATE|os.O_APPEND|os.O_WRONLY, 0o644)
		if err != nil {
			log.Fatal(err)
		}
		defer f.Close()
		w = f
	}
	enc := json.NewEncoder(w)
	var mu sync.Mutex
	var wg sync.WaitGroup
	sem := make(chan struct{}, runtime.NumCPU())
	for _, name := range strings.Split(*policies, ",") {
		f, err := sim.Ablation(name)
		if err != nil {
			log.Fatal(err)
		}
		wg.Add(1)
		sem <- struct{}{}
		go func(name string, f sim.Features) {
			defer wg.Done()
			defer func() { <-sem }()
			cfg := sim.DefaultConfig()
			cfg.InferPeakGPUs = *inferPeak
			js := append([]trace.Job(nil), jobs...)
			t0 := time.Now()
			r := sim.New(cfg, f, name, *nodes).Run(js, horizon)
			mu.Lock()
			enc.Encode(struct {
				sim.Result
				Trace   string  `json:"trace"`
				FromDay float64 `json:"from_day"`
				WallSec float64 `json:"wall_sec"`
			}{r, *path, *from, time.Since(t0).Seconds()})
			fmt.Fprintf(os.Stderr, "%-18s alloc=%.3f eff=%.3f frag=%.3f evalQ p50/p99=%.0f/%.0fs trainQ p50/p99=%.0f/%.0fs slowdown=%.2f inferSLO=%.4f lost=%.0f GPUh (%d preemptions) done=%d/%d (%.1fs)\n",
				name, r.AllocRatio, r.EffectiveUtil, r.Fragmentation, r.EvalQueueP50, r.EvalQueueP99, r.TrainQueueP50, r.TrainQueueP99, r.TrainSlowdownMean, r.InferSLO, r.PreemptLostGPUh, r.Preemptions, r.Finished, r.Jobs, time.Since(t0).Seconds())
			mu.Unlock()
		}(name, f)
	}
	wg.Wait()
}

// fill: the simulator side of the KWOK fidelity check (sim/fill.go).
func fill(args []string) {
	fs := flag.NewFlagSet("fill", flag.ExitOnError)
	nodes := fs.Int("nodes", 1000, "8-GPU nodes")
	share := fs.Float64("fill", 0.9, "requested GPUs as a share of capacity")
	seeds := fs.Int("seeds", 1, "workload seeds (1..n)")
	policies := fs.String("policies", "spread,binpack,frag", "placement policies")
	fs.Parse(args)
	enc := json.NewEncoder(os.Stdout)
	for s := 1; s <= *seeds; s++ {
		for _, p := range strings.Split(*policies, ",") {
			r := sim.Fill(p, *nodes, *share, uint64(s))
			enc.Encode(struct {
				sim.FillResult
				Seed int `json:"seed"`
			}{r, s})
		}
	}
}

// fgd: workload inflation on the Alibaba GPU trace v2023 (sim/fgd.go).
func fgd(args []string) {
	fs := flag.NewFlagSet("fgd", flag.ExitOnError)
	nodesPath := fs.String("nodes", "data/alibaba/openb_node_list_gpu_node.csv", "node list")
	podsPath := fs.String("pods", "data/alibaba/openb_pod_list_default.csv", "pod list")
	policies := fs.String("policies", "random,dotprod,clustering,packing,bestfit,mixsched,fgd", "policies")
	seeds := fs.Int("seeds", 10, "sampling seeds")
	maxA := fs.Float64("max", 1.3, "stop when arrived GPU demand reaches this share of capacity")
	out := fs.String("out", "", "append JSON lines here")
	fs.Parse(args)
	nodes, err := trace.LoadAliNodes(*nodesPath)
	if err != nil {
		log.Fatal(err)
	}
	pods, err := trace.LoadAliPods(*podsPath)
	if err != nil {
		log.Fatal(err)
	}
	w := os.Stdout
	if *out != "" {
		f, err := os.OpenFile(*out, os.O_CREATE|os.O_APPEND|os.O_WRONLY, 0o644)
		if err != nil {
			log.Fatal(err)
		}
		defer f.Close()
		w = f
	}
	enc := json.NewEncoder(w)
	var mu sync.Mutex
	var wg sync.WaitGroup
	sem := make(chan struct{}, runtime.NumCPU())
	for _, p := range strings.Split(*policies, ",") {
		for s := 1; s <= *seeds; s++ {
			wg.Add(1)
			sem <- struct{}{}
			go func(p string, s int) {
				defer wg.Done()
				defer func() { <-sem }()
				t0 := time.Now()
				r := sim.Inflate(p, nodes, pods, uint64(s), *maxA)
				mu.Lock()
				enc.Encode(struct {
					sim.InflationResult
					Trace string  `json:"trace"`
					Sec   float64 `json:"sec"`
				}{r, *podsPath, time.Since(t0).Seconds()})
				mu.Unlock()
			}(p, s)
		}
	}
	wg.Wait()
}
