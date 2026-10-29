package trace

import (
	"encoding/csv"
	"fmt"
	"io"
	"os"
	"strconv"
	"strings"
)

// Alibaba GPU cluster trace v2023 (Weng et al., "Beware of Fragmentation",
// ATC '23): 1,213 GPU nodes with heterogeneous GPU counts and models, and
// 8,152 tasks with CPU, memory, whole-GPU counts and fractional GPU shares.

type AliNode struct {
	Name     string
	CPUMilli int64
	MemMiB   int64
	GPUs     int
	Model    string
}

type AliPod struct {
	Name     string
	CPUMilli int64
	MemMiB   int64
	NumGPU   int   // 0..8
	GPUMilli int64 // per-GPU share for num_gpu == 1 (1..1000); 1000 for multi-GPU
	Spec     []string
}

func readCSV(path string) ([]map[string]string, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()
	r := csv.NewReader(f)
	hdr, err := r.Read()
	if err != nil {
		return nil, err
	}
	var out []map[string]string
	for {
		rec, err := r.Read()
		if err == io.EOF {
			break
		}
		if err != nil {
			return nil, err
		}
		m := map[string]string{}
		for i, h := range hdr {
			if i < len(rec) {
				m[h] = rec[i]
			}
		}
		out = append(out, m)
	}
	return out, nil
}

func LoadAliNodes(path string) ([]AliNode, error) {
	rows, err := readCSV(path)
	if err != nil {
		return nil, err
	}
	var ns []AliNode
	for _, r := range rows {
		cpu, _ := strconv.ParseInt(r["cpu_milli"], 10, 64)
		mem, _ := strconv.ParseInt(r["memory_mib"], 10, 64)
		g, _ := strconv.Atoi(r["gpu"])
		if g == 0 {
			continue
		}
		ns = append(ns, AliNode{Name: r["sn"], CPUMilli: cpu, MemMiB: mem, GPUs: g, Model: r["model"]})
	}
	if len(ns) == 0 {
		return nil, fmt.Errorf("%s: no GPU nodes", path)
	}
	return ns, nil
}

func LoadAliPods(path string) ([]AliPod, error) {
	rows, err := readCSV(path)
	if err != nil {
		return nil, err
	}
	var ps []AliPod
	for _, r := range rows {
		cpu, _ := strconv.ParseInt(r["cpu_milli"], 10, 64)
		mem, _ := strconv.ParseInt(r["memory_mib"], 10, 64)
		g, _ := strconv.Atoi(r["num_gpu"])
		gm, _ := strconv.ParseInt(r["gpu_milli"], 10, 64)
		if g != 1 || gm <= 0 || gm > 1000 {
			gm = 1000
		}
		p := AliPod{Name: r["name"], CPUMilli: cpu, MemMiB: mem, NumGPU: g, GPUMilli: gm}
		if s := strings.TrimSpace(r["gpu_spec"]); s != "" && s != "nan" {
			p.Spec = strings.Split(s, "|")
		}
		ps = append(ps, p)
	}
	return ps, nil
}
