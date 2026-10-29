// Command mixsched-bench drives a real kube-apiserver + kube-scheduler with
// KWOK fake GPU nodes: it creates N nodes advertising nvidia.com/gpu, fills
// the cluster with GPU pods sampled from a job-size mix (single-GPU evals,
// small jobs and multi-node training gangs), and measures
//
//   - scheduler throughput: pods bound per second while the queue is full,
//   - decision latency: kube-scheduler's own scheduling_attempt_duration
//     histogram (p50/p99), scraped from its /metrics,
//   - placement quality: GPU allocation ratio, the share of free GPUs
//     stranded on partially used nodes (node fragmentation), and how many
//     whole-node (8-GPU) jobs still fit at the end.
//
// It is built inside a Kubernetes checkout (bench/k8s/build.sh) so it uses
// the same vendored client-go as the scheduler.
package main

import (
	"bufio"
	"context"
	"crypto/tls"
	"encoding/json"
	"flag"
	"fmt"
	"log"
	"math"
	"math/rand/v2"
	"net/http"
	"os"
	"sort"
	"strconv"
	"strings"
	"sync"
	"sync/atomic"
	"time"

	v1 "k8s.io/api/core/v1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/fields"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/apimachinery/pkg/watch"
	"k8s.io/client-go/kubernetes"
	"k8s.io/client-go/tools/clientcmd"
)

const gpu = v1.ResourceName("nvidia.com/gpu")

type job struct {
	class string
	gpus  int64 // per pod
	pods  int   // gang members (1 for non-gang)
}

// sizes follows the shape of the Acme Seren trace: most jobs are 1-GPU
// evaluations, most GPU demand is in 8-GPU-per-node training gangs.
func sample(r *rand.Rand) job {
	x := r.Float64()
	switch {
	case x < 0.55:
		return job{"eval", 1, 1}
	case x < 0.70:
		return job{"train", 1, 1}
	case x < 0.80:
		return job{"train", 2, 1}
	case x < 0.88:
		return job{"train", 4, 1}
	case x < 0.94:
		return job{"infer", 1, 1}
	default:
		return job{"train", 8, []int{1, 2, 2, 4, 4, 8}[r.IntN(6)]}
	}
}

type result struct {
	Scheduler      string             `json:"scheduler"`
	Nodes          int                `json:"nodes"`
	GPUs           int64              `json:"gpus"`
	PodsCreated    int                `json:"pods_created"`
	PodsBound      int64              `json:"pods_bound"`
	FillSec        float64            `json:"fill_sec"`
	PodsPerSec     float64            `json:"pods_per_sec"`
	AttemptMs      map[string]float64 `json:"scheduling_attempt_ms"`
	ScoreMs        map[string]float64 `json:"score_extension_ms"`
	AllocRatio     float64            `json:"gpu_allocation_ratio"`
	Fragmentation  float64            `json:"node_fragmentation_ratio"`
	FreeNodes      int                `json:"fully_free_nodes"`
	WholeNodeFit   int                `json:"whole_node_jobs_fit_after"`
	Unschedulable  int                `json:"unschedulable_pods"`
	TargetFillFrac float64            `json:"target_fill"`
}

func main() {
	kubeconfig := flag.String("kubeconfig", "", "kubeconfig")
	nodes := flag.Int("nodes", 1000, "fake GPU nodes")
	perNode := flag.Int64("gpus-per-node", 8, "GPUs per node")
	fill := flag.Float64("fill", 0.9, "requested GPUs as a share of capacity")
	metricsURL := flag.String("metrics", "https://127.0.0.1:10259/metrics", "scheduler metrics endpoint")
	name := flag.String("scheduler", "default-scheduler", "schedulerName for the pods")
	label := flag.String("label", "", "name of the scheduler config for the report")
	seed := flag.Uint64("seed", 1, "workload seed")
	out := flag.String("out", "", "append the JSON result here")
	setup := flag.Bool("setup", false, "create the nodes (once per cluster) and exit")
	clean := flag.Bool("clean", false, "delete every pod in the bench namespace and exit")
	flag.Parse()

	cfg, err := clientcmd.BuildConfigFromFlags("", *kubeconfig)
	if err != nil {
		log.Fatal(err)
	}
	cfg.QPS, cfg.Burst = 2000, 4000
	// protobuf, as kube-scheduler itself uses: cheaper to encode and decode on a shared 2-vCPU box
	cfg.ContentType = "application/vnd.kubernetes.protobuf"
	cfg.AcceptContentTypes = "application/vnd.kubernetes.protobuf,application/json"
	cs, err := kubernetes.NewForConfig(cfg)
	if err != nil {
		log.Fatal(err)
	}
	ctx := context.Background()
	ns := "bench"
	if *setup {
		createNodes(ctx, cs, *nodes, *perNode)
		cs.CoreV1().Namespaces().Create(ctx, &v1.Namespace{ObjectMeta: metav1.ObjectMeta{Name: ns}}, metav1.CreateOptions{})
		// no controller-manager runs, so nobody creates the namespace's default
		// service account, and the ServiceAccount admission plugin rejects pods
		// without one
		cs.CoreV1().ServiceAccounts(ns).Create(ctx, &v1.ServiceAccount{ObjectMeta: metav1.ObjectMeta{Name: "default"}}, metav1.CreateOptions{})
		return
	}
	if *clean {
		cleanPods(ctx, cs, ns)
		return
	}

	res := result{Scheduler: *label, Nodes: *nodes, GPUs: int64(*nodes) * *perNode, TargetFillFrac: *fill}
	before := scrape(*metricsURL)

	// Generate the workload up front.
	r := rand.New(rand.NewPCG(*seed, 7))
	var pods []*v1.Pod
	var demand int64
	jobN := 0
	for demand < int64(float64(res.GPUs)**fill) {
		j := sample(r)
		jobN++
		for m := 0; m < j.pods; m++ {
			p := podFor(fmt.Sprintf("j%05d-%d", jobN, m), *name, j, jobN)
			pods = append(pods, p)
			demand += j.gpus
		}
	}
	res.PodsCreated = len(pods)

	// Watch bindings. The watch cache closes a watcher that falls behind a
	// burst of events; the watch is then re-established from the last
	// resourceVersion seen, so a reconnect delivers only the missed events
	// (an empty resourceVersion would replay every existing pod as ADDED).
	// If that version has left the watch cache (410 Gone), it starts over.
	var bound atomic.Int64
	seen := sync.Map{}
	var lastBind atomic.Int64
	var evTypes sync.Map
	stopWatch := make(chan struct{})
	watching := make(chan struct{})
	go func() {
		first := true
		rv := ""
		for {
			w, err := cs.CoreV1().Pods(ns).Watch(ctx, metav1.ListOptions{ResourceVersion: rv, AllowWatchBookmarks: true})
			if err != nil {
				if apierrors.IsGone(err) || apierrors.IsResourceExpired(err) {
					rv = ""
				}
				log.Printf("watch: %v", err)
				time.Sleep(200 * time.Millisecond)
				continue
			}
			if first {
				close(watching)
				first = false
			}
			for ev := range w.ResultChan() {
				n, _ := evTypes.LoadOrStore(string(ev.Type), new(atomic.Int64))
				n.(*atomic.Int64).Add(1)
				if ev.Type == watch.Error {
					if st, ok := ev.Object.(*metav1.Status); ok && st.Code == 410 {
						rv = "" // too old: start over (existing pods come back as ADDED; seen dedups them)
					}
					continue
				}
				p, ok := ev.Object.(*v1.Pod)
				if !ok {
					continue
				}
				rv = p.ResourceVersion
				if (ev.Type != watch.Modified && ev.Type != watch.Added) || p.Spec.NodeName == "" {
					continue
				}
				if _, dup := seen.LoadOrStore(p.Name, true); !dup {
					bound.Add(1)
					lastBind.Store(time.Now().UnixNano())
				}
			}
			select {
			case <-stopWatch:
				return
			default:
				log.Printf("watch closed by the server; re-watching")
			}
		}
	}()
	<-watching

	start := time.Now()
	var wg sync.WaitGroup
	ch := make(chan *v1.Pod)
	for i := 0; i < 32; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for p := range ch {
				for {
					_, err := cs.CoreV1().Pods(ns).Create(ctx, p, metav1.CreateOptions{})
					if err == nil || apierrors.IsAlreadyExists(err) {
						break
					}
					if !apierrors.IsTooManyRequests(err) && !apierrors.IsServerTimeout(err) && !apierrors.IsTimeout(err) {
						log.Fatalf("create %s: %v", p.Name, err) // a rejected pod is a setup bug, not load
					}
					time.Sleep(50 * time.Millisecond)
				}
			}
		}()
	}
	for _, p := range pods {
		ch <- p
	}
	close(ch)
	wg.Wait()
	created := time.Since(start)
	// Wait until bindings stop for 15 s (the rest are unschedulable or waiting gangs).
	for {
		time.Sleep(time.Second)
		lb := lastBind.Load()
		if bound.Load() == int64(len(pods)) || (lb > 0 && time.Since(time.Unix(0, lb)) > 15*time.Second) ||
			(lb == 0 && time.Since(start) > 60*time.Second) {
			break
		}
	}
	close(stopWatch)
	evTypes.Range(func(k, v any) bool { log.Printf("watch events %s: %d", k, v.(*atomic.Int64).Load()); return true })
	end := time.Unix(0, lastBind.Load())
	res.PodsBound = bound.Load()
	res.FillSec = end.Sub(start).Seconds()
	if res.FillSec > 0 {
		res.PodsPerSec = float64(res.PodsBound) / res.FillSec
	}
	log.Printf("created %d pods in %.1fs, %d bound in %.1fs (%.0f pods/s)", len(pods), created.Seconds(), res.PodsBound, res.FillSec, res.PodsPerSec)

	after := scrape(*metricsURL)
	res.AttemptMs = histDelta(before, after, "scheduler_scheduling_attempt_duration_seconds", `result="scheduled"`)
	res.ScoreMs = histDelta(before, after, "scheduler_framework_extension_point_duration_seconds", `extension_point="Score"`)

	// Placement quality from the bound pods.
	used := map[string]int64{}
	pl, err := cs.CoreV1().Pods(ns).List(ctx, metav1.ListOptions{})
	if err != nil {
		log.Fatal(err)
	}
	for _, p := range pl.Items {
		if p.Spec.NodeName != "" {
			for _, c := range p.Spec.Containers {
				q := c.Resources.Requests[gpu]
				used[p.Spec.NodeName] += q.Value()
			}
		} else {
			res.Unschedulable++
		}
	}
	nl, err := cs.CoreV1().Nodes().List(ctx, metav1.ListOptions{FieldSelector: fields.Everything().String()})
	if err != nil {
		log.Fatal(err)
	}
	var allocated, free, stranded int64
	for _, n := range nl.Items {
		q := n.Status.Allocatable[gpu]
		capN := q.Value()
		if capN == 0 {
			continue
		}
		u := used[n.Name]
		allocated += u
		f := capN - u
		free += f
		if f > 0 && f < capN {
			stranded += f
		}
		if f == capN {
			res.FreeNodes++
		}
		res.WholeNodeFit += int(f / *perNode)
	}
	res.AllocRatio = float64(allocated) / float64(res.GPUs)
	if free > 0 {
		res.Fragmentation = float64(stranded) / float64(free)
	}
	b, _ := json.Marshal(res)
	fmt.Println(string(b))
	if *out != "" {
		f, err := os.OpenFile(*out, os.O_CREATE|os.O_APPEND|os.O_WRONLY, 0o644)
		if err == nil {
			f.Write(append(b, '\n'))
			f.Close()
		}
	}
}

var no = false

func podFor(name, scheduler string, j job, jobN int) *v1.Pod {
	lbl := map[string]string{"mixsched.io/class": j.class, "app": "bench"}
	if j.pods > 1 {
		lbl["mixsched.io/gang"] = fmt.Sprintf("j%05d", jobN)
		lbl["mixsched.io/gang-size"] = strconv.Itoa(j.pods)
	}
	req := v1.ResourceList{gpu: *resource.NewQuantity(j.gpus, resource.DecimalSI),
		v1.ResourceCPU: resource.MustParse("1"), v1.ResourceMemory: resource.MustParse("1Gi")}
	return &v1.Pod{
		ObjectMeta: metav1.ObjectMeta{Name: name, Labels: lbl},
		Spec: v1.PodSpec{
			AutomountServiceAccountToken: &no,
			SchedulerName:                scheduler,
			Containers:                   []v1.Container{{Name: "c", Image: "fake", Resources: v1.ResourceRequirements{Requests: req, Limits: v1.ResourceList{gpu: req[gpu]}}}},
			Tolerations:                  []v1.Toleration{{Key: "kwok.x-k8s.io/node", Operator: v1.TolerationOpExists, Effect: v1.TaintEffectNoSchedule}},
			NodeSelector:                 map[string]string{"type": "kwok"},
		},
	}
}

var kwokTaint = []byte(`{"spec":{"taints":[{"key":"kwok.x-k8s.io/node","value":"fake","effect":"NoSchedule"}]}}`)

func createNodes(ctx context.Context, cs *kubernetes.Clientset, n int, perNode int64) {
	var wg sync.WaitGroup
	ch := make(chan int)
	for i := 0; i < 32; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for i := range ch {
				node := &v1.Node{
					ObjectMeta: metav1.ObjectMeta{
						Name:        fmt.Sprintf("kwok-gpu-%05d", i),
						Labels:      map[string]string{"type": "kwok", "kubernetes.io/hostname": fmt.Sprintf("kwok-gpu-%05d", i), "topology.kubernetes.io/zone": fmt.Sprintf("rack-%03d", i/32)},
						Annotations: map[string]string{"kwok.x-k8s.io/node": "fake", "node.alpha.kubernetes.io/ttl": "0"},
					},
					Spec: v1.NodeSpec{Taints: []v1.Taint{{Key: "kwok.x-k8s.io/node", Value: "fake", Effect: v1.TaintEffectNoSchedule}}},
					Status: v1.NodeStatus{
						Capacity: v1.ResourceList{gpu: *resource.NewQuantity(perNode, resource.DecimalSI), v1.ResourceCPU: resource.MustParse("128"),
							v1.ResourceMemory: resource.MustParse("2Ti"), v1.ResourcePods: resource.MustParse("110")},
						Allocatable: v1.ResourceList{gpu: *resource.NewQuantity(perNode, resource.DecimalSI), v1.ResourceCPU: resource.MustParse("128"),
							v1.ResourceMemory: resource.MustParse("2Ti"), v1.ResourcePods: resource.MustParse("110")},
					},
				}
				for {
					_, err := cs.CoreV1().Nodes().Create(ctx, node, metav1.CreateOptions{})
					if err == nil || apierrors.IsAlreadyExists(err) {
						break
					}
					time.Sleep(100 * time.Millisecond)
				}
				// The TaintNodesByCondition admission plugin adds node.kubernetes.io/not-ready
				// on create; the node lifecycle controller would remove it once KWOK reports
				// Ready, but no controller-manager runs here, so drop it ourselves.
				for {
					_, err := cs.CoreV1().Nodes().Patch(ctx, node.Name, types.MergePatchType, kwokTaint, metav1.PatchOptions{})
					if err == nil {
						break
					}
					time.Sleep(100 * time.Millisecond)
				}
			}
		}()
	}
	for i := 0; i < n; i++ {
		ch <- i
	}
	close(ch)
	wg.Wait()
	log.Printf("created %d nodes", n)
}

func cleanPods(ctx context.Context, cs *kubernetes.Clientset, ns string) {
	zero := int64(0)
	cs.CoreV1().Pods(ns).DeleteCollection(ctx, metav1.DeleteOptions{GracePeriodSeconds: &zero}, metav1.ListOptions{})
	for i := 0; i < 120; i++ {
		pl, err := cs.CoreV1().Pods(ns).List(ctx, metav1.ListOptions{Limit: 1})
		if err == nil && len(pl.Items) == 0 {
			return
		}
		time.Sleep(time.Second)
	}
}

// --- Prometheus text parsing (histograms only)

type hist map[string]map[float64]float64 // series labels (without le) -> le -> cumulative count

func scrape(url string) map[string]hist {
	c := &http.Client{Timeout: 10 * time.Second, Transport: &http.Transport{TLSClientConfig: &tls.Config{InsecureSkipVerify: true}}}
	resp, err := c.Get(url)
	if err != nil {
		log.Printf("scrape %s: %v", url, err)
		return nil
	}
	defer resp.Body.Close()
	out := map[string]hist{}
	sc := bufio.NewScanner(resp.Body)
	sc.Buffer(make([]byte, 1<<20), 1<<20)
	for sc.Scan() {
		line := sc.Text()
		if strings.HasPrefix(line, "#") {
			continue
		}
		i := strings.Index(line, "_bucket{")
		if i < 0 {
			continue
		}
		name := line[:i]
		rest := line[i+8:]
		j := strings.LastIndex(rest, "}")
		if j < 0 {
			continue
		}
		lbls, val := rest[:j], strings.TrimSpace(rest[j+1:])
		var le float64
		var keep []string
		for _, kv := range strings.Split(lbls, ",") {
			if strings.HasPrefix(kv, "le=") {
				s := strings.Trim(kv[3:], `"`)
				if s == "+Inf" {
					le = math.Inf(1)
				} else {
					le, _ = strconv.ParseFloat(s, 64)
				}
			} else {
				keep = append(keep, kv)
			}
		}
		v, _ := strconv.ParseFloat(val, 64)
		h := out[name]
		if h == nil {
			h = hist{}
			out[name] = h
		}
		key := strings.Join(keep, ",")
		if h[key] == nil {
			h[key] = map[float64]float64{}
		}
		h[key][le] = v
	}
	return out
}

// histDelta sums the series of a histogram whose labels contain filter,
// subtracts the earlier scrape and returns p50/p90/p99 in ms (linear
// interpolation inside buckets, as Prometheus' histogram_quantile does).
func histDelta(before, after map[string]hist, name, filter string) map[string]float64 {
	sum := map[float64]float64{}
	for key, b := range after[name] {
		if !strings.Contains(key, filter) {
			continue
		}
		for le, v := range b {
			sum[le] += v - before[name][key][le]
		}
	}
	if len(sum) == 0 {
		return nil
	}
	les := make([]float64, 0, len(sum))
	for le := range sum {
		les = append(les, le)
	}
	sort.Float64s(les)
	total := sum[les[len(les)-1]]
	q := func(p float64) float64 {
		rank := p * total
		prevLe, prevC := 0.0, 0.0
		for _, le := range les {
			c := sum[le]
			if c >= rank {
				if math.IsInf(le, 1) {
					return prevLe * 1000
				}
				if c == prevC {
					return le * 1000
				}
				return (prevLe + (le-prevLe)*(rank-prevC)/(c-prevC)) * 1000
			}
			prevLe, prevC = le, c
		}
		return prevLe * 1000
	}
	return map[string]float64{"count": total, "p50": q(0.5), "p90": q(0.9), "p99": q(0.99)}
}
