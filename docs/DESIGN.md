# MixSched design

## 1. Question

One shared GPU pool runs three kinds of work that want different things:

| class | shape | wants |
|---|---|---|
| training (Pretrain, SFT, MLLM, Other) | gang-scheduled, often whole 8-GPU nodes, hours to days, checkpointed | whole nodes, few interruptions |
| evaluation | tiny (92.8% of Kalos GPU jobs, 0.4% of its GPU time), mostly 1 GPU, minutes, bursty after each checkpoint | short queueing: a researcher is waiting on the score |
| inference | SLO-bound replicas with a diurnal demand curve | capacity when demand rises, no cold starts |

The Acme paper (NSDI '24) found evaluation jobs have the longest queueing
delays despite using almost no GPU time. MixSched asks how far scheduling
alone (no new hardware) can cut evaluation queueing while keeping training
moving and serving within its SLO, on the public Acme traces.

## 2. Simulator

Discrete-event, in Go (`sim/`). Each scheduling pass runs after all events at
the same instant.

- **Cluster**: N general 8-GPU nodes (Seren had 286, Kalos 302) plus a
  48-node inference pool. Racks of 16 nodes.
- **Jobs**: replayed from the Acme job logs: submit time, GPU count, node
  count, type, and the runtime the job had in the real cluster (used as its
  work; failed and cancelled jobs run until the time they ended). Jobs that
  never started are dropped.
- **Inference**: synthetic, because Acme has no serving fleet. Demand follows
  a daily cosine between 30% and 100% of 384 GPUs (1-GPU replicas). A new
  replica needs 120 s to load its model before it serves. SLO attainment is
  1 minus unmet replica-seconds over demanded replica-seconds.
- **Sharing**: a 1-GPU eval may join a GPU already used by one other 1-GPU
  eval (MPS-style, 2 per GPU); the joiner runs 1.3x slower (the incumbent's
  slowdown is not modelled).
- **Batching**: when no GPU is free, short evals (at most 10 minutes) from the
  same user arriving within 60 s run back to back in one process that loads
  the model once; members after the first save 29.5% of their runtime (Acme's
  measured share of eval time spent loading and preprocessing). A member's
  queueing delay includes waiting for the members before it.
- **Preemption** happens only when the inference pool reclaims lent GPUs.
  Training loses the work since its last checkpoint (every 30 minutes) plus a
  300 s restart; eval and debug restart from scratch.
- **Estimates**: backfill and lease decisions use each job's actual runtime
  as its estimate, which is optimistic (real schedulers see user-supplied
  limits).

## 3. Policies

| policy | ordering | placement | inference pool | reclaim victims |
|---|---|---|---|---|
| fifo | strict FIFO (head-of-line blocking) | first fit | dedicated, replicas kept warm | n/a |
| binpack | best-effort FIFO (a blocked job does not block later ones) | best fit | dedicated, replicas kept warm | n/a |
| kueue | best-effort FIFO | least allocated (kube-scheduler default) | lends idle GPUs to any job | newest first |
| mixsched | evals first, then by submit time | fragmentation-aware | lends idle GPUs only to evals that will finish before forecast demand returns | least work lost since checkpoint |

`kueue` is a model of Kueue's BestEffortFIFO queueing with borrowing and
reclaim plus default kube-scheduler scoring, not Kueue itself.

MixSched's mechanisms, each switchable for ablation:

- **eval-first ordering**: evaluations go ahead of other pending work;
- **fragmentation-aware placement** (FGD-style): a small job goes where it
  increases stranded capacity least, weighted by the job-size mix, so whole
  nodes stay free for gangs;
- **eval sharing** (MPS, 2 per GPU) and **adaptive batching** (one model
  load per batch, only when evals would queue anyway);
- **lend to evals only**, and only for **leases** shorter than the time
  until forecast inference demand needs the GPUs back;
- **checkpoint-aware victims** when reclaiming;
- **predictive reclaim**: size the pool for the demand one model-load time
  ahead, so replicas are warm when demand arrives;
- **EASY reservation** for the oldest blocked training gang is implemented
  but off by default: it lengthened training queues on both Seren windows
  tested (the ablation `mixsched+resv`).

### On Kubernetes: a scheduler-framework plugin

`k8s/scheduler/plugin` ports the mechanisms that belong in kube-scheduler and
is compiled into kube-scheduler v1.34.2 (`k8s/scheduler/main.go` registers it
with `app.NewSchedulerCommand(app.WithPlugin(...))`):

- **QueueSort**: evaluation pods first, then priority, then arrival
  (eval-first ordering). Quota, borrowing and reclaim stay with Kueue.
- **Score**: minus the increase in stranded GPUs a placement causes, weighted
  by the GPU demand of each job size (the simulator's `fragDelta`). Breaking
  a fully free node costs the most; filling a node exactly costs nothing.
- **PreFilter + Permit**: gang scheduling. Pods with the same
  `mixsched.io/gang` label wait at Permit until `mixsched.io/gang-size` of
  them hold reservations, then all bind; on a 30 s timeout every waiting
  member is rejected and its reservation released. PreFilter turns a gang away
  early when the cluster does not have enough free GPUs for its unplaced
  members, so partial gangs do not sit on GPUs.

Three scheduler profiles are compared (`k8s/config/`): the default
kube-scheduler (LeastAllocated spreading), GPU bin packing (NodeResourcesFit
MostAllocated on `nvidia.com/gpu`), and MixSched (the plugin's QueueSort,
PreFilter, Score and Permit; spreading scores off).

**KWOK benchmark** (`bench/k8s/`): etcd, kube-apiserver, KWOK and the
scheduler on one host, all built from source; N fake nodes with 8 GPUs each;
the driver fills the cluster to 90% of its GPUs with one job mix and records
pods bound per second, kube-scheduler's own `scheduling_attempt_duration`
histogram, and placement quality from the bound pods. There is no
controller-manager, so the driver does the two things it would have done:
create the namespace's `default` service account (otherwise the
ServiceAccount admission plugin rejects every pod) and remove the
`node.kubernetes.io/not-ready` taint the apiserver adds to new nodes (the
node lifecycle controller removes it once KWOK reports the node Ready).
Bindings are observed through a pod watch. During the creation burst the
apiserver's watch cache closes that watch several times (the client falls
behind on a shared 2-vCPU box); the driver resumes from the last
resourceVersion it saw, so a reconnect delivers only the missed events. An
earlier version re-watched from an empty resourceVersion, which made every
reconnect replay all existing pods (14,407 ADDED events for 2,814 pods in a
smoke run) and added apiserver load during the measurement. The driver's
client uses protobuf, like kube-scheduler.

**Fidelity check**: the simulator's placement (`mixsched fill`, same job
sequence, spread / binpack / frag) against the real scheduler's, on
fragmentation and on how many whole 8-GPU jobs still fit afterwards.

### Fractional GPUs: FGD on the Alibaba GPU trace v2023

Acme has whole-GPU jobs only. `mixsched fgd` replays FGD's workload-inflation
experiment (Weng et al., ATC '23) on the Alibaba trace (`data/alibaba`, from
github.com/alibaba/clusterdata): tasks sampled with replacement, placed one
at a time on the 1,213-node cluster with no departures, until arrived GPU
demand reaches 130% of capacity; tasks that do not fit are dropped. Policies:
random, dot-product, GPU clustering, GPU packing, best-fit, MixSched's
whole-GPU fragmentation score, and FGD re-implemented from the paper (the
fragment of a node for each task class, weighted by the class's popularity;
each task goes to the node and GPU that increase it least).

## 4. Metrics

- **GPU allocation**: allocated GPU-time over capacity (warm idle inference
  replicas count as allocated, which flatters dedicated pools);
- **effective utilization**: training/eval/debug allocation plus inference
  replicas actually serving demand;
- **node fragmentation**: time-averaged share of free GPUs stranded on
  partially used nodes (unusable by an 8-GPU job);
- queueing delay p50/p99 per class, mean training JCT slowdown
  ((finish - submit) / runtime), inference SLO attainment, GPU-hours lost to
  preemption.

## 5. Validity threats

- **Synthetic inference load** sits on its own pool; the interaction with
  training depends entirely on its shape.
- **Perfect runtime estimates** for backfill and leases.
- **No topology beyond racks**, no network contention, no failures (Acme
  reports many failed jobs; they are replayed with their real runtimes).
- **The scheduler pass is simplified**: placement looks at at most 3,000
  pending jobs per pass.
- **Results come from two Seren windows and one Kalos window** at three
  cluster sizes. Mechanism choices (reservation off) were made after seeing
  both Seren windows.
- **The Kubernetes port covers the scheduler, not Kueue.** Ordering,
  fragmentation scoring and gangs run in kube-scheduler; lending inference
  GPUs, leases and reclaim are only in the simulator. No Kueue
  ClusterQueues were configured.
- **kube-scheduler samples nodes.** With `percentageOfNodesToScore` unset it scores 50% minus 1% per 125 nodes of the feasible nodes (42% at 1,000, 10% at 5,000), from a rotating start. The simulator scores every node, so placement fidelity held at 1,000 nodes and broke at 5,000 (500 whole nodes in the simulator, 0 to 13 in kube-scheduler). The 5,000-node KWOK rows are one seed.
- **KWOK measures the control plane, not GPUs.** Pods bind but never run, so
  the KWOK numbers are scheduler throughput, decision latency and placement,
  on a 2-vCPU box shared with etcd, the apiserver and KWOK.

## 6. Interview defence notes

- *Why Kueue plus a plugin rather than a custom scheduler?* Kueue already
  owns quota, admission, borrowing and preemption across ClusterQueues; the
  scheduler framework owns placement. MixSched's ideas split along the same
  line: ordering, lending and reclaim are admission decisions; fragmentation-
  aware placement is a Score plugin.
- *How does gang admission interact with preemption?* A gang is admitted only
  when all its GPUs fit at once (no partial starts). Reclaim never preempts
  training in MixSched's default because only evals borrow inference GPUs; in
  the lend-to-all ablation, reclaim kills borrowers including gangs and the
  lost GPU-hours show the cost.
- *How do you define fragmentation?* Free GPUs on partially used nodes that an
  8-GPU job cannot use, over all free GPUs (time-averaged). The placement
  score is the expected stranded capacity weighted by the job-size mix.
- *What if a reclaim misses its deadline?* Replicas start late, the SLO
  counts the unmet replica-seconds, and the 120 s model load is why
  predictive reclaim exists; leases keep evals from being on those GPUs in
  the first place.
- *Why did bin packing stop packing at 5,000 nodes?* kube-scheduler stops looking for feasible nodes after a percentage of the cluster (adaptive: 10% at 5,000 nodes) and scores only those, starting each cycle where the last one stopped. Every pod is packed within a moving 500-node window, so across the cluster it looks like spreading. Setting `percentageOfNodesToScore: 100` trades scheduling throughput for placement quality; that run was not done.
- *Why is MixSched's attempt latency higher than the default's?* `scheduling_attempt_duration_seconds` is recorded after the binding cycle, so a gang member's wait at Permit for the rest of its gang is included. The default scheduler has no gangs (its gang pods simply stay unschedulable and are not in the successful-attempt histogram).
