package trace

import (
	"os"
	"path/filepath"
	"testing"
)

const sample = `job_id,user,node_num,gpu_num,cpu_num,mem_per_pod_GB,shared_mem_per_pod,type,state,submit_time,start_time,end_time,fail_time,stop_time,duration,queue,gpu_time
a,u1,8,64,960,1000,100.0,Pretrain,COMPLETED,2023-05-17 11:00:58+00:00,2023-05-17 11:01:08+00:00,2023-05-17 12:01:08+00:00,,,3600,10.0,230400.0
b,u2,1,1,8,100,10.0,Eval,COMPLETED,2023-05-17 11:00:00+00:00,2023-05-17 11:00:05+00:00,2023-05-17 11:02:05+00:00,,,120,5.0,120.0
c,u3,1,0,8,100,10.0,Other,COMPLETED,2023-05-17 11:00:00+00:00,2023-05-17 11:00:05+00:00,2023-05-17 11:02:05+00:00,,,120,5.0,0.0
d,u4,1,1,8,100,10.0,Debug,CANCELLED,2023-05-17 11:05:00+00:00,,,,,0,,
`

func TestLoadAcme(t *testing.T) {
	p := filepath.Join(t.TempDir(), "t.csv")
	os.WriteFile(p, []byte(sample), 0o644)
	jobs, err := LoadAcme(p)
	if err != nil {
		t.Fatal(err)
	}
	// CPU-only and never-started jobs are dropped; times are relative and sorted
	if len(jobs) != 2 || jobs[0].Name != "b" || jobs[0].Submit != 0 || jobs[1].Submit != 58 {
		t.Fatalf("%+v", jobs)
	}
	if jobs[1].Class != Training || jobs[1].Nodes != 8 || jobs[1].QueuedOrig != 10 || jobs[0].Class != Eval {
		t.Fatalf("%+v", jobs)
	}
}
