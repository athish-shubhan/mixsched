package plugin

import (
	"testing"
	"time"

	v1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	fwk "k8s.io/kube-scheduler/framework"
	"k8s.io/kubernetes/pkg/scheduler/framework"
)

func TestFragDeltaPrefersHoles(t *testing.T) {
	// A 1-GPU job should go where it fills a hole, not break a free node.
	if !(FragDelta(1, 1) < FragDelta(3, 1) && FragDelta(3, 1) < FragDelta(8, 1)) {
		t.Fatalf("unexpected order: %v %v %v", FragDelta(1, 1), FragDelta(3, 1), FragDelta(8, 1))
	}
	// Filling a node exactly strands nothing.
	if d := FragDelta(4, 4); d > 0 {
		t.Fatalf("exact fit should not add stranded GPUs, got %v", d)
	}
}

func qpi(class string, ts time.Time) fwk.QueuedPodInfo {
	p := &v1.Pod{ObjectMeta: metav1.ObjectMeta{Name: class, Labels: map[string]string{LabelClass: class}}}
	pi, _ := framework.NewPodInfo(p)
	return &framework.QueuedPodInfo{PodInfo: pi, Timestamp: ts}
}

func TestEvalFirst(t *testing.T) {
	m := &MixSched{}
	now := time.Now()
	if !m.Less(qpi("eval", now.Add(time.Minute)), qpi("train", now)) {
		t.Fatal("a later eval should still sort before training")
	}
	if !m.Less(qpi("train", now), qpi("train", now.Add(time.Second))) {
		t.Fatal("same class: arrival order")
	}
}
