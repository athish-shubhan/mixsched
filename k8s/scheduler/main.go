// Command mixsched-scheduler is kube-scheduler with the MixSched plugin
// compiled in (the scheduler-framework way to ship an out-of-tree plugin).
// bench/k8s/build.sh copies this directory into a Kubernetes checkout as
// cmd/mixsched-scheduler and builds it against the vendored modules.
package main

import (
	"os"

	"k8s.io/component-base/cli"
	_ "k8s.io/component-base/logs/json/register"
	_ "k8s.io/component-base/metrics/prometheus/clientgo"
	_ "k8s.io/component-base/metrics/prometheus/version"
	"k8s.io/kubernetes/cmd/kube-scheduler/app"

	"k8s.io/kubernetes/cmd/mixsched-scheduler/plugin"
)

func main() {
	cmd := app.NewSchedulerCommand(app.WithPlugin(plugin.Name, plugin.New))
	os.Exit(cli.Run(cmd))
}
