#!/usr/bin/env bash
# Builds kube-apiserver, kubectl, the MixSched scheduler (kube-scheduler with
# the plugin compiled in) and the benchmark driver from a Kubernetes checkout
# (v1.34.2: vendored modules, builds with Go 1.24). etcd and KWOK are release
# binaries (see fetch.sh).
set -euo pipefail
cd "$(dirname "$0")/../.."
K8S=${K8S_SRC:-${HOME}/src/kubernetes}
OUT=${OUT:-$PWD/bin/k8s}
[[ -d $K8S ]] || git clone --depth 1 --branch v1.34.2 https://github.com/kubernetes/kubernetes "$K8S"
mkdir -p "$OUT"
rm -rf "$K8S/cmd/mixsched-scheduler" "$K8S/cmd/mixsched-bench"
cp -r k8s/scheduler "$K8S/cmd/mixsched-scheduler"
cp -r k8s/bench "$K8S/cmd/mixsched-bench"
cd "$K8S"
export GOFLAGS=-mod=vendor GOTOOLCHAIN=local CGO_ENABLED=0
for t in kube-apiserver kubectl mixsched-scheduler mixsched-bench; do
  echo "building $t"
  go build -trimpath -ldflags "-s -w" -o "$OUT/$t" "./cmd/$t"
done
go test ./cmd/mixsched-scheduler/plugin/
ls -la "$OUT"
