#!/usr/bin/env bash
# A single-host control plane for the KWOK benchmark: etcd, kube-apiserver
# (token auth, AlwaysAllow, API priority and fairness off so the benchmark
# client is not throttled) and the KWOK controller managing every node
# annotated kwok.x-k8s.io/node=fake. No kubelets, no controller-manager:
# the benchmark creates pods directly and only the scheduler binds them.
#
#   bench/k8s/cluster.sh up | down | scheduler <config.yaml> | stop-scheduler
set -euo pipefail
cd "$(dirname "$0")/../.."
BIN=${BIN:-$PWD/bin/k8s}
W=${W:-/var/lib/mixsched-k8s}
KC=$W/kubeconfig
mkdir -p "$W/logs"
# stop <name>: SIGTERM the process in $W/<name>.pid, wait up to 10 s for it to
# exit (its ports must be free before the next start), then SIGKILL.
stop() {
  local f=$W/$1.pid pid
  [[ -f $f ]] || return 0
  pid=$(cat "$f")
  kill "$pid" 2>/dev/null || true
  for _ in $(seq 100); do kill -0 "$pid" 2>/dev/null || break; sleep 0.1; done
  kill -9 "$pid" 2>/dev/null || true
  rm -f "$f"
}

case ${1:-up} in
up)
  "$0" down || true
  rm -rf "$W/etcd"; mkdir -p "$W/certs"
  [[ -f $W/sa.key ]] || { openssl genrsa -out "$W/sa.key" 2048 2>/dev/null; openssl rsa -in "$W/sa.key" -pubout -out "$W/sa.pub" 2>/dev/null; }
  echo "benchtoken,admin,admin,system:masters" > "$W/tokens.csv"
  cat > "$KC" <<KCF
apiVersion: v1
kind: Config
clusters: [{name: bench, cluster: {server: "https://127.0.0.1:6443", insecure-skip-tls-verify: true}}]
users: [{name: admin, user: {token: benchtoken}}]
contexts: [{name: bench, context: {cluster: bench, user: admin}}]
current-context: bench
KCF
  nohup "$BIN/etcd" --data-dir "$W/etcd" --listen-client-urls http://127.0.0.1:2379 --advertise-client-urls http://127.0.0.1:2379 \
    --listen-peer-urls http://127.0.0.1:2380 --quota-backend-bytes 8589934592 --unsafe-no-fsync > "$W/logs/etcd.log" 2>&1 &
  echo $! > "$W/etcd.pid"
  sleep 2
  nohup "$BIN/kube-apiserver" --etcd-servers http://127.0.0.1:2379 --secure-port 6443 --bind-address 127.0.0.1 --cert-dir "$W/certs" \
    --service-account-issuer https://kubernetes.default.svc --service-account-key-file "$W/sa.pub" --service-account-signing-key-file "$W/sa.key" \
    --token-auth-file "$W/tokens.csv" --authorization-mode AlwaysAllow --service-cluster-ip-range 10.96.0.0/16 \
    --max-requests-inflight 3000 --max-mutating-requests-inflight 2000 --enable-priority-and-fairness=false \
    --watch-cache-sizes pods#20000,nodes#10000 > "$W/logs/apiserver.log" 2>&1 &
  echo $! > "$W/apiserver.pid"
  for i in $(seq 60); do "$BIN/kubectl" --kubeconfig "$KC" get --raw /readyz >/dev/null 2>&1 && break; sleep 1; done
  nohup "$BIN/kwok" --kubeconfig "$KC" --manage-all-nodes=false --manage-nodes-with-annotation-selector=kwok.x-k8s.io/node=fake \
    --manage-nodes-with-label-selector= --disregard-status-with-annotation-selector=kwok.x-k8s.io/status=custom \
    --disregard-status-with-label-selector= --cidr=10.0.0.1/16 --node-ip=10.0.0.1 --node-lease-duration-seconds=40 > "$W/logs/kwok.log" 2>&1 &
  echo $! > "$W/kwok.pid"
  echo "control plane up; kubeconfig $KC"
  ;;
scheduler)
  "$0" stop-scheduler || true
  sed "s#KUBECONFIG#$KC#" "$2" > "$W/scheduler.yaml"
  nohup "$BIN/mixsched-scheduler" --config "$W/scheduler.yaml" --authentication-kubeconfig "$KC" --authorization-kubeconfig "$KC" \
    --authorization-always-allow-paths=/healthz,/readyz,/livez,/metrics --secure-port 10259 -v=1 > "$W/logs/scheduler-$(basename "$2" .yaml).log" 2>&1 &
  echo $! > "$W/scheduler.pid"
  for i in $(seq 60); do curl -sk https://127.0.0.1:10259/readyz 2>/dev/null | grep -q ok && break; sleep 1; done
  echo "scheduler $(basename "$2") up"
  ;;
stop-scheduler)
  stop scheduler
  ;;
down)
  for p in scheduler kwok apiserver etcd; do stop $p; done
  ;;
esac
