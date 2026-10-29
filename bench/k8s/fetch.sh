#!/usr/bin/env bash
# etcd (v3.6.4) and KWOK (v0.7.0), built from their tagged sources. Their
# modules are fetched normally; where a vanity import path is unreachable
# (offline builds), set GO=/path/to/a wrapper that resolves it.
set -euo pipefail
cd "$(dirname "$0")/../.."
OUT=${OUT:-$PWD/bin/k8s}
SRC=${SRC:-${HOME}/src}
GO=${GO:-go}
mkdir -p "$OUT" "$SRC"
export GOTOOLCHAIN=local CGO_ENABLED=0
if [[ ! -x $OUT/etcd ]]; then
  [[ -d $SRC/etcd ]] || git clone -q --depth 1 --branch v3.6.4 https://github.com/etcd-io/etcd "$SRC/etcd"
  (cd "$SRC/etcd/server" && $GO build -trimpath -ldflags "-s -w" -o "$OUT/etcd" .)
fi
if [[ ! -x $OUT/kwok ]]; then
  [[ -d $SRC/kwok ]] || git clone -q --depth 1 --branch v0.7.0 https://github.com/kubernetes-sigs/kwok "$SRC/kwok"
  (cd "$SRC/kwok" && $GO build -trimpath -ldflags "-s -w" -o "$OUT/kwok" ./cmd/kwok)
fi
"$OUT/etcd" --version | head -1
"$OUT/kwok" --version
