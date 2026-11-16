#!/usr/bin/env bash
# Sequential job queue for phases 1-3 (phase 0 is run separately: python -m twinmask.phase0).
# Order = most important results first.
set -u
cd "$(dirname "$0")"
export PYTHONUTF8=1 TOKENIZERS_PARALLELISM=false
mkdir -p logs
STEPS=${STEPS:-3000}

run() {  # dataset variant seed [steps] [extra]
  local ds=$1 v=$2 s=$3 st=${4:-$STEPS} extra=${5:-}
  if [ -f "results/${TAG:-phase1}/${ds}_${v}_s${s}.json" ]; then echo "skip $ds $v $s"; return; fi
  echo "=== $(date +%T) $ds $v seed=$s steps=$st"
  python -m twinmask.tm --dataset "$ds" --variant "$v" --seed "$s" --steps "$st" --k 12 --max_pairs 1000 --tag "${TAG:-phase1}" $extra \
    > "logs/${TAG:-phase1}_${ds}_${v}_s${s}.log" 2>&1 || echo "FAILED $ds $v $s"
  grep -E "LP|NC|TEXT" "logs/${TAG:-phase1}_${ds}_${v}_s${s}.log" | tail -3
}

# ---------------- Phase 1: main model + modality ablations
for ds in cora citeseer pubmed; do
  run $ds joint 0
  run $ds edge_only 0
  run $ds text_only 0
  run $ds tied 0
done
# ---------------- Phase 2: topology ablations + extra seed + TSMI datasets
run cora static 0
run cora notopo 0
run pubmed static 0
run pubmed notopo 0
run cora joint 1
for ds in wikics instagram reddit; do run $ds joint 0; done
python -m twinmask.phase2 --datasets cora citeseer pubmed wikics instagram reddit > logs/phase2.log 2>&1 || echo "FAILED phase2"
# ---------------- Phase 3: scale to ogbn-arxiv (169k nodes)
TAG=phase3 run arxiv joint 0 8000 "--max_nc 20000"
TAG=phase3 run arxiv edge_only 0 8000 "--max_nc 20000"
echo "ALL DONE $(date +%T)"
