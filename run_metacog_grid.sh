#!/usr/bin/env bash
# Run the paper's grids: every dataset and encoder over seeds 42, 43 and 44.
#
#   RUN_ROOT=... MVTEC_AD_ROOT=... MVTEC_AD_2_ROOT=... VISA_ROOT=... ./run_metacog_grid.sh
#
# with, optionally,
#
#   DATASETS=ad2 SEEDS="42"                  a subset (the default seeds are 42 43 44)
#   DATASETS="ad2_vits ad2_vitb ad2_vitl"    the encoder sweep, for Figure 2
#   GPUS="0 1"                               the GPUs to use (default 0)
#   MERGE_ONLY=1                             only re-join the parts that already exist
#
# Each dataset's categories are split across the GPUs. Every worker writes its own part<i>.json and
# is retried up to MAX_ATTEMPTS times, resuming from the categories it already finished; the parts
# are then joined into results/<ds>/seed<N>/<ds>.json, which is what tables/run_all.sh reads.
set -uo pipefail
cd "$(dirname "$0")"

# Unquoted at each use: PYTHON may be a command with arguments.
PYTHON="${PYTHON:-uv run python}"
RUN_ROOT="${RUN_ROOT:?set RUN_ROOT to a directory for the results JSONs and logs}"
MAX_ATTEMPTS="${MAX_ATTEMPTS:-6}"
MERGE_ONLY="${MERGE_ONLY:-0}"
GPUS="${GPUS:-0}"
read -r -a GPU_ARR <<< "$GPUS"
NGPU=${#GPU_ARR[@]}
DATASETS="${DATASETS:-ad2 ad visa}"
SEEDS="${SEEDS:-42 43 44}"

declare -A CATEGORIES=(
  [ad2]="can fabric fruit_jelly rice sheet_metal vial wallplugs walnuts"
  [ad]="bottle cable capsule carpet grid hazelnut leather metal_nut pill screw tile toothbrush transistor wood zipper"
  [visa]="candle capsules cashew chewinggum fryum macaroni1 macaroni2 pcb1 pcb2 pcb3 pcb4 pipe_fryum"
)
for _b in vits vitb vitl; do CATEGORIES[ad2_$_b]="${CATEGORIES[ad2]}"; done

declare -A CONFIG=( [ad2]=configs/metacog_ad2.json [ad]=configs/metacog_ad.json [visa]=configs/metacog_visa.json
  [ad2_vits]=configs/metacog_ad2_vits.json [ad2_vitb]=configs/metacog_ad2_vitb.json
  [ad2_vitl]=configs/metacog_ad2_vitl.json )

data_root_for() {  # $1=dataset key
  case "$1" in
    ad)        echo "${MVTEC_AD_ROOT:?set MVTEC_AD_ROOT to the MVTec AD directory}" ;;
    ad2|ad2_*) echo "${MVTEC_AD_2_ROOT:?set MVTEC_AD_2_ROOT to the MVTec AD 2 directory}" ;;
    visa)      echo "${VISA_ROOT:?set VISA_ROOT to the VisA directory}" ;;
    *)         echo "unknown dataset $1" >&2; exit 1 ;;
  esac
}

shard() {  # $1=dataset key $2=shard index: every NGPU-th category, comma-separated
  local i=0 out=()
  for c in ${CATEGORIES[$1]}; do (( i++ % NGPU == $2 )) && out+=("$c"); done
  local IFS=,; echo "${out[*]}"
}

run_worker() {  # $1=config $2=categories $3=gpu $4=out $5=log $6=seed $7=data root
  for attempt in $(seq 1 "$MAX_ATTEMPTS"); do
    echo "=== $(date '+%F %T')  [$2] gpu $3 seed $6 attempt $attempt/$MAX_ATTEMPTS ===" >>"$5"
    CUDA_VISIBLE_DEVICES="$3" $PYTHON -u -m metacog.run \
        --config "$1" --data-root "$7" --category "$2" --seed "$6" --output "$4" >>"$5" 2>&1 \
      && { echo "=== $(date '+%F %T')  COMPLETE ===" >>"$5"; return 0; }
    echo "=== $(date '+%F %T')  exited $?; retrying in 30s ===" >>"$5"
    sleep 30
  done
  return 1
}

for ds in $DATASETS; do
  droot="$(data_root_for "$ds")"
  for seed in $SEEDS; do
    rdir="$RUN_ROOT/results/$ds/seed$seed"; ldir="$RUN_ROOT/logs/$ds/seed$seed"
    mkdir -p "$rdir" "$ldir"
    if [[ "$MERGE_ONLY" != "1" ]]; then
      echo "=== $(date '+%F %T')  $ds seed $seed on gpus $GPUS ==="
      for i in "${!GPU_ARR[@]}"; do
        cats="$(shard "$ds" "$i")"; [[ -z "$cats" ]] && continue
        run_worker "${CONFIG[$ds]}" "$cats" "${GPU_ARR[$i]}" "$rdir/part$i.json" "$ldir/part$i.log" "$seed" "$droot" &
      done
      wait
    fi
    parts=()
    for i in "${!GPU_ARR[@]}"; do [[ -f "$rdir/part$i.json" ]] && parts+=("$rdir/part$i.json"); done
    (( ${#parts[@]} )) || { echo "no results for $ds seed $seed" >&2; continue; }
    echo "=== $(date '+%F %T')  merging ${#parts[@]} parts -> $rdir/$ds.json ==="
    $PYTHON merge_shards.py --parts "${parts[@]}" --output "$rdir/$ds.json" 2>&1 | tee -a "$ldir/merge.log"
  done
done
echo "=== $(date '+%F %T')  all grids done ==="
