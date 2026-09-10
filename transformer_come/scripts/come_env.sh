# shellcheck shell=bash
# Shared environment and preflight for the COME-Transformer launch scripts.
# Sourced by transformer_come/<variant>/run_all.sh; not executable on its own.

PROJECT_ROOT=/home/nas3/biod/wangkangyi/transformer-tta
PYTHON_BIN=/home/nas3/biod/wangkangyi/envs/lbi/bin/python
CHECKPOINT_ROOT=/home/nas3/biod/wangkangyi/checkpoints/source_models
RESULTS_ROOT=/home/nas3/biod/wangkangyi/results

export HF_HOME=/home/nas3/biod/wangkangyi/hf-cache
export TORCH_HOME=/home/nas3/biod/wangkangyi/hf-cache/torch
export PIP_CACHE_DIR=/home/nas3/biod/wangkangyi/pip-cache
export CONDA_PKGS_DIRS=/home/nas3/biod/wangkangyi/conda-pkgs
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export OMP_NUM_THREADS=4

# The expensive Saliency state-hash audit must be off for formal runs; the
# audited correctness pass sets this explicitly instead.
export COME_SELECTION_STATE_AUDIT_BATCHES=${COME_SELECTION_STATE_AUDIT_BATCHES:-0}

come_preflight() {
  local variant="$1" devices="$2" config="$3" output_root="$4" dry_run="$5" rhos="${6:-}"

  [[ -x "$PYTHON_BIN" ]] || { echo "Missing interpreter: $PYTHON_BIN" >&2; exit 1; }
  [[ -f "$config" ]] || { echo "Missing config: $config" >&2; exit 1; }
  local checkpoint
  for checkpoint in office31/amazon.pth office31/dslr.pth office31/webcam.pth \
                    visda-c/train.pth; do
    [[ -f "$CHECKPOINT_ROOT/$checkpoint" ]] || {
      echo "Missing source W0: $CHECKPOINT_ROOT/$checkpoint" >&2; exit 1; }
    [[ -f "$CHECKPOINT_ROOT/${checkpoint%.pth}.manifest.json" ]] || {
      echo "Missing source manifest for $checkpoint" >&2; exit 1; }
  done
  mkdir -p "$output_root" "$TMPDIR"

  echo "=================================================================="
  echo "COME-Transformer launch: $variant"
  echo "  method            : come (SHOT-Transformer substrate + COME objective)"
  echo "  repo commit       : $(git -C "$PROJECT_ROOT" rev-parse --short HEAD 2>/dev/null || echo unknown)"
  echo "  dirty worktree    : $(if [[ -n "$(git -C "$PROJECT_ROOT" status --porcelain 2>/dev/null)" ]]; then echo yes; else echo no; fi)"
  echo "  config            : $config"
  echo "  output root       : $output_root"
  echo "  devices           : $devices"
  [[ -n "$rhos" ]] && echo "  rho               : $rhos"
  echo "  saliency audit    : COME_SELECTION_STATE_AUDIT_BATCHES=$COME_SELECTION_STATE_AUDIT_BATCHES"
  echo "  started           : $(date -Is)"
  echo "=================================================================="

  [[ "$dry_run" == "1" ]] && return 0
  [[ "$devices" == "cpu" ]] && return 0

  # Protocol section 18: formal runtime comparability needs exactly one
  # experiment process per GPU. nvidia-smi enumerates in PCI bus order, which
  # is the order the scheduler passes as CUDA_VISIBLE_DEVICES.
  local busy=() index pids
  IFS=',' read -r -a _indices <<< "$devices"
  for index in "${_indices[@]}"; do
    pids=$(nvidia-smi -i "$index" --query-compute-apps=pid --format=csv,noheader 2>/dev/null || true)
    [[ -n "${pids//[[:space:]]/}" ]] && busy+=("$index")
  done
  if ((${#busy[@]})); then
    echo "" >&2
    echo "GPUs already running compute processes: ${busy[*]}" >&2
    echo "Sharing a GPU makes the measured runtime formally non-comparable" >&2
    echo "(protocol section 18: one experiment process per GPU)." >&2
    echo "Either wait, restrict the *_GPUS variable to idle GPUs, or set" >&2
    echo "COME_ALLOW_BUSY_GPUS=1 to proceed with runtime_comparable=false." >&2
    [[ "${COME_ALLOW_BUSY_GPUS:-0}" == "1" ]] || exit 1
    echo "COME_ALLOW_BUSY_GPUS=1 set; continuing on shared GPUs." >&2
  fi
}
