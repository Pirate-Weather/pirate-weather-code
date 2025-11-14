#!/usr/bin/env bash
set -euo pipefail

BASE_DIR="/mnt/nvme/data/ProdTest3"
READY_FILE="/mnt/nvme/data/ProdTest3/models_ready"

# Models to update
MODELS=('HRRR'
        'GFS'
        'NBM'
        'HRRR_6H'
        'RTMA_RU'
        'GEFS'
        'ECMWF'
        'NWS_Alerts'
        'WMO_Alerts'
        'SubH'
        'NBM_Fire'
        'ETOPO_DA_C')

cleanup() {
    # If the container/script exits, mark not-ready
    rm -f "$READY_FILE"
}
trap cleanup EXIT

# first_run = 1 ? no bwlimit
# first_run = 0 ? use --bwlimit on all future loops
first_run=1

while true; do
  echo "Starting update loop at $(date -Iseconds)"
  loop_ok=1

  for MODEL in "${MODELS[@]}"; do
    echo "=== Updating $MODEL ==="

    REMOTE="s3:piratezarr2/ForecastTar_v2/v30/${MODEL}.zarr.zip"
    ZIP_LOCAL="${BASE_DIR}/${MODEL}.zarr.zip"
    STATE_FILE="${BASE_DIR}/${MODEL}.zarr.state"

    # 1) Query remote info via rclone lsl (size, date, time, path)
    remote_info="$(rclone lsl "$REMOTE" 2>/dev/null || true)"

    if [ -z "$remote_info" ]; then
      echo "Remote file not found or rclone lsl failed for $MODEL"
      loop_ok=0
      continue
    fi

    # 2) Load last-seen remote info from state file (if any)
    last_info="$(cat "$STATE_FILE" 2>/dev/null || echo "")"

    if [ "$remote_info" = "$last_info" ]; then
      echo "No change detected on remote for $MODEL. Skipping download/unzip."
      continue
    fi

    echo "Remote changed for $MODEL (or first run). Downloading ZIP…"

    # 3) Download ZIP only if remote changed; apply bwlimit after first run
    if [ "$first_run" -eq 1 ]; then
      if ! rclone copy "$REMOTE" "$BASE_DIR" \
           --checksum --update --progress; then
        echo "rclone copy failed for $MODEL"
        loop_ok=0
        continue
      fi
    else
      if ! rclone copy "$REMOTE" "$BASE_DIR" \
           --checksum --update --progress --bwlimit 120M; then
        echo "rclone copy failed for $MODEL"
        loop_ok=0
        continue
      fi
    fi

    if [ ! -f "$ZIP_LOCAL" ]; then
      echo "ZIP not found after rclone copy for $MODEL: $ZIP_LOCAL"
      loop_ok=0
      continue
    fi

    echo "New ZIP for $MODEL downloaded. Extracting…"

    # 4) Create versioned directory and temp extraction dir
    timestamp="$(date -u +"%Y%m%dT%H%M%SZ")"
    version_dir="${BASE_DIR}/${MODEL}_${timestamp}.zarr"
    tmpdir="$(mktemp -d "${version_dir}.tmp.XXXXXX")" || {
      echo "Failed to create temp dir for $MODEL"
      loop_ok=0
      continue
    }

    # 5) Unzip with low priority so it doesn't hammer the disk, only for subsequent runs
    if [ "$first_run" -eq 1 ]; then
      # first run: full speed
      if ! unzip -q "$ZIP_LOCAL" -d "$tmpdir"; then
        echo "Unzip failed for $MODEL"
        rm -rf "$tmpdir"
        loop_ok=0
        continue
      fi
    else
      # later runs: low priority I/O
      if ! ionice -c3 nice -n 19 unzip -q "$ZIP_LOCAL" -d "$tmpdir"; then
        echo "Unzip failed for $MODEL"
        rm -rf "$tmpdir"
        loop_ok=0
        continue
      fi
    fi

    # 6) Move temp dir to final versioned dir name
    mv "$tmpdir" "$version_dir"
    echo "Created versioned dir: $version_dir"

    # 7) Atomically repoint MODEL.zarr symlink ? MODEL_<timestamp>.zarr (relative)
    (
      cd "$BASE_DIR"
      ln -sfn "$(basename "$version_dir")" "${MODEL}.zarr"
    )
    echo "Updated symlink: ${BASE_DIR}/${MODEL}.zarr ? $(basename "$version_dir")"

    # 8) Keep ONLY the newest version dir per model, delete older ones
    (
      cd "$BASE_DIR"
      # newest ? oldest, drop everything after the first
      old_versions="$(ls -dt "${MODEL}_"*.zarr 2>/dev/null | tail -n +2 || true)"
      if [ -n "$old_versions" ]; then
        echo "Pruning old versions for $MODEL:"
        echo "$old_versions"
        rm -rf $old_versions
      fi
    )

    # 9) Update state file with the latest remote info
    echo "$remote_info" > "$STATE_FILE"

    # 10) Remove the local ZIP to save disk space
    rm -f "$ZIP_LOCAL"
    echo "Removed ZIP: $ZIP_LOCAL"

  done

  # 11) Mark ready / not-ready based on loop result
  if [ "$loop_ok" -eq 1 ]; then
    echo "Loop completed successfully, marking ready."
    touch "$READY_FILE"
  else
    echo "Loop had failures, clearing ready flag."
    rm -f "$READY_FILE"
  fi

  # After first loop, future runs use bwlimit on rclone
  first_run=0

  echo "Update loop finished. Sleeping 300s..."
  sleep 300
done
