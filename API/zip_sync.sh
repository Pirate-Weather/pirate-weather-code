#!/bin/sh
set -eu

# Need bsdtar for streaming ZIP extraction, plus pv for throttling
apk add --no-cache libarchive-tools

BASE_DIR="${BASE_DIR:-/mnt/nvme/data/ProdTest3}"
REMOTE_BASE="${REMOTE_BASE:-s3:piratezarr2/ForecastTar_v2/v30}"
READY_FILE="${BASE_DIR}/models_ready"

# Remove ready flag on startup
[ -f "$READY_FILE" ] && rm -f "$READY_FILE"

# Models to update (space-separated)
MODELS="NBM HRRR GFS HRRR_6H RTMA_RU GEFS ECMWF NWS_Alerts WMO_Alerts SubH NBM_Fire ETOPO_DA_C"

# Cleanup on exit
cleanup() {
    rm -f "$READY_FILE"
}
trap cleanup EXIT

# Track whether it's the first full update loop
first_run=1

while true; do
  echo "Starting update loop at $(date -Iseconds)"
  loop_ok=1

  for MODEL in $MODELS; do
    echo "=== Updating $MODEL ==="

    REMOTE="${REMOTE_BASE}/${MODEL}.zarr.zip"
    STATE_FILE="${BASE_DIR}/${MODEL}.zarr.state"

    # Get remote file info (size, date, path)
    if ! remote_info=$(rclone lsl "$REMOTE" 2>/dev/null); then
      echo "Remote file not found or inaccessible for $MODEL"
      loop_ok=0
      continue
    fi

    # Load last known info
    last_info=""
    [ -f "$STATE_FILE" ] && last_info=$(cat "$STATE_FILE")

    # Skip if unchanged
    if [ "$remote_info" = "$last_info" ]; then
      echo "No change detected for $MODEL"
      continue
    fi

    echo "Remote changed for $MODEL — streaming and extracting..."

    timestamp=$(date -u +"%Y%m%dT%H%M%SZ")
    version_dir="${BASE_DIR}/${MODEL}_${timestamp}.zarr"
    tmpdir=$(mktemp -d "${version_dir}.tmp.XXXXXX") || {
      echo "Failed to create temp dir for $MODEL"
      loop_ok=0
      continue
    }

    # Stream and extract directly from S3
    if [ "$first_run" -eq 1 ]; then
      echo "(First run: unlimited speed)"
      if ! rclone cat "$REMOTE" | bsdtar -xf - -C "$tmpdir"; then
        echo "Extraction failed for $MODEL (first run)"
        rm -rf "$tmpdir"
        loop_ok=0
        continue
      fi
    else
      echo "(Subsequent run: throttled to 100 MB/s)"
      if ! rclone --bwlimit 100M cat "$REMOTE" | bsdtar -xf - -C "$tmpdir"; then
        echo "Extraction failed for $MODEL (throttled run)"
        rm -rf "$tmpdir"
        loop_ok=0
        continue
      fi
    fi

    mv "$tmpdir" "$version_dir"
    echo "Created versioned dir: $version_dir"

    # Update symlink atomically
    (
      cd "$BASE_DIR" || exit 1
      ln -sfn "$(basename "$version_dir")" "${MODEL}.zarr"
    )
    echo "Updated symlink for $MODEL → $(basename "$version_dir")"

    # Prune old versions (keep only the most recent)
    (
      cd "$BASE_DIR" || exit 1
      old_versions=$(ls -dt "${MODEL}_"[0-9][0-9][0-9][0-9][0-9][0-9][0-9][0-9]T[0-9][0-9][0-9][0-9][0-9][0-9]Z.zarr 2>/dev/null \
      | sed '1d' || true)
      if [ -n "$old_versions" ]; then
        echo "Pruning old versions for $MODEL:"
        printf '%s\n' "$old_versions"
printf '%s\n' "$old_versions" | xargs -r rm -rf
      fi
    )

    # Save current state
    echo "$remote_info" > "$STATE_FILE"
  done

  # Mark ready or not based on success
  if [ "$loop_ok" -eq 1 ]; then
    echo "Loop completed successfully — marking ready."
    touch "$READY_FILE"
  else
    echo "Loop had failures — clearing ready flag."
    rm -f "$READY_FILE"
  fi

  first_run=0
  echo "Update loop finished. Sleeping 300s..."
  sleep 300
done
