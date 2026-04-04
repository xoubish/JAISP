#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "$script_dir/.." && pwd)"

url="${1:-https://drive.usercontent.google.com/download?id=1d393O3bBMV9N1NgbcZcNccpD1svlCqBK&export=download&confirm=t}"
part_file="${2:-$repo_root/data/tiles_product.tar.gz.part}"
final_file="${3:-$repo_root/data/tiles_product.tar.gz}"
sleep_seconds="${SLEEP_SECONDS:-600}"

mkdir -p "$(dirname "$part_file")"

probe_headers="$(mktemp /tmp/tiles_product_probe_headers.XXXXXX)"
probe_body="$(mktemp /tmp/tiles_product_probe_body.XXXXXX)"

cleanup() {
  rm -f "$probe_headers" "$probe_body"
}

log() {
  printf '[%s] %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$*"
}

probe_download() {
  local resume_from="$1"

  : >"$probe_headers"
  : >"$probe_body"

  if ! curl -L -r "${resume_from}-${resume_from}" -D "$probe_headers" -o "$probe_body" "$url"; then
    log "Probe request failed."
    return 1
  fi

  if grep -qi '^content-type: application/octet-stream' "$probe_headers" &&
     grep -qi "^content-range: bytes ${resume_from}-${resume_from}/" "$probe_headers"; then
    return 0
  fi

  log "Google Drive is not serving resumable tarball bytes yet."
  return 1
}

resume_download() {
  local current_size=0

  if [ -f "$part_file" ]; then
    current_size="$(stat -c %s "$part_file")"
  fi

  log "Resuming from byte $current_size"
  curl --retry 5 --retry-delay 10 -L -C - -o "$part_file" "$url"
}

trap cleanup EXIT

if [ -f "$final_file" ]; then
  log "Final file already exists at $final_file"
  exit 0
fi

while true; do
  current_size=0
  if [ -f "$part_file" ]; then
    current_size="$(stat -c %s "$part_file")"
  fi

  if probe_download "$current_size" && resume_download; then
    mv "$part_file" "$final_file"
    log "Download complete: $final_file"
    exit 0
  fi

  log "Retrying in $sleep_seconds seconds."
  sleep "$sleep_seconds"
done
