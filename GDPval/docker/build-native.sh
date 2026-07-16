#!/usr/bin/env bash
set -euo pipefail

# Build only for the host architecture: QEMU/emulation corrupts tool timings.
case "$(uname -m)" in
  arm64|aarch64) arch=arm64 ;;
  x86_64|amd64) arch=amd64 ;;
  *) echo "Unsupported host architecture: $(uname -m)" >&2; exit 2 ;;
esac

image="${GDPVAL_DOCKER_IMAGE:-gdpval-timing:latest}"
cpus="${GDPVAL_BUILD_CPUS:-2}"
memory="${GDPVAL_BUILD_MEMORY:-8g}"
attempts="${GDPVAL_BUILD_ATTEMPTS:-3}"

if [[ ! "$cpus" =~ ^[1-9][0-9]*$ ]]; then
  echo "GDPVAL_BUILD_CPUS must be a positive integer" >&2
  exit 2
fi
if [[ ! "$attempts" =~ ^[1-9][0-9]*$ ]]; then
  echo "GDPVAL_BUILD_ATTEMPTS must be a positive integer" >&2
  exit 2
fi

if ! command -v docker >/dev/null 2>&1; then
  echo "Docker CLI is not installed. Install Docker Desktop or Colima first." >&2
  exit 1
fi

attempt=1
while true; do
  echo "Building ${image} for linux/${arch} (CPU limit ${cpus}, memory ${memory}), attempt ${attempt}/${attempts}"
  if docker build \
    --platform "linux/${arch}" \
    --cpu-quota "$((cpus * 100000))" \
    --memory "${memory}" \
    --tag "${image}" \
    --file docker/Dockerfile .; then
    break
  fi
  if (( attempt >= attempts )); then
    exit 1
  fi
  delay=$((15 * 2 ** (attempt - 1)))
  echo "Build failed; backing off for ${delay}s before retry." >&2
  sleep "${delay}"
  attempt=$((attempt + 1))
done

docker run --rm --platform "linux/${arch}" --network none --read-only \
  --tmpfs /tmp:rw,noexec,nosuid,size=256m --cap-drop ALL \
  --security-opt no-new-privileges "${image}" gdpval-image-healthcheck
