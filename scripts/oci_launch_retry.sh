#!/usr/bin/env bash
# Launch the Always Free A1 instance, retrying until Oracle has capacity.
#
# "Out of host capacity" for VM.Standard.A1.Flex is not a configuration error
# and not something waiting in the console fixes: capacity is released in short
# bursts and taken within seconds. The only reliable answer is to keep asking.
# This asks every INTERVAL seconds, across every availability domain in the
# region, and stops the moment one succeeds.
#
# Setup, once:
#   brew install oci-cli
#   oci setup config          # then upload ~/.oci/oci_api_key_public.pem in the
#                             # console: Profile -> My profile -> API keys
#   cp scripts/oci-launch.env.example scripts/oci-launch.env   # fill it in
#
# Then:
#   ./scripts/oci_launch_retry.sh
set -uo pipefail

cd "$(dirname "$0")/.."
CONFIG="scripts/oci-launch.env"
[[ -f "$CONFIG" ]] || { echo "missing $CONFIG — copy the .example and fill it in"; exit 1; }
# shellcheck disable=SC1090
source "$CONFIG"

: "${COMPARTMENT_ID:?set COMPARTMENT_ID in $CONFIG}"
: "${SUBNET_ID:?set SUBNET_ID in $CONFIG}"
SSH_KEY="${SSH_KEY:-$HOME/.ssh/id_ed25519.pub}"
NAME="${NAME:-atlas}"
OCPUS="${OCPUS:-4}"
MEMORY="${MEMORY:-24}"
BOOT_GB="${BOOT_GB:-100}"
INTERVAL="${INTERVAL:-60}"

[[ -f "$SSH_KEY" ]] || { echo "no public key at $SSH_KEY"; exit 1; }

echo "resolving the newest Ubuntu 24.04 image for A1…"
IMAGE_ID=$(oci compute image list \
  --compartment-id "$COMPARTMENT_ID" \
  --operating-system "Canonical Ubuntu" \
  --operating-system-version "24.04" \
  --shape VM.Standard.A1.Flex \
  --sort-by TIMECREATED --sort-order DESC --limit 1 \
  --query 'data[0].id' --raw-output) || exit 1
[[ -n "$IMAGE_ID" ]] || { echo "no matching image found"; exit 1; }

# Not `mapfile`: macOS ships bash 3.2, where it does not exist. And not
# `--raw-output` on an array either — that prints JSON, not lines.
ADS=()
while IFS= read -r ad; do
  [[ -n "$ad" ]] && ADS+=("$ad")
done < <(oci iam availability-domain list \
  --compartment-id "$COMPARTMENT_ID" --query 'data[].name' \
  | python3 -c 'import json,sys; print("\n".join(json.load(sys.stdin)))')
[[ ${#ADS[@]} -gt 0 ]] || { echo "could not list availability domains"; exit 1; }

echo "image:  $IMAGE_ID"
echo "shape:  VM.Standard.A1.Flex  ${OCPUS} OCPU / ${MEMORY} GB / ${BOOT_GB} GB boot"
echo "domains: ${ADS[*]}"
echo "asking every ${INTERVAL}s until one has capacity. Ctrl-C to stop."
echo

attempt=0
while true; do
  for ad in "${ADS[@]}"; do
    attempt=$((attempt + 1))
    printf '[%s] attempt %d — %s … ' "$(date +%H:%M:%S)" "$attempt" "$ad"

    out=$(oci compute instance launch \
      --availability-domain "$ad" \
      --compartment-id "$COMPARTMENT_ID" \
      --subnet-id "$SUBNET_ID" \
      --image-id "$IMAGE_ID" \
      --shape VM.Standard.A1.Flex \
      --shape-config "{\"ocpus\":$OCPUS,\"memoryInGBs\":$MEMORY}" \
      --boot-volume-size-in-gbs "$BOOT_GB" \
      --assign-public-ip true \
      --ssh-authorized-keys-file "$SSH_KEY" \
      --display-name "$NAME" \
      --wait-for-state RUNNING 2>&1)

    if [[ $? -eq 0 ]]; then
      echo "LAUNCHED"
      id=$(printf '%s' "$out" | grep -o '"id": "ocid1.instance[^"]*"' | head -1 | cut -d'"' -f4)
      echo
      echo "instance: $id"
      oci compute instance list-vnics --instance-id "$id" \
        --query 'data[0]."public-ip"' --raw-output 2>/dev/null \
        | sed 's/^/public IP: /'
      echo "next: ssh ubuntu@<that IP>"
      exit 0
    fi

    # Capacity is the expected failure and worth retrying. Anything else — a bad
    # OCID, a missing permission, a malformed shape — will never succeed, so
    # print it and stop rather than looping on a mistake for hours.
    if printf '%s' "$out" | grep -qiE 'out of (host )?capacity|outofcapacity'; then
      echo "no capacity"
    else
      echo "FAILED"
      echo
      printf '%s\n' "$out" | tail -20
      exit 1
    fi
  done
  sleep "$INTERVAL"
done
