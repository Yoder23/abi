#!/bin/bash
set -euo pipefail

: "${ABI_CAPSULE_PATH:?ABI_CAPSULE_PATH is required}"
: "${ABI_SANDBOX_ROOT:?ABI_SANDBOX_ROOT is required}"
: "${ABI_HOST_KEY:?ABI_HOST_KEY is required}"
: "${ABI_DEVICE:?ABI_DEVICE is required}"
: "${ABI_RUNTIME_SITE_PATH:?ABI_RUNTIME_SITE_PATH is required}"

case "$ABI_CAPSULE_PATH" in
  /tmp/abi-certification-*) ;;
  *) echo "unsafe capsule path" >&2; exit 81 ;;
esac
case "$ABI_SANDBOX_ROOT" in
  /tmp/abi-certification-root-*) ;;
  *) echo "unsafe sandbox root" >&2; exit 82 ;;
esac
if test ! -d "$ABI_CAPSULE_PATH/abi_release"; then
  echo "capsule is unavailable" >&2
  exit 83
fi
if test -e "$ABI_SANDBOX_ROOT"; then
  echo "sandbox root already exists" >&2
  exit 84
fi

if test ! -d "$ABI_RUNTIME_SITE_PATH"; then
  echo "Python user-site runtime is unavailable" >&2
  exit 85
fi

mkdir -m 700 "$ABI_SANDBOX_ROOT"
mount -t tmpfs -o size=256m,mode=700 tmpfs "$ABI_SANDBOX_ROOT"
mkdir -p \
  "$ABI_SANDBOX_ROOT/usr" \
  "$ABI_SANDBOX_ROOT/etc" \
  "$ABI_SANDBOX_ROOT/dev" \
  "$ABI_SANDBOX_ROOT/proc" \
  "$ABI_SANDBOX_ROOT/tmp" \
  "$ABI_SANDBOX_ROOT/run" \
  "$ABI_SANDBOX_ROOT/mnt" \
  "$ABI_SANDBOX_ROOT/capsule" \
  "$ABI_SANDBOX_ROOT/oldroot" \
  "$ABI_SANDBOX_ROOT$ABI_RUNTIME_SITE_PATH"

# Runtime libraries are the only non-capsule file trees admitted.  Ordinary
# bind mounts (not rbind) deliberately exclude nested WSL/Windows mounts.
mount -o bind,ro /usr "$ABI_SANDBOX_ROOT/usr"
mount -o bind,ro /etc "$ABI_SANDBOX_ROOT/etc"
mount -o bind,ro "$ABI_RUNTIME_SITE_PATH" "$ABI_SANDBOX_ROOT$ABI_RUNTIME_SITE_PATH"

# Construct a minimal device tree instead of exposing the host /dev tree.
mount -t tmpfs -o size=16m,mode=755 tmpfs "$ABI_SANDBOX_ROOT/dev"
for ABI_DEVICE_NAME in null zero random urandom; do
  touch "$ABI_SANDBOX_ROOT/dev/$ABI_DEVICE_NAME"
  mount -o bind,ro "/dev/$ABI_DEVICE_NAME" "$ABI_SANDBOX_ROOT/dev/$ABI_DEVICE_NAME"
done
mkdir -m 1777 "$ABI_SANDBOX_ROOT/dev/shm"

# The exact allowlisted capsule is the only writable campaign mount.
mount --bind "$ABI_CAPSULE_PATH" "$ABI_SANDBOX_ROOT/capsule"
ln -s usr/bin "$ABI_SANDBOX_ROOT/bin"
ln -s usr/sbin "$ABI_SANDBOX_ROOT/sbin"
ln -s usr/lib "$ABI_SANDBOX_ROOT/lib"
if test -d /usr/lib64; then
  ln -s usr/lib64 "$ABI_SANDBOX_ROOT/lib64"
fi

cd "$ABI_SANDBOX_ROOT"
pivot_root . oldroot
export ABI_RUNTIME_SITE_PATH ABI_HOST_KEY ABI_DEVICE
exec chroot . /bin/bash -c '
  set -euo pipefail
  mount -t proc -o nosuid,nodev,noexec proc /proc
  umount -l /oldroot
  rmdir /oldroot
  export PYTHONDONTWRITEBYTECODE=1
  export PYTHONPATH="$ABI_RUNTIME_SITE_PATH"
  export HF_HUB_OFFLINE=1
  export TRANSFORMERS_OFFLINE=1
  export HF_HOME=/tmp/abi-hf-runtime
  export ABI_CERTIFICATION_PIVOT_ROOT=1
  cd /capsule/abi_release
  exec /usr/bin/python3 -B -m abi_v2.isolated_certification worker \
    --capsule /capsule --host "$ABI_HOST_KEY" --device "$ABI_DEVICE"
'
