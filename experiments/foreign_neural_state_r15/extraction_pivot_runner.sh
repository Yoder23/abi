#!/bin/bash
set -euo pipefail

: "${ABI_CAPSULE_PATH:?ABI_CAPSULE_PATH is required}"
: "${ABI_SANDBOX_ROOT:?ABI_SANDBOX_ROOT is required}"

case "$ABI_CAPSULE_PATH" in
  /tmp/abi-r15-extraction-*) ;;
  *) echo "unsafe R15 capsule path" >&2; exit 81 ;;
esac
case "$ABI_SANDBOX_ROOT" in
  /tmp/abi-r15-extraction-root-*) ;;
  *) echo "unsafe R15 sandbox root" >&2; exit 82 ;;
esac
test -f "$ABI_CAPSULE_PATH/isolated_worker.py"
test ! -e "$ABI_SANDBOX_ROOT"

mkdir -m 700 "$ABI_SANDBOX_ROOT"
mount -t tmpfs -o size=128m,mode=700 tmpfs "$ABI_SANDBOX_ROOT"
mkdir -p \
  "$ABI_SANDBOX_ROOT/usr" \
  "$ABI_SANDBOX_ROOT/etc" \
  "$ABI_SANDBOX_ROOT/dev" \
  "$ABI_SANDBOX_ROOT/proc" \
  "$ABI_SANDBOX_ROOT/tmp" \
  "$ABI_SANDBOX_ROOT/run" \
  "$ABI_SANDBOX_ROOT/mnt" \
  "$ABI_SANDBOX_ROOT/capsule" \
  "$ABI_SANDBOX_ROOT/oldroot"
mount -o bind,ro /usr "$ABI_SANDBOX_ROOT/usr"
mount -t tmpfs -o size=4m,mode=755 tmpfs "$ABI_SANDBOX_ROOT/etc"
for ABI_ETC_FILE in group hostname hosts ld.so.cache localtime nsswitch.conf passwd; do
  if test -f "/etc/$ABI_ETC_FILE"; then
    cp -L --preserve=mode,timestamps "/etc/$ABI_ETC_FILE" \
      "$ABI_SANDBOX_ROOT/etc/$ABI_ETC_FILE"
  fi
done
mount -o remount,ro "$ABI_SANDBOX_ROOT/etc"
mount -t tmpfs -o size=16m,mode=755 tmpfs "$ABI_SANDBOX_ROOT/dev"
for ABI_DEVICE_NAME in null zero random urandom; do
  touch "$ABI_SANDBOX_ROOT/dev/$ABI_DEVICE_NAME"
  mount -o bind,ro "/dev/$ABI_DEVICE_NAME" "$ABI_SANDBOX_ROOT/dev/$ABI_DEVICE_NAME"
done
mkdir -m 1777 "$ABI_SANDBOX_ROOT/dev/shm"
mount --bind "$ABI_CAPSULE_PATH" "$ABI_SANDBOX_ROOT/capsule"
ln -s usr/bin "$ABI_SANDBOX_ROOT/bin"
ln -s usr/sbin "$ABI_SANDBOX_ROOT/sbin"
ln -s usr/lib "$ABI_SANDBOX_ROOT/lib"
if test -d /usr/lib64; then
  ln -s usr/lib64 "$ABI_SANDBOX_ROOT/lib64"
fi

cd "$ABI_SANDBOX_ROOT"
pivot_root . oldroot
exec chroot . /bin/bash -c '
  set -euo pipefail
  mount -t proc -o nosuid,nodev,noexec proc /proc
  umount -l /oldroot
  rmdir /oldroot
  export PYTHONDONTWRITEBYTECODE=1
  export ABI_R15_ISOLATED=1
  cd /capsule
  exec /usr/bin/python3 -B isolated_worker.py --capsule /capsule
'
