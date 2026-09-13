#!/bin/bash
set -euo pipefail
: "${ABI_CAPSULE_PATH:?}" "${ABI_SANDBOX_ROOT:?}"
case "$ABI_CAPSULE_PATH" in /tmp/abi-r28-extraction-*) ;; *) exit 81 ;; esac
case "$ABI_SANDBOX_ROOT" in /tmp/abi-r28-root-*) ;; *) exit 82 ;; esac
test -f "$ABI_CAPSULE_PATH/isolated_worker.py"; test ! -e "$ABI_SANDBOX_ROOT"
mkdir -m 700 "$ABI_SANDBOX_ROOT"; mount -t tmpfs -o size=64m,mode=700 tmpfs "$ABI_SANDBOX_ROOT"
mkdir -p "$ABI_SANDBOX_ROOT"/{usr,etc,dev,proc,tmp,run,mnt,capsule,oldroot}; mount -o bind,ro /usr "$ABI_SANDBOX_ROOT/usr"; mount -t tmpfs -o size=4m,mode=755 tmpfs "$ABI_SANDBOX_ROOT/etc"
for ABI_ETC_FILE in group hostname hosts ld.so.cache localtime nsswitch.conf passwd; do test ! -f "/etc/$ABI_ETC_FILE" || cp -L --preserve=mode,timestamps "/etc/$ABI_ETC_FILE" "$ABI_SANDBOX_ROOT/etc/$ABI_ETC_FILE"; done
mount -o remount,ro "$ABI_SANDBOX_ROOT/etc"; mount -t tmpfs -o size=16m,mode=755 tmpfs "$ABI_SANDBOX_ROOT/dev"
for ABI_DEVICE_NAME in null zero random urandom; do touch "$ABI_SANDBOX_ROOT/dev/$ABI_DEVICE_NAME"; mount -o bind,ro "/dev/$ABI_DEVICE_NAME" "$ABI_SANDBOX_ROOT/dev/$ABI_DEVICE_NAME"; done
mkdir -m 1777 "$ABI_SANDBOX_ROOT/dev/shm"; mount --bind "$ABI_CAPSULE_PATH" "$ABI_SANDBOX_ROOT/capsule"; ln -s usr/bin "$ABI_SANDBOX_ROOT/bin"; ln -s usr/sbin "$ABI_SANDBOX_ROOT/sbin"; ln -s usr/lib "$ABI_SANDBOX_ROOT/lib"; test ! -d /usr/lib64 || ln -s usr/lib64 "$ABI_SANDBOX_ROOT/lib64"
cd "$ABI_SANDBOX_ROOT"; pivot_root . oldroot
exec chroot . /bin/bash -c '
set -euo pipefail
mount -t proc -o nosuid,nodev,noexec proc /proc
umount -l /oldroot; rmdir /oldroot
export PYTHONDONTWRITEBYTECODE=1 ABI_R28_ISOLATED=1
cd /capsule
exec /usr/bin/python3 -B isolated_worker.py --capsule /capsule
'
