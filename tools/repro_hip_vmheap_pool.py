"""Reproducer for a HIP stream-ordered memory pool bug (ROCm 7.13 nightlies, gfx1151).

After a large pool allocation is freed, a host copy of ~20 KB or more into or out of a new pool
allocation zero-fills a different live allocation from the same freed range. The classic pool
(DEBUG_HIP_MEM_POOL_VMHEAP=0) is not affected. Uses only the HIP runtime via ctypes.

Usage (after `source scripts/env.sh`, which sets the workaround):
    DEBUG_HIP_MEM_POOL_VMHEAP=1 python tools/repro_hip_vmheap_pool.py 64e6 8192 pageable  # CORRUPT
    DEBUG_HIP_MEM_POOL_VMHEAP=0 python tools/repro_hip_vmheap_pool.py 64e6 8192 pageable  # OK

Arguments: <bytes of the freed block> <float32 count of the copy> <pageable|pinned|memset|d2d|d2h> [free 0/1]
"""

import ctypes
import glob
import os
import sys

root = os.popen("rocm-sdk path --root").read().strip()
lib = ctypes.CDLL(glob.glob(root + "/lib/libamdhip64.so*")[0])
bigsz = int(float(sys.argv[1]))
zn = int(sys.argv[2])
op = sys.argv[3]
dofree = sys.argv[4] == "1" if len(sys.argv) > 4 else True

s = ctypes.c_void_p()
lib.hipStreamCreateWithFlags(ctypes.byref(s), 1)
pool = ctypes.c_void_p()
lib.hipDeviceGetDefaultMemPool(ctypes.byref(pool), 0)
# hipMemPoolAttrReleaseThreshold: keep freed memory in the pool (as Warp does)
thr = ctypes.c_uint64(int(os.environ.get("THR", 2**64 - 1)))
lib.hipMemPoolSetAttribute(pool, 4, ctypes.byref(thr))


def malloc_async(n):
    p = ctypes.c_void_p()
    r = lib.hipMallocAsync(ctypes.byref(p), ctypes.c_size_t(n), s)
    assert r == 0, r
    return p


# Allocate, fill and free a large block so later allocations reuse its range.
if bigsz:
    a = malloc_async(bigsz)
    lib.hipMemsetD8Async(a, 1, ctypes.c_size_t(bigsz), s)
    lib.hipStreamSynchronize(s)
    if dofree:
        lib.hipFreeAsync(a, s)
        lib.hipStreamSynchronize(s)

# Witness allocation holding 1..10.
w = malloc_async(40)
hw = (ctypes.c_float * 10)(*range(1, 11))
lib.hipMemcpyAsync(w, hw, ctypes.c_size_t(40), 1, s)
lib.hipStreamSynchronize(s)

# Unrelated allocation and copy; must not touch the witness.
z = malloc_async(zn * 4)
if op == "pageable":
    hz = (ctypes.c_float * zn)(*([1.0] * zn))
    r = lib.hipMemcpyAsync(z, hz, ctypes.c_size_t(zn * 4), 1, s)
elif op == "pinned":
    hp = ctypes.c_void_p()
    lib.hipHostMalloc(ctypes.byref(hp), ctypes.c_size_t(zn * 4), 0)
    r = lib.hipMemcpyAsync(z, hp, ctypes.c_size_t(zn * 4), 1, s)
elif op == "memset":
    r = lib.hipMemsetD8Async(z, 3, ctypes.c_size_t(zn * 4), s)
elif op == "d2d":
    src = malloc_async(zn * 4)
    r = lib.hipMemcpyAsync(z, src, ctypes.c_size_t(zn * 4), 3, s)
elif op == "d2h":
    hz = (ctypes.c_float * zn)()
    r = lib.hipMemcpyAsync(hz, z, ctypes.c_size_t(zn * 4), 2, s)
lib.hipStreamSynchronize(s)

out = (ctypes.c_float * 10)()
lib.hipMemcpyAsync(out, w, ctypes.c_size_t(40), 2, s)
lib.hipStreamSynchronize(s)
print(
    f"big={bigsz:.3g} free={dofree} zn={zn} op={op} rc={r} w={hex(w.value)}",
    "OK" if out[0] == 1.0 else f"CORRUPT {list(out)[:3]}",
)
