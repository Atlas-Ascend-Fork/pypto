# Memory Sanitizer

<!-- md-trans-meta sourceCommit=50972549b8df8d831de411545a92c8acac5eac27 translatedAt=2026-11-15T02:12:02.772Z pushedAt=2026-11-18T06:03:01.350Z nl=2 -->

## Description

Memory access errors in the Kernel (GM out-of-bounds, tile out-of-bounds, on-chip address overlap, mutex pairing errors), phase usage errors (AccPhase/STPhase and K chain structure errors, where the K chain refers to the entire K-loop write between two stores on the same Acc tile), and sync pairing errors (sync_src/sync_dst) usually do not immediately cause exceptions on the Host side. They typically manifest as corrupted results, intermittent errors, or Kernel hangs, which are costly to troubleshoot manually. The memory sanitizer is enabled by setting sanitizer=True in @pypto_pro.language.jit(): the framework automatically instruments GM/tile access points, mutex call points, phase call points (AccPhase/STPhase), and sync call points (sync_src/sync_dst), and performs a unified check after each Kernel launch to give a conclusion.

## Constraints

- For debugging only: instrumentation and checking change the Kernel execution overhead. Before performance measurement, remove sanitizer=True from @pypto_pro.language.jit().
- Each Kernel launch allocates a log buffer of about 1 GB on GM. Repeated defects at the same location are aggregated into a single entry with a count.
- pypto_pro.language.Ptr carries only the address, not the size of the memory it points to. A view whose source is a pointer cannot determine the source size, so the check for a declared shape that exceeds the source is not performed.
- sync pairing detection does not yet support cross-core scenarios. Pairing checks are performed within a single execution core (region). A sync_src and sync_dst located on different cores are not paired, and each side is reported as a missing pairing.
- The check covers accesses and declarations within the Kernel body. All accesses (GM/tile/mutex/phase/sync) within SIMT thread blocks are not checked.

## Procedure

### Enabling Memory Sanitizer

Set `@pypto_pro.language.jit(sanitizer=True)` on the kernel function decorator:

```python
import pypto_pro.language as pl

@pl.jit(sanitizer=True, auto_mutex=True)
def kernel(x: pl.Tensor[[64, 64], pl.DT_FP16]):
    ...
```

sanitizer defaults to False. When it is disabled, the compilation and execution behavior of the Kernel remains completely unchanged. After it is enabled, detection runs automatically each time the Kernel is launched, and passes silently if no issue is found. When an issue is detected, the result is written in append mode to `pypto_sanitizer_report.txt` in the compilation output directory of the kernel (in the same directory as the raw record file `sanitizer_report.<launch_id>.bin` of that launch, with a path like `./build/<kernel_name>__<arch>/tk_none/`). Results from multiple launches of the same kernel are aggregated in the same file.

### Detection Report Description

In the report file, each result block begins with a kernel label line. The kernel name and launch identifier uniquely mark that launch, and the parenthesized part is the compilation output directory of the kernel used for that launch, such as `launch 9b3f1c2e` and `(./build/view_over_source_kernel__3510__d0/tk_none)`. When multiple compilation instances of the same kernel exist (different TilingKey, data type, or device), this directory is used to distinguish which instance the detection result comes from. This directory contains both the detection result file `pypto_sanitizer_report.txt` and the raw record file `sanitizer_report.<launch_id>.bin` of that launch (the launch identifier is a unique hexadecimal UUID for that launch, abbreviated in the examples in this document).

Each result block first gives the number of detections, then lists the detected items one by one. Taking GM out-of-bounds access as an example:

```text
===== kernel 'oob_kernel' launch 6e0b31f4 (./build/oob_kernel__3510__d0/tk_none) =====
PyPTO Sanitizer: 1 issue(s) detected
[1/1] GM_OUT_OF_BOUNDS: offsets [32, 0] + valid_shape [64, 64] exceed tensor shape (64, 64) — dim0: offset 32 + valid_shape 64 = 96 > 64 (over by 32) at demo.py:68
      Hint: The tile valid_shape plus the loop offsets must stay inside the tensor logical shape on every dimension
```

Each detected item consists of a type, a message, a source code location, and a Hint: the message describes the violation (including access parameters and the per-dimension out-of-bounds determination), the source code location is the source file and line number corresponding to the detected item, allowing direct location of the problematic statement, and the Hint is the fix suggestion for this type of issue. Duplicate defects at the same location (for example, out-of-bounds in every iteration of a loop) are aggregated into a single item, with the count marked as `same defect hit N times at this location`.

The detection types and their levels are as follows, where ERROR indicates a deterministic error that must be resolved, and WARNING indicates a suspicious issue that needs to be checked manually to determine whether it requires handling:

| Detection Type | Level | Detection Content |
|---|---|---|
| GM_OUT_OF_BOUNDS | ERROR | GM out-of-bounds access: offsets + valid_shape of the tile exceeds the tensor shape (up to 5 dimensions, including getval/setval scalar access and make_tensor declaration whose shape exceeds the source tensor). |
| TILE_OUT_OF_BOUNDS | ERROR | Tile out-of-bounds access: offset + valid_shape exceeds the declared dimension of the TileType (move/insert, set_validshape, tile scalar access). |
| TILE_OVERLAP | WARNING | The address ranges of two tiles overlap in the same memory space. |
| MUTEX_UNLOCK_BEFORE_LOCK / UNPAIRED_MUTEX_LOCK / MUTEX_DOUBLE_LOCK | ERROR | mutex_lock/mutex_unlock are not strictly alternated and paired by the same (region, mutex_id, pipe) (unlock without lock, lock without subsequent unlock, repeated lock). |
| PHASE_MISMATCH | ERROR | Phase (AccPhase/STPhase, hardware unit_flag synchronization) usage error: the dst and acc of matmul_acc/matmul_mx_acc are not the same Tile (in-place contract); reads and writes with and without phase are mixed on the same Acc tile (hardware handshake and software synchronization do not order each other); K chain structure error (the chain head mistakenly uses matmul_acc to accumulate onto a residual value, or matmul/matmul_bias appears in the middle of the chain and overwrites or clears the partial sum). |
| PHASE_DEADLOCK | ERROR | Phase state machine deadlock: a Final write is not consumed by a Final read (flag left pending), a read with STPhase waits for a flag that is never set, or a write waits for a flag that is never cleared. |
| SYNC_UNPAIRED_SRC / SYNC_UNPAIRED_DST | ERROR | sync_src/sync_dst are not paired by the same (region, set_pipe, wait_pipe, event_id) (a dangling src is a data race, and a dangling dst is a deadlock). |
| SYNC_EVENT_REUSE | ERROR | Before the sync_dst completion wait, the same event_id is set again by sync_src (the dangling flag is overwritten). |

### Disabling Memory Sanitizer

After the issue is fixed, remove sanitizer=True from @pypto_pro.language.jit() and restore the production configuration. Re-run regression tests covering all target Shapes, data types, TilingKeys, and block_dim to confirm that the issue is resolved and performance returns to normal.

## Detection Examples

### GM Out-of-Bounds Access

GM out-of-bounds access includes three types: block access out-of-bounds, scalar access out-of-bounds, and make_tensor declaration shape exceeding the source tensor.

#### Block Access Out-of-Bounds

Example description: The row offset 32 plus the tile valid window 64 exceeds the tensor row count 64.

```python
@pypto_pro.language.jit(sanitizer=True, auto_mutex=True)
def oob_kernel(x: pypto_pro.language.Tensor[[64, 64], pypto_pro.language.DT_FP16], z: pypto_pro.language.Tensor[[64, 64], pypto_pro.language.DT_FP16]):
    tt = pypto_pro.language.TileType(shape=[64, 64], dtype=pypto_pro.language.DT_FP16, target_memory=pypto_pro.language.MemorySpace.Vec)
    a_group = pypto_pro.language.make_tile_group(type=tt, addrs=0x0000, mutex_ids=[0])
    with pypto_pro.language.section_vector():
        a = a_group.current()
        # Row offset 32: 32 + 64 = 96 > 64
        pypto_pro.language.load(a, x, [32, 0])
        pypto_pro.language.store(z, a, [0, 0])
```

```text
===== kernel 'oob_kernel' launch 6e0b31f4 (./build/oob_kernel__3510__d0/tk_none) =====
PyPTO Sanitizer: 1 issue(s) detected
[1/1] GM_OUT_OF_BOUNDS: offsets [32, 0] + valid_shape [64, 64] exceed tensor shape (64, 64) — dim0: offset 32 + valid_shape 64 = 96 > 64 (over by 32) at demo.py:68
      Hint: The tile valid_shape plus the loop offsets must stay inside the tensor logical shape on every dimension
```

Field interpretation:

- offsets: The starting position of this access in each dimension of the tensor (for example, the third parameter of load).
- valid_shape: The tile valid window used by the access.

Conclusion: The load on line 68 of demo.py accesses a [64, 64] tensor with offset [32, 0]. In the row direction, 32 + 64 = 96 exceeds 64 by 32 elements. This can be located as a loop offset that is not capped by the tensor row count, and the loop boundary or offset calculation needs to be checked.

#### Scalar Access Out-of-Bounds

Example description: The linear offset 64 of setval exceeds the 64 elements of the tensor.

```python
@pypto_pro.language.jit(sanitizer=True, auto_mutex=True)
def gm_scalar_oob_kernel(out: pypto_pro.language.Tensor[[64], pypto_pro.language.DT_INT32]):
    with pypto_pro.language.section_vector():
        # Linear offset 64: exceeds 64 elements
        pypto_pro.language.setval(out, 64, 1)
```

```text
===== kernel 'gm_scalar_oob_kernel' launch 1c94f7a2 (./build/gm_scalar_oob_kernel__3510__d0/tk_none) =====
PyPTO Sanitizer: 1 issue(s) detected
[1/1] GM_OUT_OF_BOUNDS: scalar access at linear offset 64 exceeds 64 elements at demo.py:17
      Hint: getval/setval offsets must stay inside the tensor element count
```

Field interpretation:

- linear offset: element index on the tensor.

Conclusion: The setval at line 17 of demo.py writes index 64 (the last valid index is 63) on a tensor of 64 elements. This can be located as a scalar write position out-of-bounds, and the upper bound of the offset needs to be checked.

#### make_tensor Declaration Shape Exceeds Source Tensor

Example description: make_tensor declares a [200, 200] view on a [128, 128] source tensor. The shared storage causes the declaration to exceed the source capacity, which is detected at declaration time.

```python
# Source x is [128, 128], declared [200, 200]
t2 = pypto_pro.language.make_tensor(x, [200, 200])
```

```text
===== kernel 'view_over_source_kernel' launch 9b3f1c2e (./build/view_over_source_kernel__3510__d0/tk_none) =====
PyPTO Sanitizer: 1 issue(s) detected
[1/1] GM_OUT_OF_BOUNDS: make_tensor view [200, 200] covers 80000 bytes, source tensor [128, 128] holds 32768 bytes at demo.py:1255
      Hint: A view shares its source's storage; keep its shape/stride size within the source
```

Field interpretation:

- The square brackets after the make_tensor view declare the shape.
- covers: the number of bytes actually covered after expansion by shape and stride.
- holds: the byte capacity of the source tensor.

Conclusion: The view declaration [200, 200] (covering 80,000 bytes) on line 1255 of demo.py is attached to a [128, 128] (32,768 bytes) source tensor, which indicates that the declaration exceeds the source capacity. The view shape and the actual size of the source tensor need to be checked.

### tile out-of-bounds access

tile out-of-bounds access includes four types: move offset out-of-bounds, insert offset out-of-bounds, dynamic set_validshape window out-of-bounds, and scalar access out-of-bounds.

#### move Offset Out-of-Bounds

Example description: move accesses a tile declared as [64, 64] with offset [32, 0], and the offset plus the window exceeds the declared dimension.

```python
# tile declared as [64, 64], offset 32 + window 64 = 96 > 64
pypto_pro.language.move(a_l0a, cur_a, offset=[32, 0])
```

```text
===== kernel 'tile_move_offset_oob_kernel' launch 4d2a80be (./build/tile_move_offset_oob_kernel__3510__d0/tk_none) =====
PyPTO Sanitizer: 1 issue(s) detected
[1/1] TILE_OUT_OF_BOUNDS: offsets [32, 0] + valid_shape [64, 64] exceed tile shape [64, 64] — dim0: offset 32 + valid_shape 64 = 96 > 64 (over by 32) at demo.py:227
      Hint: Keep the tile's valid shape within the declared TileType dims
```

Field interpretation:

- The square bracket following exceed tile shape: the TileShape declared by TileType (the corresponding position in GM out-of-bounds access is the tensor shape).

Conclusion: the move at line 227 of demo.py accesses a tile declared as [64, 64] with offset [32, 0]. In the row direction, 32 + 64 = 96 exceeds 64 by 32 elements. This can be located as the offset within the tile exceeding the declared dimension, and the movement offset needs to be checked, or a larger tile should be used instead.

#### insert Offset Out-of-Bounds

Example description: insert writes the [64, 64] src to the offset [96, 0] of the dst declared as [128, 64]. The write offset plus the window exceeds the declared dimension.

```python
# dst is declared as [128, 64], src window is [64, 64], write offset is 96: 96 + 64 = 160 > 128
pypto_pro.language.insert(dst, src, offset=[96, 0])
```

```text
===== kernel 'insert_oob_kernel' launch 7f51c39d (./build/insert_oob_kernel__3510__d0/tk_none) =====
PyPTO Sanitizer: 1 issue(s) detected
[1/1] TILE_OUT_OF_BOUNDS: offsets [96, 0] + valid_shape [64, 64] exceed tile shape [128, 64] — dim0: offset 96 + valid_shape 64 = 160 > 128 (over by 32) at demo.py:64
      Hint: Keep the tile's valid shape within the declared TileType dims
```

Field interpretation:

- The square bracket following "exceed tile shape": the declared dimension of the destination tile (dst).
- offsets and valid_shape: the write offset and the valid window of src. Their sum must not exceed the declared dimension of dst.

Conclusion: the insert on line 64 of demo.py writes the [64, 64] src to the offset [96, 0] of the [128, 64] dst. In the row direction, 96 + 64 = 160 exceeds 128 by 32 elements. This can be located as an insufficient declared dimension of the destination tile or an excessively large write offset.

#### Dynamic set_validshape Window Out-of-Bounds

Example description: The dynamic window m set by set_validshape is 65 at runtime, which exceeds the declared dimension 64 of the tile. The window setting statement itself reports TILE_OUT_OF_BOUNDS, and subsequent accesses using the out-of-range window also report GM_OUT_OF_BOUNDS as a side effect.

```python
# tile is declared as [64, 64], dynamic window m is 65 at runtime (not capped by min)
pypto_pro.language.set_validshape(t, [m, 64])
pypto_pro.language.load(t, a, [0, 0])
pypto_pro.language.store(out, t, [0, 0])
```

```text
===== kernel 'validshape_oob_kernel' launch 2a8e64cf (./build/validshape_oob_kernel__3510__d0/tk_none) =====
PyPTO Sanitizer: 3 issue(s) detected
[1/3] GM_OUT_OF_BOUNDS: offsets [0, 0] + valid_shape [65, 64] exceed tensor shape (64, 64) — dim0: offset 0 + valid_shape 65 = 65 > 64 (over by 1) at demo.py:46
      Hint: The tile valid_shape plus the loop offsets must stay inside the tensor logical shape on every dimension
[2/3] GM_OUT_OF_BOUNDS: offsets [0, 0] + valid_shape [65, 64] exceed tensor shape (64, 64) — dim0: offset 0 + valid_shape 65 = 65 > 64 (over by 1) at demo.py:47
      Hint: The tile valid_shape plus the loop offsets must stay inside the tensor logical shape on every dimension
[3/3] TILE_OUT_OF_BOUNDS: offsets [0, 0] + valid_shape [65, 64] exceed tile shape [64, 64] — dim0: offset 0 + valid_shape 65 = 65 > 64 (over by 1) at demo.py:45
      Hint: Keep the tile's valid shape within the declared TileType dims
```

Field interpretation:

- [3/3]: Out-of-bounds of the window setting statement itself (the window [65, 64] set by set_validshape on line 45 of demo.py exceeds the tile declaration [64, 64]).
- [1/3] and [2/3]: Side-effect errors reported by subsequent load/store operations that access the tensor using the out-of-range window.

Conclusion: The dynamic window is not capped to the declared dimension of the tile. The upper bound of the window needs to be limited using min or similar means.

#### Scalar Access Out-of-Bounds

Example description: The linear offset 4096 of getval exceeds the 4096 elements of the [64, 64] tile.

```python
# tile is declared as [64, 64] (4096 elements), linear offset 4096: out-of-bounds
pypto_pro.language.setval(out, 0, pypto_pro.language.getval(t, 4096))
```

```text
===== kernel 'tile_scalar_oob_kernel' launch 5c0d97b1 (./build/tile_scalar_oob_kernel__3510__d0/tk_none) =====
PyPTO Sanitizer: 1 issue(s) detected
[1/1] TILE_OUT_OF_BOUNDS: scalar tile access at linear offset 4096 exceeds 4096 elements (dims [64,64]) at demo.py:31
      Hint: getval/setval offsets must stay inside the tile element count
```

Field interpretation:

- Square brackets after dims: the declared dimensions of the tile.

Conclusion: getval on line 31 of demo.py reads sequence number 4096 (the last valid sequence number is 4095) on the [64, 64] (4096 elements) tile, which can be located as a tile scalar access out-of-bounds. The calculation of this offset needs to be checked.

### Tile Address Overlap

Example description: The address ranges of two tiles on the UB overlap, and a WARNING is reported (this may be an intentional reuse arrangement or an address planning error).

```python
ta_group = pypto_pro.language.make_tile_group(type=tt, addrs=0x0000, mutex_ids=[0])
tb_group = pypto_pro.language.make_tile_group(type=tt, addrs=0x1000, mutex_ids=[1])  # overlaps with the address of ta_group
ta = ta_group.current()
tb = tb_group.current()
```

```text
===== kernel 'overlap_kernel' launch 3b7e05d8 (./build/overlap_kernel__3510__d0/tk_none) =====
PyPTO Sanitizer: 1 issue(s) detected
[1/1] TILE_OVERLAP: tile range [Vec 0x0, 0x2000) overlaps another tile [Vec 0x1000, 0x3000) at demo.py:124
      Hint: Two tiles must not share on-chip address space
```

Field interpretation:

- Square brackets after "tile range": the memory space and address range (left-closed, right-open) of the tile where the detected item resides.
- Square brackets after "overlaps another tile": the range of the other tile that overlaps it.
- Same memory space: both are located in Vec (UB).

Conclusion: The two tiles on line 124 of demo.py occupy [0x0, 0x2000) and [0x1000, 0x3000) on the UB respectively, overlapping 0x1000 bytes starting from 0x1000. Check the addr planning of the two tiles or confirm whether the reuse is intentional.

### mutex Pairing Error

mutex pairing errors include three types: unlock without lock, lock without unlock, and repeated lock.

#### Unlock Without Lock

Example description: mutex_unlock is not paired with a mutex_lock of the same (region, mutex_id, pipe).

```python
# Unlock without lock
pypto_pro.language.system.mutex_unlock(mutex_id=7, pipe=pypto_pro.language.PipeType.MTE2)
```

```text
===== kernel 'mutex_unlock_before_lock_kernel' launch e9120fa4 (./build/mutex_unlock_before_lock_kernel__3510__d0/tk_none) =====
PyPTO Sanitizer: 1 issue(s) detected
[1/1] MUTEX_UNLOCK_BEFORE_LOCK: mutex_unlock(mutex_id=7, pipe=MTE2, region=1) without a preceding mutex_lock at demo.py:705
      Hint: Every mutex_unlock must be paired with a mutex_lock on the same (region, mutex_id, pipe)
```

Field interpretation:

- region: The execution core number (mutex hardware slots of different execution cores are independent, and pairing is distinguished by region).
- mutex_id: The lock number.
- pipe: The pipeline protected by this lock.

Conclusion: The mutex_unlock at line 705 of demo.py unlocks the lock with mutex_id 7 and pipe MTE2 on region 1, but there is no corresponding mutex_lock before it. This can be identified as an unlock missing its paired lock; add the lock or remove the redundant unlock.

#### Repeated Lock

Example description: The same (region, mutex_id, pipe) calls mutex_lock again before the previous unlock. The second lock cannot obtain the already-occupied lock slot, and the pipe will be permanently blocked (MUTEX_DOUBLE_LOCK).

```python
# The same lock is locked twice consecutively without an unlock in between.
pypto_pro.language.system.mutex_lock(mutex_id=7, pipe=pypto_pro.language.PipeType.MTE2)
pypto_pro.language.system.mutex_lock(mutex_id=7, pipe=pypto_pro.language.PipeType.MTE2)
```

```text
===== kernel 'mutex_double_lock_kernel' launch 2e58c9a1 (./build/mutex_double_lock_kernel__3510__d0/tk_none) =====
PyPTO Sanitizer: 1 issue(s) detected
[1/1] MUTEX_DOUBLE_LOCK: mutex_lock(mutex_id=7, pipe=MTE2, region=0) executes before the previous lock was unlocked and will block this pipe permanently at demo.py:15
      Hint: A mutex key must alternate lock and unlock; do not lock the same (region, mutex_id, pipe) twice
```

Field interpretation:

- executes before the previous lock was unlocked: When the second lock on the same pairing key occurs, the previous lock has not yet been unlocked.
- will block this pipe permanently: The lock slot is already occupied, and the second lock cannot wait for its release, so subsequent instructions on this pipe cannot be executed.

Conclusion: Line 15 of demo.py repeatedly locks the lock with mutex_id 7 and pipe MTE2. Delete the redundant lock, or insert a paired unlock before it.

In addition, if no paired mutex_unlock is received by the time the kernel ends after mutex_lock is executed, UNPAIRED_MUTEX_LOCK is reported, and the report location is anchored at the dangling mutex_lock statement. An unreleased lock slot prevents subsequent locks with the same key from being obtained, which, like a repeated lock, ultimately manifests as blocking of that pipe.

### Phase (AccPhase/STPhase) Error

Phase detection targets the AccPhase of matmul and the STPhase of store/move/insert (hardware unit_flag synchronization). Detection includes two types: PHASE_MISMATCH (in-place contract violation, mixed use of synchronization mechanisms, K chain structure error) and PHASE_DEADLOCK (flag state machine deadlock). The in-place contract means that the dst and acc of matmul_acc/matmul_mx_acc must be the same Tile, that is, in-place accumulation. The following examples are all Cube-segment matmul scenarios and do not cover the in-place contract violation subtype.

#### matmul with Phase, store Missing STPhase

Example description: matmul has used AccPhase.Final to declare the hardware unit_flag handshake, but the store that consumes it is not configured with STPhase, mixing two synchronization mechanisms. Fixpipe may read the data before matmul completes (PHASE_MISMATCH).

```python
@pypto_pro.language.jit(sanitizer=True, auto_mutex=True)
def phase_mixed_sync_kernel(a: pypto_pro.language.Tensor[[64, 64], pypto_pro.language.DT_FP16], b: pypto_pro.language.Tensor[[64, 64], pypto_pro.language.DT_FP16], out: pypto_pro.language.Tensor[[64, 64], pypto_pro.language.DT_FP32]):
    mat_a = pypto_pro.language.make_tile_group(type=pypto_pro.language.TileType(shape=[64, 64], dtype=pypto_pro.language.DT_FP16, target_memory=pypto_pro.language.MemorySpace.Mat), addrs=0x0000, mutex_ids=[0])
    mat_b = pypto_pro.language.make_tile_group(type=pypto_pro.language.TileType(shape=[64, 64], dtype=pypto_pro.language.DT_FP16, target_memory=pypto_pro.language.MemorySpace.Mat), addrs=0x8000, mutex_ids=[1])
    left_g = pypto_pro.language.make_tile_group(type=pypto_pro.language.TileType(shape=[64, 64], dtype=pypto_pro.language.DT_FP16, target_memory=pypto_pro.language.MemorySpace.Left), addrs=0x0000, mutex_ids=[2])
    right_g = pypto_pro.language.make_tile_group(type=pypto_pro.language.TileType(shape=[64, 64], dtype=pypto_pro.language.DT_FP16, target_memory=pypto_pro.language.MemorySpace.Right), addrs=0x0000, mutex_ids=[3])
    acc_g = pypto_pro.language.make_tile_group(type=pypto_pro.language.TileType(shape=[64, 64], dtype=pypto_pro.language.DT_FP32, target_memory=pypto_pro.language.MemorySpace.Acc), addrs=0x0000, mutex_ids=[4])
    with pypto_pro.language.section_cube():
        ma = mat_a.current()
        mb = mat_b.current()
        pypto_pro.language.load(ma, a, [0, 0])
        pypto_pro.language.load(mb, b, [0, 0])
        al = left_g.current()
        br = right_g.current()
        pypto_pro.language.move(al, ma)
        pypto_pro.language.move(br, mb)
        ac = acc_g.current()
        pypto_pro.language.matmul(ac, al, br, phase=pypto_pro.language.AccPhase.Final)
        # Violation point: matmul carries phase, but the store that consumes it is not configured with STPhase.
        pypto_pro.language.store(out, ac, [0, 0])
```

```text
===== kernel 'phase_mixed_sync_kernel' launch 3e7f1a2b (./build/phase_mixed_sync_kernel__3510__d0/tk_none) =====
PyPTO Sanitizer: 1 issue(s) detected
[1/1] PHASE_MISMATCH: region 0, Acc tile 0x0: matmul uses phase (hardware unit_flag sync) while a store/move on the same block omits STPhase (software sync) -- the two mechanisms mix and Fixpipe may read data before the matmul finishes at demo.py:82
      Hint: When any matmul of a chain uses phase, the consuming store/move must use STPhase as well
```

Field interpretation:

- region / Acc tile: The phase state is tracked independently by (execution core, Acc tile address), and the report anchors to the statement where the mixing occurs.
- Two synchronization mechanisms: The write of matmul with phase is ordered by the hardware unit_flag handshake, while the read of store without STPhase is ordered by software synchronization. No ordering relationship is formed between the two.

Conclusion: The store on line 82 of demo.py consumes the matmul result with AccPhase.Final, but it does not carry STPhase.Final itself. Fixpipe may read the data before matmul completes. You need to add phase=STPhase.Final to the store, or remove phase from both sides at the same time.

#### Final Write Without Read Consumption

Example description: After the last block of the K chain completes its write with AccPhase.Final, no store/move with STPhase.Final consumes it, so the flag of that Acc tile remains set when the kernel ends. This launch can run to completion, but the next write to the same Acc tile will wait for the flag to be cleared and deadlock (PHASE_DEADLOCK).

```python
    # Tile declaration and loading are the same as in the previous example
    with pypto_pro.language.section_cube():
        ...
        # Violation point: the store is missing after the Final write, so this Final is consumed by no one
        pypto_pro.language.matmul(ac, al, br, phase=pypto_pro.language.AccPhase.Final)
```

```text
===== kernel 'phase_dangling_final_kernel' launch 7c21d0e8 (./build/phase_dangling_final_kernel__3510__d0/tk_none) =====
PyPTO Sanitizer: 1 issue(s) detected
[1/1] PHASE_DEADLOCK: region 0, Acc tile 0x0 ends with the unit_flag still set: the Final write has no Final read to consume it at demo.py:95
      Hint: A Final write must be followed by a Final read (store/move) in the same chain
```

Field interpretation:

- ends with the unit_flag still set: When replaying to the end of the kernel, the unit_flag of this (region, Acc tile) is still set by the Final write and is not cleared by any Final read.
- Report location: anchored at the Final write statement that set the flag.

Conclusion: The matmul on line 95 of demo.py writes the Acc tile with AccPhase.Final, but no Final read consumes it before the kernel ends, leaving the flag dangling. You need to add the missing store, or, if no data needs to be moved out, remove the phase and use software synchronization instead.

#### K Chain Head Block Misusing matmul_acc

Example description: When the same Acc tile is reused by splitting along the M dimension, the chain head block in the second loop iteration misuses matmul_acc (the middle-segment code was copied but not corrected). The partial sum is accumulated onto the residual value from the previous iteration, resulting in a silent error (PHASE_MISMATCH).

```python
    # Tile declaration and loading are the same as in the previous example, where a is [128, 128] and out is [128, 64]
    with pypto_pro.language.section_cube():
        ac = acc_g.current()
        for i in pypto_pro.language.range(0, 128, 64):
            for k in pypto_pro.language.range(0, 128, 64):
                # al/br: load the operands of the current iteration by [i, k] (same as in the previous example)
                if k == 0:
                    if i == 0:
                        pypto_pro.language.matmul(ac, al, br, phase=pypto_pro.language.AccPhase.Partial)
                    else:
                        # Violation point: the chain head in the second iteration misuses matmul_acc, accumulating onto the residual partial sum from the previous iteration
                        pypto_pro.language.matmul_acc(ac, ac, al, br, phase=pypto_pro.language.AccPhase.Partial)
                else:
                    pypto_pro.language.matmul_acc(ac, ac, al, br, phase=pypto_pro.language.AccPhase.Final)
            pypto_pro.language.store(out, ac, [i, 0], phase=pypto_pro.language.STPhase.Final)
```

```text
===== kernel 'phase_acc_chain_head_kernel' launch 41ab09f7 (./build/phase_acc_chain_head_kernel__3510__d0/tk_none) =====
PyPTO Sanitizer: 1 issue(s) detected
[1/1] PHASE_MISMATCH: matmul_acc is the first write of a chain on region 0, Acc tile 0x0: it accumulates onto leftover or uninitialized L0C data instead of overwriting it at demo.py:118
      Hint: Start an in-place chain with matmul (overwrite), not matmul_acc
```

Field interpretation:

- K chain: the entire K-loop write between two stores (Final reads) on the same Acc tile. The chain head must use an overwrite (matmul or matmul_bias) to establish the base value; only the middle segment and the last block use matmul_acc.
- is the first write of a chain: in the second i-loop iteration, the first write of the new chain is matmul_acc, which does not overwrite first but directly accumulates onto the residual partial sum from the previous iteration.

Conclusion: The matmul_acc on line 118 of demo.py is the first write of a new K chain, but its accumulation semantics treat the residual value from the previous iteration as the base value, causing the output to be silently too large. The chain head must be changed back to matmul (or matmul_bias) for an overwrite. Symmetrically, matmul appearing in the middle segment of a chain overwrites and discards the existing partial sum; matmul_bias appearing in the middle segment of a chain performs L0C Buffer initialization and clears the existing partial sum. Both belong to the same type of K chain structure error.

### sync pairing error

sync detection covers pairing errors of sync_src/sync_dst. sync_src and sync_dst form a pair only when their four parameters region, set_pipe, wait_pipe, and event_id are completely identical. Pairing errors include three types: SYNC_UNPAIRED_SRC (after src is issued, no paired dst waits for it, causing a data race), SYNC_UNPAIRED_DST (the paired src that dst waits for does not exist, causing a deadlock), and SYNC_EVENT_REUSE (the same event_id is set again before dst completes the wait).

#### Missing sync_dst

Example description: After the MTE2 movement, only sync_src is issued and the paired sync_dst is missing. The V-side computation does not form an order with the movement, resulting in a data race (SYNC_UNPAIRED_SRC).

```python
@pypto_pro.language.jit(sanitizer=True)
def sync_missing_dst_kernel(a: pypto_pro.language.Tensor[[64, 64], pypto_pro.language.DT_FP32], b: pypto_pro.language.Tensor[[64, 64], pypto_pro.language.DT_FP32], out: pypto_pro.language.Tensor[[64, 64], pypto_pro.language.DT_FP32]):
    tt = pypto_pro.language.TileType(shape=[64, 64], dtype=pypto_pro.language.DT_FP32, target_memory=pypto_pro.language.MemorySpace.Vec)
    ta = pypto_pro.language.make_tile(tt, addr=0x0000)
    tb = pypto_pro.language.make_tile(tt, addr=0x4000)
    tc = pypto_pro.language.make_tile(tt, addr=0x8000)
    with pypto_pro.language.section_vector():
        pypto_pro.language.load(ta, a, [0, 0])
        pypto_pro.language.load(tb, b, [0, 0])
        # Violation point: sync_dst(set_pipe=MTE2, wait_pipe=V, event_id=0) is missing.
        pypto_pro.language.system.sync_src(set_pipe=pypto_pro.language.PipeType.MTE2, wait_pipe=pypto_pro.language.PipeType.V, event_id=0)
        pypto_pro.language.add(tc, ta, tb)
        pypto_pro.language.store(out, tc, [0, 0])
```

```text
===== kernel 'sync_missing_dst_kernel' launch 8d40e6b2 (./build/sync_missing_dst_kernel__3510__d0/tk_none) =====
PyPTO Sanitizer: 1 issue(s) detected
[1/1] SYNC_UNPAIRED_SRC: sync_src on region 0, MTE2 -> V, event_id 0 sets a flag no sync_dst waits for (data race: the consumer runs unordered against the producer) at demo.py:217
      Hint: Every sync_src must be followed by a matching sync_dst with the same set_pipe, wait_pipe and event_id
```

Field interpretation:

- region / set_pipe / wait_pipe / event_id: The four pairing key elements, namely the execution core, set pipeline, wait pipeline, and event ID. Only when all four are identical do they form a pair.
- data race: No one waits for this flag, so there is no order guarantee between the V-side add/store and the MTE2 movement.

Conclusion: The sync_src at line 217 of demo.py sets event 0 but no sync_dst waits for it. You need to add sync_dst(set_pipe=MTE2, wait_pipe=V, event_id=0) immediately after it to ensure that the movement is complete when add is executed.

Symmetric to the missing sync_dst, when a sync_dst appears but no paired sync_src precedes it, SYNC_UNPAIRED_DST is reported. The flag it waits for is never set, so this wait never completes (deadlock).

#### Event Reused Before Being Consumed

Example description: The sync_src of the same event_id is set again before the paired sync_dst completes its wait.

```python
# Double-buffered movement reuses event 0, but the second round's sync_src sets it before the sync_dst wait
for i in pypto_pro.language.range(0, 2):
    pypto_pro.language.load(ta, a, [i * 64, 0])
    # Violation point: the sync_src at i=1 fires again before sync_dst waits for event 0
    pypto_pro.language.system.sync_src(set_pipe=pypto_pro.language.PipeType.MTE2, wait_pipe=pypto_pro.language.PipeType.V, event_id=0)
pypto_pro.language.system.sync_dst(set_pipe=pypto_pro.language.PipeType.MTE2, wait_pipe=pypto_pro.language.PipeType.V, event_id=0)
```

```text
===== kernel 'sync_event_reuse_kernel' launch 6a3b5c7e (./build/sync_event_reuse_kernel__3510__d0/tk_none) =====
PyPTO Sanitizer: 1 issue(s) detected
[1/1] SYNC_EVENT_REUSE: sync_src on region 0, MTE2 -> V, event_id 0 fires again before the previous sync_dst consumed the event (the pending flag is overwritten) at demo.py:232
      Hint: The same event_id may only be reused after its sync_dst has completed the wait
```

Field interpretation:

- fires again before the previous sync_dst consumed the event: When the second sync_src appears, no sync_dst has yet waited for the previously set event 0.
- the pending flag is overwritten: The hardware flag has only one bit, so the second set overwrites the first, and the waiting side cannot distinguish which set operation it actually waited for.

Conclusion: The sync_src on line 232 of demo.py (the i=1 round) sets the same event again before sync_dst waits for event 0. Before reusing the same event_id, you must first let the paired sync_dst complete its wait, or use a new event_id for the new round of movement.
