# Memory Sanitizer

<!-- md-trans-meta sourceCommit=49c9842f2975007e96468639469c30745f0bd315 translatedAt=2026-10-27T01:48:28.271Z pushedAt=2026-10-27T06:24:06.474Z nl=2 -->

## Description

Memory access errors in the Kernel (GM out-of-bounds, tile out-of-bounds, on-chip address overlap, and mutex pairing errors) usually do not immediately cause exceptions on the Host side. Instead, they manifest as incorrect results, intermittent errors, or Kernel hangs, which are costly to troubleshoot manually. The memory sanitizer is enabled by setting sanitizer=True in @pypto_pro.language.jit(): the framework automatically instruments GM/tile access points and mutex call sites, and after each Kernel launch completes, it performs a unified check and reports the conclusion.

## Constraints

- For debugging only: instrumentation and checking change the Kernel execution overhead. Before performance measurement, remove sanitizer=True from @pypto_pro.language.jit().
- Each Kernel launch allocates a log buffer of about 1 GB on GM. Repeated defects at the same location are aggregated into a single entry with a count.
- pypto_pro.language.Ptr carries only the address, not the size of the memory it points to. A view whose source is a pointer cannot determine the source size, so the check for a declared shape that exceeds the source is not performed.
- The check covers accesses and declarations within the Kernel body. All accesses (GM/tile/mutex) within SIMT thread blocks are not checked.

## Procedure

### Enabling Memory Sanitizer

Set `@pypto_pro.language.jit(sanitizer=True)` on the kernel function decorator:

```python
import pypto_pro.language as pl

@pl.jit(sanitizer=True, auto_mutex=True)
def kernel(x: pl.Tensor[[64, 64], pl.DT_FP16]):
    ...
```

sanitizer defaults to False. When it is disabled, the compilation and execution behavior of the Kernel remains completely unchanged. After it is enabled, detection runs automatically each time the Kernel is launched, and passes silently if no issue is found. When an issue is detected, the result is written in append mode to `pypto_sanitizer_report.txt` in the compilation output directory of the kernel (in the same directory as `sanitizer_report.bin`, with a path like `./build/<kernel_name>__<arch>/tk_none/`). Results from multiple launches of the same kernel are aggregated in the same file.

### Detection Report Description

Each result block in the report file starts with a kernel tag line, with the compilation output directory of the kernel used for that launch in parentheses, for example `(./build/view_over_source_kernel__3510__d0/tk_none)`. When multiple build instances of the same kernel exist (with different TilingKey, data types, or devices), this directory is used to distinguish which instance a detection result comes from. This directory also contains the detection result file `pypto_sanitizer_report.txt` and the raw record `sanitizer_report.bin`.

Each result block first gives the number of detections, then lists the detected items one by one. Taking GM out-of-bounds access as an example:

```text
===== kernel 'oob_kernel' (./build/oob_kernel__3510__d0/tk_none) =====
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
| MUTEX_UNLOCK_BEFORE_LOCK / UNPAIRED_MUTEX_LOCK | ERROR | mutex_lock/mutex_unlock are not paired with the same (region, mutex_id, pipe). |

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
===== kernel 'oob_kernel' (./build/oob_kernel__3510__d0/tk_none) =====
PyPTO Sanitizer: 1 issue(s) detected
[1/1] GM_OUT_OF_BOUNDS: offsets [32, 0] + valid_shape [64, 64] exceed tensor shape (64, 64) — dim0: offset 32 + valid_shape 64 = 96 > 64 (over by 32) at demo.py:68
      Hint: The tile valid_shape plus the loop offsets must stay inside the tensor logical shape on every dimension
```

Field interpretation:

- offsets: The starting position of this access in each dimension of the tensor (for example, the third parameter of load).
- valid_shape: The tile valid window used by the access.

Conclusion: The load on line 68 of demo.py accesses a [64, 64] tensor with offset [32, 0]. In the row direction, 32 + 64 = 96 exceeds 64 by 32 elements. This can be located as a loop offset that is not capped by the tensor row count, and the loop boundary or offset calculation needs to be checked.

#### scalar access out-of-bounds

Example description: The linear offset 64 of setval exceeds the 64 elements of the tensor.

```python
@pypto_pro.language.jit(sanitizer=True, auto_mutex=True)
def gm_scalar_oob_kernel(out: pypto_pro.language.Tensor[[64], pypto_pro.language.DT_INT32]):
    with pypto_pro.language.section_vector():
        # Linear offset 64: exceeds 64 elements
        pypto_pro.language.setval(out, 64, 1)
```

```text
===== kernel 'gm_scalar_oob_kernel' (./build/gm_scalar_oob_kernel__3510__d0/tk_none) =====
PyPTO Sanitizer: 1 issue(s) detected
[1/1] GM_OUT_OF_BOUNDS: scalar access at linear offset 64 exceeds 64 elements at demo.py:17
      Hint: getval/setval offsets must stay inside the tensor element count
```

Field interpretation:

- linear offset: element index on the tensor.

Conclusion: The setval at line 17 of demo.py writes index 64 (the last valid index is 63) on a tensor of 64 elements. This can be located as a scalar write position out-of-bounds, and the upper bound of the offset needs to be checked.

#### make_tensor declaration shape exceeds source tensor

Example description: make_tensor declares a [200, 200] view on a [128, 128] source tensor. The shared storage causes the declaration to exceed the source capacity, which is detected at declaration time.

```python
# Source x is [128, 128], declaring [200, 200]
t2 = pypto_pro.language.make_tensor(x, [200, 200])
```

```text
===== kernel 'view_over_source_kernel' (./build/view_over_source_kernel__3510__d0/tk_none) =====
PyPTO Sanitizer: 1 issue(s) detected
[1/1] GM_OUT_OF_BOUNDS: make_tensor view [200, 200] covers 80000 bytes, source tensor [128, 128] holds 32768 bytes at demo.py:1255
      Hint: A view shares its source's storage; keep its shape/stride size within the source
```

Field interpretation:

- The square brackets after the make_tensor view: declare the shape.
- covers: the number of bytes actually covered after expansion by shape and stride.
- holds: the byte capacity of the source tensor.

Conclusion: The view declaration [200, 200] (covering 80,000 bytes) on line 1255 of demo.py is attached to a [128, 128] (32,768 bytes) source tensor, which indicates that the declaration exceeds the source capacity. The view shape and the actual size of the source tensor need to be checked.

### tile out-of-bounds access

tile out-of-bounds access includes four types: move offset out-of-bounds, insert offset out-of-bounds, dynamic set_validshape window out-of-bounds, and scalar access out-of-bounds.

#### move offset out-of-bounds

Example description: move accesses a tile declared as [64, 64] with offset [32, 0], and the offset plus the window exceeds the declared dimension.

```python
# tile is declared as [64, 64], offset 32 + window 64 = 96 > 64
pypto_pro.language.move(a_l0a, cur_a, offset=[32, 0])
```

```text
===== kernel 'tile_move_offset_oob_kernel' (./build/tile_move_offset_oob_kernel__3510__d0/tk_none) =====
PyPTO Sanitizer: 1 issue(s) detected
[1/1] TILE_OUT_OF_BOUNDS: offsets [32, 0] + valid_shape [64, 64] exceed tile shape [64, 64] — dim0: offset 32 + valid_shape 64 = 96 > 64 (over by 32) at demo.py:227
      Hint: Keep the tile's valid shape within the declared TileType dims
```

Field interpretation:

- The square bracket following exceed tile shape: the tile shape declared by TileType (the corresponding position in GM out-of-bounds access is the tensor shape).

Conclusion: the move at line 227 of demo.py accesses a tile declared as [64, 64] with offset [32, 0]. In the row direction, 32 + 64 = 96 exceeds 64 by 32 elements. This can be located as the offset within the tile exceeding the declared dimension, and the copy offset needs to be checked, or a larger tile should be used instead.

#### insert offset out-of-bounds

Example description: insert writes the [64, 64] src to the offset [96, 0] of the dst declared as [128, 64]. The write offset plus the window exceeds the declared dimension.

```python
# dst is declared as [128, 64], src window is [64, 64], write offset is 96: 96 + 64 = 160 > 128
pypto_pro.language.insert(dst, src, offset=[96, 0])
```

```text
===== kernel 'insert_oob_kernel' (./build/insert_oob_kernel__3510__d0/tk_none) =====
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
# The tile is declared as [64, 64], and the dynamic window m is 65 at runtime (not capped with min)
pypto_pro.language.set_validshape(t, [m, 64])
pypto_pro.language.load(t, a, [0, 0])
pypto_pro.language.store(out, t, [0, 0])
```

```text
===== kernel 'validshape_oob_kernel' (./build/validshape_oob_kernel__3510__d0/tk_none) =====
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
# tile declared as [64, 64] (4096 elements), linear offset 4096: out of bounds
pypto_pro.language.setval(out, 0, pypto_pro.language.getval(t, 4096))
```

```text
===== kernel 'tile_scalar_oob_kernel' (./build/tile_scalar_oob_kernel__3510__d0/tk_none) =====
PyPTO Sanitizer: 1 issue(s) detected
[1/1] TILE_OUT_OF_BOUNDS: scalar tile access at linear offset 4096 exceeds 4096 elements (dims [64,64]) at demo.py:31
      Hint: getval/setval offsets must stay inside the tile element count
```

Field interpretation:

- Square brackets after dims: the declared dimensions of the tile.

Conclusion: getval on line 31 of demo.py reads sequence number 4096 (the last valid sequence number is 4095) on the [64, 64] (4096 elements) tile, which can be located as a tile scalar access out-of-bounds. The calculation of this offset needs to be checked.

### tile Address Overlap

Example description: The address ranges of two tiles on the UB overlap, and a WARNING is reported (this may be an intentional reuse arrangement or an address planning error).

```python
ta_group = pypto_pro.language.make_tile_group(type=tt, addrs=0x0000, mutex_ids=[0])
tb_group = pypto_pro.language.make_tile_group(type=tt, addrs=0x1000, mutex_ids=[1])  # Overlaps the address range of ta_group
ta = ta_group.current()
tb = tb_group.current()
```

```text
===== kernel 'overlap_kernel' (./build/overlap_kernel__3510__d0/tk_none) =====
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

Example description: mutex_unlock is not paired with a mutex_lock of the same (region, mutex_id, pipe).

```python
# Unlock without lock
pypto_pro.language.system.mutex_unlock(mutex_id=7, pipe=pypto_pro.language.PipeType.MTE2)
```

```text
===== kernel 'mutex_unlock_before_lock_kernel' (./build/mutex_unlock_before_lock_kernel__3510__d0/tk_none) =====
PyPTO Sanitizer: 1 issue(s) detected
[1/1] MUTEX_UNLOCK_BEFORE_LOCK: mutex_unlock(mutex_id=7, pipe=MTE2, region=1) without a preceding mutex_lock at demo.py:705
      Hint: Every mutex_unlock must be paired with a mutex_lock on the same (region, mutex_id, pipe)
```

Field interpretation:

- region: The execution core number (mutex hardware slots of different execution cores are independent, and pairing is distinguished by region).
- mutex_id: The lock number.
- pipe: The pipeline protected by this lock.

Conclusion: The mutex_unlock at line 705 of demo.py unlocks the lock with mutex_id 7 and pipe MTE2 on region 1, but there is no corresponding mutex_lock before it. This can be located as an unlock missing its paired lock, and needs to be checked by adding the lock or removing the redundant unlock.
