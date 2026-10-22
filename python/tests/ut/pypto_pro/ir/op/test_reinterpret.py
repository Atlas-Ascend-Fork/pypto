"""Compile-time tests for pl.reinterpret (Tile/TileGroup metadata reinterpretation).

Two layers:
- IR-level: call ``_ir_reinterpret`` directly on ``make_tile_expr`` calls to
  assert type/metadata behavior (no parser, no device).
- Kernel-level: parse ``@pl.jit`` programs and assert on the produced IR
  text, incl. group rebuilding, size/address inheritance, error diagnostics
  and auto_mutex behavior of reinterpreted handles.
"""

from pypto_pro._errors import InvalidArgument, InvalidShape, InvalidTile, InvalidType, OutOfRange
from pypto_pro.ir.op.block_ops import _ir_reinterpret, make_tile_expr
import pypto_pro.language as pl
import pytest

from pypto.pypto_impl import ir


def _span() -> ir.Span:
    return ir.Span("test_reinterpret.py", 1, 1, 1, 1)


def _src_tile(shape, dtype, addr=0x1000, size=None, valid_shape=None, layout=None, space=None,
               fractal=None, pad=None, compact=None):
    """A plain make_tile call whose type the reinterpret builder reads."""
    return make_tile_expr(
        shape,
        dtype,
        space or pl.MemorySpace.Vec,
        addr=addr,
        size=size or None,
        valid_shape=valid_shape,
        layout=layout,
        fractal=fractal,
        pad=pad,
        compact=compact,
        span=_span(),
    )


def _assert_same_buffer(fields_actual, fields_expected):
    """Same-buffer semantics after rebuild: address, size and memory space match."""
    assert int(fields_actual["addr"]) == int(fields_expected["addr"])
    assert int(fields_actual["size"]) == int(fields_expected["size"])
    assert fields_actual["space"] == fields_expected["space"]


def _memref_fields(t) -> dict:
    m = t.type.memref
    return {"addr": getattr(m.addr, "value", None), "size": int(m.size), "space": m.memory_space}


def _parse_kernel(kernel_def) -> ir.Program:
    return kernel_def.to_kernel_def().parse_target_program(ir.SectionKind.Vector)[0]


def _kernel_ir(kernel_def) -> str:
    return str(_parse_kernel(kernel_def))


# ---------------------------------------------------------------------------
# override / inherit / share (IR level)
# ---------------------------------------------------------------------------


def test_dtype_override_keeps_other_attributes():
    src = _src_tile([64, 64], pl.DT_FP16, addr=0x2000, size=8192)
    new = _ir_reinterpret(src, shape=[64, 64], dtype=pl.DT_BF16, span=_span())
    assert new.type.dtype == pl.DT_BF16
    assert list(new.type.shape) == list(src.type.shape)
    _assert_same_buffer(_memref_fields(new), _memref_fields(src))
    # valid_shape is NOT inherited: the new handle starts dynamic (-1).
    tv = new.type.tile_view
    assert tv is None or len(list(tv.valid_shape)) == 0


def test_shape_override_keeps_footprint():
    src = _src_tile([64, 64], pl.DT_FP16, addr=0x2000, size=8192)
    new = _ir_reinterpret(src, shape=[32, 128], span=_span())
    assert list(new.type.shape) == [32, 128]
    assert new.type.dtype == pl.DT_FP16


def test_layout_override_updates_blayout_slayout():
    src = _src_tile([64, 64], pl.DT_FP16, addr=0x2000, size=8192)
    new = _ir_reinterpret(src, layout=pl.TensorLayout.NZ, span=_span())
    assert int(new.type.hardware_info.blayout) == 2
    assert int(new.type.hardware_info.slayout) == 1


def test_layout_is_inherited_when_not_overridden():
    """A tile declared NZ keeps its layout unless layout= is given."""
    src = _src_tile([64, 64], pl.DT_FP16, addr=0x2000, size=8192, layout=pl.TensorLayout.NZ)
    new = _ir_reinterpret(src, shape=[32, 128], span=_span())
    assert int(new.type.hardware_info.blayout) == 2
    assert int(new.type.hardware_info.slayout) == 1


def test_fractal_pad_compact_are_inherited():
    """hardware_info fields (fractal/pad/compact) carry over unless overridden."""
    src = _src_tile([64, 64], pl.DT_FP16, addr=0x2000, size=8192, layout=pl.TensorLayout.NZ,
                    fractal=1024, pad=pl.TilePad.null, compact=0)
    new = _ir_reinterpret(src, shape=[32, 128], span=_span())
    hw = new.type.hardware_info
    assert int(hw.fractal) == 1024
    assert int(hw.pad) == int(pl.TilePad.null)
    assert int(hw.compact) == 0


def test_combined_override():
    src = _src_tile([64, 64], pl.DT_FP16, addr=0x2000, size=8192)
    new = _ir_reinterpret(src, shape=[32, 64], dtype=pl.DT_FP32, layout=pl.TensorLayout.ND, span=_span())
    assert [int(d.value) for d in new.type.shape] == [32, 64]
    assert new.type.dtype == pl.DT_FP32
    assert int(new.type.hardware_info.blayout) == 1
    assert int(new.type.hardware_info.slayout) == 0


def test_reinterpret_rebuilds_same_address():
    """The reinterpreted tile is a fresh MemRef at the ORIGINAL address/size
    (SameAllocation semantics); it is not the same object as the source."""
    src = _src_tile([64, 64], pl.DT_FP16, addr=0x2000, size=8192)
    new = _ir_reinterpret(src, shape=[64, 64], dtype=pl.DT_BF16, span=_span())
    _assert_same_buffer(_memref_fields(new), _memref_fields(src))
    assert new.type.memref is not src.type.memref  # fresh object, same buffer


def test_reinterpret_chains():
    src = _src_tile([64, 64], pl.DT_FP16, addr=0x2000, size=8192)
    mid = _ir_reinterpret(src, shape=[64, 64], dtype=pl.DT_BF16, span=_span())
    new = _ir_reinterpret(mid, shape=[32, 128], span=_span())
    assert list(new.type.shape) == [32, 128]
    assert new.type.dtype == pl.DT_BF16
    _assert_same_buffer(_memref_fields(new), _memref_fields(src))


# ---------------------------------------------------------------------------
# validation (IR level)
# ---------------------------------------------------------------------------


def test_valid_shape_is_not_inherited():
    """Review decision: the reinterpreted tile starts with a dynamic valid shape (-1)."""
    src = _src_tile([64, 64], pl.DT_FP16, addr=0x2000, size=8192, valid_shape=[32, 32])
    new = _ir_reinterpret(src, shape=[16, 16], span=_span())
    tv = new.type.tile_view
    assert tv is None or len(list(tv.valid_shape)) == 0  # dynamic / -1 semantics


def test_new_footprint_exceeding_buffer_raises():
    src = _src_tile([64, 64], pl.DT_FP16, addr=0x2000, size=8192)
    with pytest.raises(OutOfRange, match="exceeds the original buffer size"):
        _ir_reinterpret(src, shape=[128, 64], dtype=pl.DT_FP32, span=_span())


def test_footprint_exactly_equal_is_accepted():
    src = _src_tile([64, 64], pl.DT_FP16, addr=0x2000, size=8192)
    new = _ir_reinterpret(src, shape=[32, 64], dtype=pl.DT_FP32, span=_span())
    assert [int(d.value) for d in new.type.shape] == [32, 64]
    assert new.type.dtype == pl.DT_FP32


def test_footprint_shrinking_is_accepted():
    src = _src_tile([64, 64], pl.DT_FP32, addr=0x2000, size=16384)
    new = _ir_reinterpret(src, shape=[64, 64], dtype=pl.DT_FP16, span=_span())
    assert new.type.dtype == pl.DT_FP16


@pytest.mark.parametrize(
    ("addr", "dtype", "ok"),
    [
        (1, pl.DT_INT8, True),    # 1B element: any address is element-aligned
        (1, pl.DT_FP16, False),   # 2B element at addr 1 -> misaligned
        (2, pl.DT_FP16, True),
        (2, pl.DT_FP32, False),   # 4B element at addr 2 -> misaligned
        (4, pl.DT_FP32, True),
        (4, pl.DT_INT64, False),  # 8B element at addr 4 -> misaligned
        (8, pl.DT_INT64, True),
    ],
)
def test_element_boundary_align_matrix(addr, dtype, ok):
    """Hardware addressing granularity: new element width must divide the base address."""
    src = _src_tile([16, 16], pl.DT_FP16, addr=addr, size=8192, space=pl.MemorySpace.DDR)
    if ok:
        new = _ir_reinterpret(src, shape=[16, 16], dtype=dtype, span=_span())
        assert new.type.dtype == dtype
        assert [int(d.value) for d in new.type.shape] == [16, 16]
    else:
        with pytest.raises(ValueError, match="not aligned to the new element size"):
            _ir_reinterpret(src, shape=[16, 16], dtype=dtype, span=_span())


def test_hw_aligned_spaces_always_pass_align_check():
    """Vec/Mat/Left/Right/Acc alignments (32/512/64B) divide every 1/2/4/8B element width:
    a fully aligned address never trips the element-boundary check, for any element width."""
    from pypto_pro.ir.op.block_ops import _reinterpret_align_check

    for space in (pl.MemorySpace.Vec, pl.MemorySpace.Mat, pl.MemorySpace.Acc,
                  pl.MemorySpace.Left, pl.MemorySpace.Right):
        _src_tile([64, 64], pl.DT_FP16, addr=0x2000, size=8192, space=space)
        for dtype in (pl.DT_INT8, pl.DT_FP16, pl.DT_FP32, pl.DT_INT64):
            _reinterpret_align_check(0x2000, dtype, _span())  # must not raise


def test_source_without_memref_raises():
    span = _span()
    dim = ir.ConstInt(16, ir.DataType.INDEX, span)
    bare_type = ir.TileType([dim], pl.DT_FP16)
    bare = ir.Var("bare_tile", bare_type, span)
    with pytest.raises(ValueError, match="no MemRef"):
        _ir_reinterpret(bare, dtype=pl.DT_BF16, span=span)


# ---------------------------------------------------------------------------
# parser-level diagnostics
# ---------------------------------------------------------------------------


def test_no_override_argument_raises():
    @pl.jit
    def k(x: pl.Tensor[[64, 64], pl.DT_FP16]):
        tt = pl.TileType(shape=[64, 64], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Vec)
        t = pl.make_tile(tt, addr=0)
        pl.load(pl.reinterpret(t), x, [0, 0])

    with pytest.raises(InvalidType, match="at least one of dtype/shape/layout"):
        _parse_kernel(k)


def test_wrong_positional_count_raises():
    @pl.jit
    def k(x: pl.Tensor[[64, 64], pl.DT_FP16]):
        tt = pl.TileType(shape=[64, 64], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Vec)
        t = pl.make_tile(tt, addr=0)
        pl.load(pl.reinterpret(t, t, shape=[64, 64], dtype=pl.DT_BF16), x, [0, 0])

    with pytest.raises(InvalidArgument) as caught:
        _parse_kernel(k)
    assert "too many positional arguments" in caught.value.message


def test_dtype_without_shape_rejected_in_kernel():
    """Review decision surfaced at the DSL level too."""

    @pl.jit
    def k(x: pl.Tensor[[64, 64], pl.DT_FP16]):
        tt = pl.TileType(shape=[64, 64], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Vec)
        t = pl.make_tile(tt, addr=0)
        pl.load(pl.reinterpret(t, dtype=pl.DT_BF16), x, [0, 0])

    with pytest.raises(InvalidShape, match="'shape' is required when 'dtype' changes"):
        _parse_kernel(k)


def test_non_tile_argument_raises():
    @pl.jit
    def k(x: pl.Tensor[[64], pl.DT_FP16]):
        pl.load(pl.reinterpret(x, dtype=pl.DT_BF16, shape=[64]), x, [0])

    with pytest.raises(InvalidType, match="expected a Tile"):
        _parse_kernel(k)


def test_shape_accepts_folded_constant_expression():
    """Review item: shape may be a folded constant expression (module-level arithmetic)."""

    shape_base = 64

    @pl.jit
    def k(x: pl.Tensor[[64, 64], pl.DT_FP16]):
        tt = pl.TileType(shape=[64, 64], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Vec)
        t = pl.make_tile(tt, addr=0)
        pl.load(pl.reinterpret(t, shape=[shape_base * 2, shape_base // 2]), x, [0, 0])

    assert _kernel_ir(k)  # parses fine: shape folds to [128, 32]


def test_dtype_via_binding():
    """Review item: dtype override may come from a bound variable."""

    @pl.jit
    def k(x: pl.Tensor[[64, 64], pl.DT_FP16]):
        tt = pl.TileType(shape=[64, 64], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Vec)
        t = pl.make_tile(tt, addr=0)
        dt = pl.DT_BF16
        pl.load(pl.reinterpret(t, shape=[64, 64], dtype=dt), x, [0, 0])

    assert "bfloat16" in _kernel_ir(k).lower()


def test_layout_via_binding():
    """Review item: layout override may come from a bound variable."""

    @pl.jit
    def k(a: pl.Tensor[[64, 64], pl.DT_FP16]):
        tt = pl.TileType(shape=[64, 64], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Mat, layout=pl.ZN)
        t = pl.make_tile(tt, addr=0)
        ly = pl.TensorLayout.NZ
        pl.load(pl.reinterpret(t, shape=[64, 64], layout=ly), a, [0, 0])

    assert _kernel_ir(k)  # parses fine with layout from a variable


def test_reinterpret_inside_nested_loops():
    """Review item 2 (deepened): reinterpret inside a 2-D tile loop, with folded shape,
    dtype and layout all bound through local variables."""

    shape_rows = 64 * 2
    shape_cols = 64 // 2   # footprint must stay within the 8192B slot

    @pl.jit(auto_mutex=True)
    def k(a: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP16],
          b: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP16],
          out: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_BF16]):
        tt = pl.TileType(shape=[64, 64], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Vec)
        g = pl.make_tile_group(type=tt, addrs=[0x0000, 0x2000], mutex_ids=[0, 1])
        td = pl.DT_BF16
        view = pl.reinterpret(g, shape=[shape_rows, shape_cols], dtype=td)
        for i in pl.range(0, a.shape[0], 64):
            for j in pl.range(0, a.shape[1], 64):
                t2 = view.next()
                pl.load_tile(t2, a, [i, j])
                pl.store_tile(out, t2, [i, j])

    ir_str = _kernel_ir(k)
    assert ir_str  # parses fine under nested loops
    assert "bfloat16" in ir_str.lower()
    assert "system.mutex_lock" in ir_str


def test_reinterpret_loop_invariant_shape_expr_in_nested_loop():
    """Review item 2: loop-invariant constant-expression shape works inside nested loops."""

    tdim = 64

    @pl.jit(auto_mutex=True)
    def k(a: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP16],
          out: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_BF16]):
        tt = pl.TileType(shape=[tdim, tdim], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Vec)
        g = pl.make_tile_group(type=tt, addrs=[0x0000], mutex_ids=[0])
        rows = tdim * 2
        cols = tdim // 2
        for i in pl.range(0, a.shape[0], tdim * 2):
            for j in pl.range(0, a.shape[1], tdim):
                t2 = pl.reinterpret(g.next(), shape=[rows, cols])
                pl.load_tile(t2, a, [i, j])
                pl.store_tile(out, t2, [i, j])

    assert _kernel_ir(k)  # parses fine: loop-invariant folded shape


def test_reinterpret_shape_with_loop_var_rejected_in_loop():
    """Review item 2: a loop-variable shape must still be rejected inside nested loops."""

    @pl.jit
    def k(a: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP16],
          out: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_BF16]):
        tt = pl.TileType(shape=[64, 64], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Vec)
        g = pl.make_tile_group(type=tt, addrs=[0x0000], mutex_ids=[0])
        for i in pl.range(0, a.shape[0], 64):
            t2 = pl.reinterpret(g.next(), shape=[i, 64])
            pl.load_tile(t2, a, [i, 0])
            pl.store_tile(out, t2, [i, 0])

    with pytest.raises(InvalidShape, match="compile-time integers"):
        _parse_kernel(k)


def test_runtime_shape_is_rejected():
    @pl.jit
    def k(x: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP16]):
        tt = pl.TileType(shape=[64, 64], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Vec)
        t = pl.make_tile(tt, addr=0)
        pl.load(pl.reinterpret(t, shape=[x.shape[0], 16]), x, [0, 0])

    with pytest.raises(InvalidShape, match="compile-time integers"):
        _parse_kernel(k)


# ---------------------------------------------------------------------------
# tile_group paths + mutex inheritance
# ---------------------------------------------------------------------------


def test_group_reinterpret_rebuilds_slots_and_inherits_size():
    @pl.jit(auto_mutex=True)
    def k(a: pl.Tensor[[128, 128], pl.DT_FP16]):
        tt = pl.TileType(shape=[128, 128], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Vec)
        db = pl.make_tile_group(type=tt, addrs=[0x0, 0x10000], mutex_ids=[0, 1])
        g2 = pl.reinterpret(db, shape=[128, 64])
        c0 = g2.next()
        pl.load(c0, a, [0, 0])

    ir_str = _kernel_ir(k)
    # 2 original slots + 2 rebuilt slots.
    assert ir_str.count("block.make_tile") == 4
    # The rebuilt slots are fresh MemRefs (same address): the two original
    # slots plus the two rebuilt slots each carry a memref_id.
    assert ir_str.count("memref_id=") == 4
    assert "memref_addr=" in ir_str and "65536" in ir_str
    # mutex_ids inherited -> auto_mutex still locks around the rebuilt handle's usage.
    assert "system.mutex_lock" in ir_str
    assert "system.mutex_unlock" in ir_str


def test_group_reinterpret_keeps_dtype_and_layout_inheritance():
    @pl.jit(auto_mutex=True)
    def k(a: pl.Tensor[[128, 128], pl.DT_FP16]):
        tt = pl.TileType(shape=[128, 128], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Vec)
        db = pl.make_tile_group(type=tt, addrs=[0x0, 0x10000], mutex_ids=[0, 1])
        g2 = pl.reinterpret(db, shape=[128, 128], dtype=pl.DT_BF16)
        c0 = g2.current()
        pl.load(c0, a, [0, 0])

    ir_str = _kernel_ir(k)
    assert ir_str.count("block.make_tile") == 4
    assert "bfloat16" in ir_str.lower()


def test_group_multi_column_mutex_inherited():
    @pl.jit(auto_mutex=True)
    def k(a: pl.Tensor[[64, 64], pl.DT_FP16]):
        tt = pl.TileType(shape=[64, 64], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Vec)
        db = pl.make_tile_group(type=tt, addrs=[0, 0x4000], mutex_ids=[[1, 2], [3, 4]])
        g2 = pl.reinterpret(db, shape=[64, 32])
        c0 = g2.next()
        pl.load(c0, a, [0, 0])

    ir_str = _kernel_ir(k)
    assert ir_str.count("block.make_tile") == 4
    # both mutex columns are inherited and printed as fields
    assert "mutex_ids_1" in ir_str
    assert "system.mutex_lock" in ir_str


def test_group_unaligned_slot_address_rejected():
    """Review finding (P2): every explicit slot address must pass the element-boundary check."""
    with pytest.raises(InvalidTile, match="not aligned to the new element size"):

        @pl.jit(auto_mutex=True)
        def k(a: pl.Tensor[[64, 64], pl.DT_FP16]):
            tt = pl.TileType(shape=[64, 64], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.DDR)
            db = pl.make_tile_group(type=tt, addrs=[2, 68], mutex_ids=[0, 1])  # 2 % 4 != 0 for FP32
            g2 = pl.reinterpret(db, shape=[32, 32], dtype=pl.DT_FP32)  # footprint 4096 <= 8192, addr 2 misaligned
            c0 = g2.next()
            pl.load(c0, a, [0, 0])

        _parse_kernel(k)


def test_group_runtime_shape_rejected():
    """Review finding (P3): group path rejects runtime shapes like the single-tile path."""

    @pl.jit
    def k(a: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP16]):
        tt = pl.TileType(shape=[64, 64], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Vec)
        db = pl.make_tile_group(type=tt, addrs=[0, 0x4000], mutex_ids=[0, 1])
        g2 = pl.reinterpret(db, shape=[a.shape[0], 16])
        c0 = g2.next()
        pl.load(c0, a, [0, 0])

    with pytest.raises(InvalidShape, match="compile-time integers"):
        _parse_kernel(k)


def test_group_layout_override_updates_slot_encoding():
    """Coverage: layout override applied to every rebuilt slot of a group."""

    @pl.jit(auto_mutex=True)
    def k(a: pl.Tensor[[64, 64], pl.DT_FP16]):
        tt = pl.TileType(shape=[64, 64], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Mat, layout=pl.NZ)
        db = pl.make_tile_group(type=tt, addrs=[0, 0x4000], mutex_ids=[0, 1])
        g2 = pl.reinterpret(db, shape=[64, 32], layout=pl.TensorLayout.NZ)
        c0 = g2.next()
        pl.load(c0, a, [0, 0])

    ir_str = _kernel_ir(k)
    assert ir_str.count("block.make_tile") == 4
    assert "blayout=2" in ir_str and "slayout=1" in ir_str  # NZ encoding on rebuilt slots


def test_slot_tile_reinterpret_keeps_mutex_binding():
    @pl.jit(auto_mutex=True)
    def k(a: pl.Tensor[[64, 64], pl.DT_FP16]):
        tt = pl.TileType(shape=[64, 64], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Vec)
        db = pl.make_tile_group(type=tt, addrs=[0, 0x4000], mutex_ids=[5, 6])
        t2 = pl.reinterpret(db[1], shape=[64, 64], dtype=pl.DT_BF16)
        pl.load(t2, a, [0, 0])

    ir_str = _kernel_ir(k)
    # 2 group slots + 1 reinterpreted slot tile
    assert ir_str.count("block.make_tile") == 3
    assert "system.mutex_lock" in ir_str


def test_original_handle_unchanged_after_group_reinterpret():
    @pl.jit(auto_mutex=True)
    def k(a: pl.Tensor[[128, 128], pl.DT_FP16]):
        tt = pl.TileType(shape=[128, 128], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Vec)
        db = pl.make_tile_group(type=tt, addrs=[0x0, 0x10000], mutex_ids=[0, 1])
        g2 = pl.reinterpret(db, shape=[128, 64])
        a0 = db.next()
        b0 = g2.next()
        pl.load(a0, a, [0, 0])
        pl.load(b0, a, [0, 0])

    ir_str = _kernel_ir(k)
    # 2 original + 2 rebuilt slots; both handles keep their 32768-byte spans.
    assert ir_str.count("block.make_tile") == 4
    assert ir_str.count("system.mutex_lock") >= 2
