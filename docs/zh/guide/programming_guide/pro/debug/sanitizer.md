# 内存检测

## 功能说明

Kernel中的内存访问错误（GM越界、tile越界、片上地址重叠、mutex配对错误）、相位使用错误（AccPhase/STPhase与K链结构错误，K链指同一Acc tile上两次store之间的整段K循环写）与sync配对错误（sync_src/sync_dst）通常不会立即导致Host侧异常。其表现多为结果错乱、偶发错误或Kernel挂死，人工排查成本高。内存检测工具（sanitizer）通过在@pypto_pro.language.jit()中配置sanitizer=True开启：框架自动在GM/tile访问点、mutex调用点、相位调用点（AccPhase/STPhase）和sync调用点（sync_src/sync_dst）插桩，每次Kernel启动结束后统一检测并给出结论。

## 使用约束

- 仅供调试使用：插桩与检测会改变Kernel执行开销，性能测量前应在@pypto_pro.language.jit()中移除sanitizer=True。
- 每次启动Kernel会在GM上分配约1GB的日志缓冲，同一位置的重复缺陷会聚合为一条并计数。
- pypto_pro.language.Ptr只携带地址、不携带所指内存的大小，以指针为源的视图无法获知源大小，不做声明shape超出源的检查。
- sync配对检测暂不支持跨核。配对检查在单个执行核（region）内进行，分处不同核的sync_src与sync_dst不会被配对，两侧会各自被判为漏配对并报告。
- 检测覆盖Kernel主体内的访问与声明；SIMT线程块内的所有访问（GM/tile/mutex/相位/sync）均不检测。

## 操作步骤

### 开启内存检测

在核函数装饰器上设置`@pypto_pro.language.jit(sanitizer=True)`：

```python
import pypto_pro.language as pl

@pl.jit(sanitizer=True, auto_mutex=True)
def kernel(x: pl.Tensor[[64, 64], pl.DT_FP16]):
    ...
```

sanitizer默认为False，不开启时，Kernel的编译和执行行为完全不变。开启后，每次启动Kernel，检测自动执行，无问题则静默通过；检出问题时，结果以追加模式写入该kernel编译产物目录下的`pypto_sanitizer_report.txt`（与该次启动的原始记录文件`sanitizer_report.<launch_id>.bin`同目录，路径形如`./build/<kernel_name>__<arch>/tk_none/`）。同一kernel多次启动的结果汇总在同一文件。

### 检测报告说明

报告文件中每段结果以kernel标签行开头。kernel名与launch标识唯一标记该次启动，括号内为该次启动所用kernel的编译产物目录，如`launch 9b3f1c2e`、`(./build/view_over_source_kernel__3510__d0/tk_none)`。同名kernel存在多个编译实例（不同TilingKey、数据类型或设备）时，该目录用于区分检测结果来自哪个实例。该目录下同时有检测结果文件`pypto_sanitizer_report.txt`和该次启动的原始记录文件`sanitizer_report.<launch_id>.bin`（launch标识为该次启动的唯一十六进制UUID，本文示例中缩略）。

每段结果先给出检出数量，再逐条列出检出项，以GM访问越界为例：

```text
===== kernel 'oob_kernel' launch 6e0b31f4 (./build/oob_kernel__3510__d0/tk_none) =====
PyPTO Sanitizer: 1 issue(s) detected
[1/1] GM_OUT_OF_BOUNDS: offsets [32, 0] + valid_shape [64, 64] exceed tensor shape (64, 64) — dim0: offset 32 + valid_shape 64 = 96 > 64 (over by 32) at demo.py:68
      Hint: The tile valid_shape plus the loop offsets must stay inside the tensor logical shape on every dimension
```

每个检出项由类型、消息、源码位置和Hint组成：消息描述违规事实（含访问参数与逐维越界判定），源码位置为检出项对应的源码文件与行号，可直接定位问题语句，Hint为该类问题的修复提示。同一位置的重复缺陷（如循环中每轮迭代都越界）聚合为一条，并以`same defect hit N times at this location`标注次数。

各检测类型及其级别如下，其中ERROR为确定性错误，必须解决；WARNING为可疑问题，需人工确认是否需要处理：

| 检测类型 | 级别 | 检测内容 |
|---|---|---|
| GM_OUT_OF_BOUNDS | ERROR | GM访问越界：offsets + tile的valid_shape超出tensor shape（最多5维，含getval/setval标量访问、make_tensor声明的shape超出源tensor）。 |
| TILE_OUT_OF_BOUNDS | ERROR | tile访问越界：offset + valid_shape超出TileType声明维度（move/insert、set_validshape、tile标量访问）。 |
| TILE_OVERLAP | WARNING | 两个tile在同一内存空间的地址区间重叠。 |
| MUTEX_UNLOCK_BEFORE_LOCK / UNPAIRED_MUTEX_LOCK / MUTEX_DOUBLE_LOCK | ERROR | mutex_lock/mutex_unlock未按相同(region, mutex_id, pipe)严格交替配对（无锁解锁、锁后未解锁、重复加锁）。 |
| PHASE_MISMATCH | ERROR | 相位（AccPhase/STPhase，硬件unit_flag同步）使用错误：matmul_acc/matmul_mx_acc的dst与acc不是同一Tile（原地契约）；同一Acc tile上混用带phase与不带phase的读写（硬件握手与软件同步互不排序）；K链结构错误（链首误用matmul_acc累加到残留值、matmul/matmul_bias出现在链中段覆盖或清空部分和）。 |
| PHASE_DEADLOCK | ERROR | 相位状态机死锁：Final写后无Final读消费（flag悬挂）、带STPhase的读等待从未置位的flag、写等待未清零的flag。 |
| SYNC_UNPAIRED_SRC / SYNC_UNPAIRED_DST | ERROR | sync_src/sync_dst未按相同(region, set_pipe, wait_pipe, event_id)配对（src悬空为数据竞争，dst悬空为死锁）。 |
| SYNC_EVENT_REUSE | ERROR | sync_dst完成等待前，同一event_id被sync_src再次置位（悬空flag被覆盖）。 |

### 关闭内存检测

问题修复后，在@pypto_pro.language.jit()中移除sanitizer=True并恢复生产配置，覆盖全部目标Shape、数据类型、TilingKey和block_dim重新回归，确认问题消失且性能恢复正常。

## 检测示例

### GM访问越界

GM访问越界包含三类：块访问越界、标量访问越界、make_tensor声明shape超出源tensor。

#### 块访问越界

示例说明：行偏移32加tile有效窗口64超出tensor行数64。

```python
@pypto_pro.language.jit(sanitizer=True, auto_mutex=True)
def oob_kernel(x: pypto_pro.language.Tensor[[64, 64], pypto_pro.language.DT_FP16], z: pypto_pro.language.Tensor[[64, 64], pypto_pro.language.DT_FP16]):
    tt = pypto_pro.language.TileType(shape=[64, 64], dtype=pypto_pro.language.DT_FP16, target_memory=pypto_pro.language.MemorySpace.Vec)
    a_group = pypto_pro.language.make_tile_group(type=tt, addrs=0x0000, mutex_ids=[0])
    with pypto_pro.language.section_vector():
        a = a_group.current()
        # 行偏移32：32 + 64 = 96 > 64
        pypto_pro.language.load(a, x, [32, 0])
        pypto_pro.language.store(z, a, [0, 0])
```

```text
===== kernel 'oob_kernel' launch 6e0b31f4 (./build/oob_kernel__3510__d0/tk_none) =====
PyPTO Sanitizer: 1 issue(s) detected
[1/1] GM_OUT_OF_BOUNDS: offsets [32, 0] + valid_shape [64, 64] exceed tensor shape (64, 64) — dim0: offset 32 + valid_shape 64 = 96 > 64 (over by 32) at demo.py:68
      Hint: The tile valid_shape plus the loop offsets must stay inside the tensor logical shape on every dimension
```

字段解读：

- offsets：本次访问在tensor各维上的起始位置（如load的第三个参数）。
- valid_shape：访问所使用的tile有效窗口。

结论：demo.py第68行的load以偏移[32, 0]访问[64, 64]的tensor，行方向32 + 64 = 96超出64共32个元素，可定位为循环偏移未按tensor行数封顶，需要检查循环边界或偏移计算。

#### 标量访问越界

示例说明：setval的线性偏移64超出tensor的64个元素。

```python
@pypto_pro.language.jit(sanitizer=True, auto_mutex=True)
def gm_scalar_oob_kernel(out: pypto_pro.language.Tensor[[64], pypto_pro.language.DT_INT32]):
    with pypto_pro.language.section_vector():
        # 线性偏移64：超出64元素
        pypto_pro.language.setval(out, 64, 1)
```

```text
===== kernel 'gm_scalar_oob_kernel' launch 1c94f7a2 (./build/gm_scalar_oob_kernel__3510__d0/tk_none) =====
PyPTO Sanitizer: 1 issue(s) detected
[1/1] GM_OUT_OF_BOUNDS: scalar access at linear offset 64 exceeds 64 elements at demo.py:17
      Hint: getval/setval offsets must stay inside the tensor element count
```

字段解读：

- linear offset：tensor上的元素序号。

结论：demo.py第17行的setval在64个元素的tensor上写入序号64（最后一个合法序号为63），可定位为标量写入位置越界，需要检查该偏移的上界。

#### make_tensor声明shape超出源tensor

示例说明：make_tensor在[128, 128]的源tensor上声明[200, 200]的视图，共享存储导致声明超出源的容量，声明时即检出。

```python
# 源x为[128, 128]，声明[200, 200]
t2 = pypto_pro.language.make_tensor(x, [200, 200])
```

```text
===== kernel 'view_over_source_kernel' launch 9b3f1c2e (./build/view_over_source_kernel__3510__d0/tk_none) =====
PyPTO Sanitizer: 1 issue(s) detected
[1/1] GM_OUT_OF_BOUNDS: make_tensor view [200, 200] covers 80000 bytes, source tensor [128, 128] holds 32768 bytes at demo.py:1255
      Hint: A view shares its source's storage; keep its shape/stride size within the source
```

字段解读：

- make_tensor view后的方括号：声明shape。
- covers：按shape和stride展开后实际覆盖的字节数。
- holds：源tensor的字节容量。

结论：demo.py第1255行的视图声明[200, 200]（覆盖80000字节）挂在[128, 128]（32768字节）的源tensor上，可定位为声明超出源的容量，需要核对视图shape与源tensor的实际大小。

### tile访问越界

tile访问越界包含四类：move偏移越界、insert偏移越界、动态set_validshape窗口越界、标量访问越界。

#### move偏移越界

示例说明：move以offset [32, 0]访问声明为[64, 64]的tile，offset加窗口超出声明维度。

```python
# tile声明为[64, 64]，offset 32 + 窗口64 = 96 > 64
pypto_pro.language.move(a_l0a, cur_a, offset=[32, 0])
```

```text
===== kernel 'tile_move_offset_oob_kernel' launch 4d2a80be (./build/tile_move_offset_oob_kernel__3510__d0/tk_none) =====
PyPTO Sanitizer: 1 issue(s) detected
[1/1] TILE_OUT_OF_BOUNDS: offsets [32, 0] + valid_shape [64, 64] exceed tile shape [64, 64] — dim0: offset 32 + valid_shape 64 = 96 > 64 (over by 32) at demo.py:227
      Hint: Keep the tile's valid shape within the declared TileType dims
```

字段解读：

- exceed tile shape后的方括号：TileType声明的tile shape（GM访问越界中对应位置为tensor shape）。

结论：demo.py第227行的move以偏移[32, 0]访问声明为[64, 64]的tile，行方向32 + 64 = 96超出64共32个元素，可定位为tile内偏移超出声明维度，需要检查搬运偏移或改用更大的tile。

#### insert偏移越界

示例说明：insert将[64, 64]的src写入声明为[128, 64]的dst的offset [96, 0]处，写入偏移加窗口超出声明维度。

```python
# dst声明为[128, 64]，src窗口[64, 64]，写入偏移96：96 + 64 = 160 > 128
pypto_pro.language.insert(dst, src, offset=[96, 0])
```

```text
===== kernel 'insert_oob_kernel' launch 7f51c39d (./build/insert_oob_kernel__3510__d0/tk_none) =====
PyPTO Sanitizer: 1 issue(s) detected
[1/1] TILE_OUT_OF_BOUNDS: offsets [96, 0] + valid_shape [64, 64] exceed tile shape [128, 64] — dim0: offset 96 + valid_shape 64 = 160 > 128 (over by 32) at demo.py:64
      Hint: Keep the tile's valid shape within the declared TileType dims
```

字段解读：

- exceed tile shape后的方括号：目的tile（dst）声明维度。
- offsets与valid_shape：写入偏移与src的有效窗口，二者之和不得超过dst声明维度。

结论：demo.py第64行的insert将[64, 64]的src写入[128, 64]的dst偏移[96, 0]处，行方向96 + 64 = 160超出128共32个元素，可定位为目的tile声明维度不足或写入偏移过大。

#### 动态set_validshape窗口越界

示例说明：set_validshape设置的动态窗口m运行时为65，超出tile声明维度64，窗口设置语句本身报TILE_OUT_OF_BOUNDS，后续按超限窗口的访问还会连带报GM_OUT_OF_BOUNDS。

```python
# tile声明为[64, 64]，动态窗口m运行时为65（未用min封顶）
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

字段解读：

- [3/3]：窗口设置语句本身的越界（demo.py第45行set_validshape的窗口[65, 64]超出tile声明[64, 64]）。
- [1/3]和[2/3]：后续load/store按超限窗口访问tensor的连带报错。

结论：动态窗口未按tile声明维度封顶，需要用min等手段限制窗口上界。

#### 标量访问越界

示例说明：getval的线性偏移4096超出[64, 64] tile的4096个元素。

```python
# tile声明为[64, 64]（4096元素），线性偏移4096：超出
pypto_pro.language.setval(out, 0, pypto_pro.language.getval(t, 4096))
```

```text
===== kernel 'tile_scalar_oob_kernel' launch 5c0d97b1 (./build/tile_scalar_oob_kernel__3510__d0/tk_none) =====
PyPTO Sanitizer: 1 issue(s) detected
[1/1] TILE_OUT_OF_BOUNDS: scalar tile access at linear offset 4096 exceeds 4096 elements (dims [64,64]) at demo.py:31
      Hint: getval/setval offsets must stay inside the tile element count
```

字段解读：

- dims后的方括号：tile声明维度。

结论：demo.py第31行的getval在[64, 64]（4096个元素）的tile上读取序号4096（最后一个合法序号为4095），可定位为tile标量访问越界，需要检查该偏移的计算。

### tile地址重叠

示例说明：两个tile在UB上的地址区间相交，给出WARNING（可能是刻意的复用安排，也可能是地址规划错误）。

```python
ta_group = pypto_pro.language.make_tile_group(type=tt, addrs=0x0000, mutex_ids=[0])
tb_group = pypto_pro.language.make_tile_group(type=tt, addrs=0x1000, mutex_ids=[1])  # 与ta_group地址相交
ta = ta_group.current()
tb = tb_group.current()
```

```text
===== kernel 'overlap_kernel' launch 3b7e05d8 (./build/overlap_kernel__3510__d0/tk_none) =====
PyPTO Sanitizer: 1 issue(s) detected
[1/1] TILE_OVERLAP: tile range [Vec 0x0, 0x2000) overlaps another tile [Vec 0x1000, 0x3000) at demo.py:124
      Hint: Two tiles must not share on-chip address space
```

字段解读：

- tile range后的方括号：检出项所在tile的内存空间与地址区间（左闭右开）。
- overlaps another tile后的方括号：与之相交的另一个tile的区间。
- 同一内存空间：二者均落在Vec（UB）。

结论：demo.py第124行的两个tile在UB上分别占用[0x0, 0x2000)和[0x1000, 0x3000)，从0x1000起重叠0x1000字节，需要核对两个tile的addr规划或确认是否为有意的复用。

### mutex配对错误

mutex配对错误包含三类：无锁解锁、锁后未解锁、重复加锁。

#### 无锁解锁

示例说明：mutex_unlock未与相同(region, mutex_id, pipe)的mutex_lock配对。

```python
# 只解锁不上锁
pypto_pro.language.system.mutex_unlock(mutex_id=7, pipe=pypto_pro.language.PipeType.MTE2)
```

```text
===== kernel 'mutex_unlock_before_lock_kernel' launch e9120fa4 (./build/mutex_unlock_before_lock_kernel__3510__d0/tk_none) =====
PyPTO Sanitizer: 1 issue(s) detected
[1/1] MUTEX_UNLOCK_BEFORE_LOCK: mutex_unlock(mutex_id=7, pipe=MTE2, region=1) without a preceding mutex_lock at demo.py:705
      Hint: Every mutex_unlock must be paired with a mutex_lock on the same (region, mutex_id, pipe)
```

字段解读：

- region：执行核编号（不同执行核的mutex硬件槽位独立，配对按region区分）。
- mutex_id：锁编号。
- pipe：该锁保护的流水线。

结论：demo.py第705行的mutex_unlock在region 1上解锁mutex_id为7、pipe为MTE2的锁，但此前没有对应的mutex_lock，可定位为解锁缺少配对的加锁，需要补齐lock或删除多余的unlock。

#### 重复加锁

示例说明：同一(region, mutex_id, pipe)在上一次unlock前再次mutex_lock，第二次lock无法获得已被占用的锁槽位，该pipe将永久阻塞（MUTEX_DOUBLE_LOCK）。

```python
# 同一锁连续lock两次，中间未unlock
pypto_pro.language.system.mutex_lock(mutex_id=7, pipe=pypto_pro.language.PipeType.MTE2)
pypto_pro.language.system.mutex_lock(mutex_id=7, pipe=pypto_pro.language.PipeType.MTE2)
```

```text
===== kernel 'mutex_double_lock_kernel' launch 2e58c9a1 (./build/mutex_double_lock_kernel__3510__d0/tk_none) =====
PyPTO Sanitizer: 1 issue(s) detected
[1/1] MUTEX_DOUBLE_LOCK: mutex_lock(mutex_id=7, pipe=MTE2, region=0) executes before the previous lock was unlocked and will block this pipe permanently at demo.py:15
      Hint: A mutex key must alternate lock and unlock; do not lock the same (region, mutex_id, pipe) twice
```

字段解读：

- executes before the previous lock was unlocked：同一配对键上第二次lock出现时，前一次lock尚未unlock。
- will block this pipe permanently：锁槽位已被占用，第二次lock等待不到释放，该pipe上的后续指令无法执行。

结论：demo.py第15行对mutex_id为7、pipe为MTE2的锁重复加锁，需要删除多余的lock，或在其前插入配对的unlock。

此外，mutex_lock执行后直到kernel结束都没有等到配对的mutex_unlock时，报告UNPAIRED_MUTEX_LOCK，报告位置锚定在悬空的mutex_lock语句。未释放的锁槽位会使同键的后续lock无法获得，与重复加锁一样最终表现为该pipe阻塞。

### 相位（AccPhase/STPhase）错误

相位检测针对matmul的AccPhase与store/move/insert的STPhase（硬件unit_flag同步）。检测包含两类：PHASE_MISMATCH（原地契约违反、同步机制混用、K链结构错误）与PHASE_DEADLOCK（flag状态机死锁）。原地契约指matmul_acc/matmul_mx_acc的dst与acc必须为同一Tile，即原地累加。以下示例均为Cube段matmul场景，未覆盖原地契约违反子类。

#### matmul带相位、store漏配相位

示例说明：matmul已用AccPhase.Final声明硬件unit_flag握手，消费它的store未配STPhase，两套同步机制混用。Fixpipe可能在matmul完成前读到数据（PHASE_MISMATCH）。

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
        # 违规点：matmul带phase，消费它的store未配STPhase
        pypto_pro.language.store(out, ac, [0, 0])
```

```text
===== kernel 'phase_mixed_sync_kernel' launch 3e7f1a2b (./build/phase_mixed_sync_kernel__3510__d0/tk_none) =====
PyPTO Sanitizer: 1 issue(s) detected
[1/1] PHASE_MISMATCH: region 0, Acc tile 0x0: matmul uses phase (hardware unit_flag sync) while a store/move on the same block omits STPhase (software sync) -- the two mechanisms mix and Fixpipe may read data before the matmul finishes at demo.py:82
      Hint: When any matmul of a chain uses phase, the consuming store/move must use STPhase as well
```

字段解读：

- region / Acc tile：相位状态按（执行核，Acc tile地址）独立跟踪，报告锚定到混用发生的语句。
- 两套同步机制：带phase的matmul写由硬件unit_flag握手排序，不带STPhase的store读由软件同步排序，二者之间不构成顺序关系。

结论：demo.py第82行的store消费了带AccPhase.Final的matmul结果，但自身未带STPhase.Final，Fixpipe可能在matmul完成前读走数据。需要为store补上phase=STPhase.Final，或两侧同时去掉phase。

#### Final写后无读消费

示例说明：K链末块用AccPhase.Final完成写后，没有带STPhase.Final的store/move去消费，kernel结束时该Acc tile的flag仍处于置位状态。本次启动能跑完，但同一Acc tile的下一次写会等待flag清零而死锁（PHASE_DEADLOCK）。

```python
    # tile声明与装载同上例
    with pypto_pro.language.section_cube():
        ...
        # 违规点：Final写之后漏store，该次Final无人消费
        pypto_pro.language.matmul(ac, al, br, phase=pypto_pro.language.AccPhase.Final)
```

```text
===== kernel 'phase_dangling_final_kernel' launch 7c21d0e8 (./build/phase_dangling_final_kernel__3510__d0/tk_none) =====
PyPTO Sanitizer: 1 issue(s) detected
[1/1] PHASE_DEADLOCK: region 0, Acc tile 0x0 ends with the unit_flag still set: the Final write has no Final read to consume it at demo.py:95
      Hint: A Final write must be followed by a Final read (store/move) in the same chain
```

字段解读：

- ends with the unit_flag still set：回放到kernel结束时，该（region, Acc tile）的unit_flag仍由Final写置位，没有Final读清零。
- 报告位置：锚定在置位flag的那个Final写语句。

结论：demo.py第95行的matmul以AccPhase.Final写Acc tile后直到kernel结束都没有Final读消费，flag悬挂。需要补上漏掉的store，或在无需搬出时去掉phase改用软件同步。

#### K链首块误用matmul_acc

示例说明：按M维切块复用同一Acc tile时，第二轮循环的链首块误用matmul_acc（复制中段代码忘改）。部分和累加到上一轮残留值上，结果静默错误（PHASE_MISMATCH）。

```python
    # tile声明与装载同上例，a为[128, 128]，out为[128, 64]
    with pypto_pro.language.section_cube():
        ac = acc_g.current()
        for i in pypto_pro.language.range(0, 128, 64):
            for k in pypto_pro.language.range(0, 128, 64):
                # al/br：按[i, k]装载本轮操作数（同上例）
                if k == 0:
                    if i == 0:
                        pypto_pro.language.matmul(ac, al, br, phase=pypto_pro.language.AccPhase.Partial)
                    else:
                        # 违规点：第二轮链首误用matmul_acc，累加到上一轮残留的部分和
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

字段解读：

- K链：同一Acc tile上两次store（Final读）之间的整段K循环写；链首必须用覆盖写（matmul或matmul_bias）建立基值，中段及末块才用matmul_acc。
- is the first write of a chain：第二轮i循环在新链的首个写是matmul_acc，未先覆盖，直接累加到上一轮残留的部分和上。

结论：demo.py第118行的matmul_acc是新K链的首个写，但其累加语义把上一轮残留值当成了基值，输出静默偏大，链首须改回matmul（或matmul_bias）覆盖写。对称地，matmul出现在链中段会覆盖丢弃已有部分和；matmul_bias出现在链中段会执行L0C Buffer初始化，清空已有部分和，同属此类K链结构错误。

### sync配对错误

sync检测覆盖sync_src/sync_dst的配对错误。sync_src与sync_dst的region、set_pipe、wait_pipe、event_id四个参数完全相同才构成一对。配对错误包含三类：SYNC_UNPAIRED_SRC（src发出后无配对dst等待，数据竞争）、SYNC_UNPAIRED_DST（dst要等的配对src不存在，死锁）、SYNC_EVENT_REUSE（dst完成等待前同一event_id被再次置位）。

#### 漏写sync_dst

示例说明：MTE2搬运后只发了sync_src，漏写配对的sync_dst，V侧计算未与搬运构成顺序，数据竞争（SYNC_UNPAIRED_SRC）。

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
        # 违规点：漏写sync_dst(set_pipe=MTE2, wait_pipe=V, event_id=0)
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

字段解读：

- region / set_pipe / wait_pipe / event_id：配对键四要素，即执行核、置位流水线、等待流水线与事件编号，四者完全相同才构成一对。
- data race：无人等待该flag，V侧add/store与MTE2搬运之间没有顺序保证。

结论：demo.py第217行的sync_src置位event 0后没有sync_dst等待，需要紧随其后补上sync_dst(set_pipe=MTE2, wait_pipe=V, event_id=0)，保证add执行时搬运已完成。

与漏写sync_dst相对称，出现sync_dst但它前面没有任何一个配对的sync_src时，报告SYNC_UNPAIRED_DST。它要等的flag从未被置位，这次等待永远不会完成（死锁）。

#### event未消费就复用

示例说明：同一event_id的sync_src在配对的sync_dst完成等待前再次置位。

```python
# 双缓冲搬运复用event 0，但第二轮置位发生在sync_dst等待之前
for i in pypto_pro.language.range(0, 2):
    pypto_pro.language.load(ta, a, [i * 64, 0])
    # 违规点：i=1时的sync_src在sync_dst等待event 0之前再次置位
    pypto_pro.language.system.sync_src(set_pipe=pypto_pro.language.PipeType.MTE2, wait_pipe=pypto_pro.language.PipeType.V, event_id=0)
pypto_pro.language.system.sync_dst(set_pipe=pypto_pro.language.PipeType.MTE2, wait_pipe=pypto_pro.language.PipeType.V, event_id=0)
```

```text
===== kernel 'sync_event_reuse_kernel' launch 6a3b5c7e (./build/sync_event_reuse_kernel__3510__d0/tk_none) =====
PyPTO Sanitizer: 1 issue(s) detected
[1/1] SYNC_EVENT_REUSE: sync_src on region 0, MTE2 -> V, event_id 0 fires again before the previous sync_dst consumed the event (the pending flag is overwritten) at demo.py:232
      Hint: The same event_id may only be reused after its sync_dst has completed the wait
```

字段解读：

- fires again before the previous sync_dst consumed the event：第二次sync_src出现时，上一次置位的event 0还没有任何sync_dst等待过。
- the pending flag is overwritten：硬件flag只有一个比特，第二次置位盖掉第一次，等待方无法区分自己等到的是哪一次。

结论：demo.py第232行（i=1轮）的sync_src在sync_dst等待event 0之前再次置位同一事件；复用同一event_id前，必须先让配对的sync_dst完成等待，或为新一轮搬运换用新的event_id。
