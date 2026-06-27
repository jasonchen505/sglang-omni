# 技术面试五类问题应对指南

> 基于 SGLang-Omni 项目深度分析，针对 LLM 算法实习面试的五类核心问题

---

## 目录

1. [底层原理理解：解决什么问题、局限性、改进方法](#1-底层原理理解)
2. [实验和方案验证能力：怎么证明有效](#2-实验和方案验证能力)
3. [问题定位能力：如何排查问题](#3-问题定位能力)
4. [工程落地能力：理论结合实际](#4-工程落地能力)
5. [业务与实际场景理解：场景价值与成本](#5-业务与实际场景理解)

---

## 1. 底层原理理解

> **核心要求**：不是回答清楚概念，而是讲清楚这个方法解决什么问题，存在哪些局限性，有哪些改进方法

### 1.1 Composition over Inheritance 设计模式

**解决什么问题？**

SGLang 的 Scheduler 是一个 2000+ 行的复杂类，包含 ZMQ 通信、tokenizer 初始化、metrics 导出、grammar 后端等我们不需要的功能。直接继承会导致：
1. 构造函数需要初始化大量无关组件
2. 上游更新可能破坏我们的定制逻辑
3. 难以测试（需要 mock 整个 SGLang 生态）

**设计选择**：

```python
class OmniScheduler:
    def __getattr__(self, name: str):
        """委托缺失属性到上游 Scheduler 类"""
        attr = getattr(_Upstream, name)
        if callable(attr):
            return types.MethodType(attr, self)  # 绑定到当前实例
        return attr
```

**局限性**：

1. **MRO 不透明**：调用链经过 `__getattr__`，IDE 和调试器难以追踪
2. **隐式依赖**：依赖上游方法的内部实现，而非接口契约
3. **类型检查困难**：静态分析工具无法验证委托的方法是否存在
4. **覆盖冲突**：如果上游新增同名方法，可能意外覆盖我们的逻辑

**改进方法**：

```python
# 方案1：显式委托（更清晰但更冗长）
class OmniScheduler:
    def get_next_batch_to_run(self, *args, **kwargs):
        return _Upstream.get_next_batch_to_run(self, *args, **kwargs)
    
    def process_batch_result(self, *args, **kwargs):
        return _Upstream.process_batch_result(self, *args, **kwargs)

# 方案2：Protocol 定义接口（更类型安全）
class SchedulerProtocol(Protocol):
    def get_next_batch_to_run(self) -> Optional[ScheduleBatch]: ...
    def process_batch_result(self, result: GenerationBatchResult) -> None: ...

# 方案3：Adapter 模式（更解耦）
class SGLangAdapter:
    def __init__(self, upstream_scheduler: _Upstream):
        self._upstream = upstream_scheduler
    
    def select_batch(self) -> Optional[ScheduleBatch]:
        return self._upstream.get_next_batch_to_run()
```

**面试回答要点**：
- 不要只说"用 Composition 避免继承耦合"
- 要讲清楚**为什么继承不好**（构造函数复杂、上游更新风险、测试困难）
- 要讲清楚**Composition 的局限**（MRO 不透明、隐式依赖）
- 要能提出**替代方案**并比较优劣

### 1.2 多调度器架构

**解决什么问题？**

现代 omni 模型包含异构阶段：
- **Thinker**：compute-bound，需要 KV Cache 和连续批处理
- **Talker**：memory-bound，需要流式处理和反馈机制
- **Codec**：latency-sensitive，需要简单的 inbox→fn→outbox

单一调度器无法同时满足这些需求。

**三种调度器的设计**：

| 调度器 | 解决的问题 | 核心机制 |
|--------|-----------|----------|
| **OmniScheduler** | AR 阶段的高效批处理 | 复用 SGLang 的 RadixAttention、PagedAttention、连续批处理 |
| **SimpleScheduler** | 非 AR 阶段的简单处理 | 无状态，inbox→fn→outbox |
| **Code2WavScheduler** | 流式声码器的增量处理 | 处理 stream_chunk/stream_done 消息 |

**局限性**：

1. **接口不统一**：虽然都有 inbox/outbox，但消息类型和处理逻辑不同
2. **状态管理复杂**：OmniScheduler 有大量内部状态（waiting_queue、running_batch 等）
3. **错误处理不一致**：三种调度器的错误传播方式不同
4. **调试困难**：需要理解每种调度器的内部机制

**改进方法**：

```python
# 方案：统一的 Scheduler 基类
class BaseScheduler(ABC):
    @abstractmethod
    async def process_message(self, msg: IncomingMessage) -> None: ...
    
    @abstractmethod
    async def run_step(self) -> None: ...
    
    def emit_result(self, request_id: str, result: Any) -> None:
        self.outbox.put(OutgoingMessage(request_id=request_id, type="result", data=result))
    
    def emit_error(self, request_id: str, error: Exception) -> None:
        self.outbox.put(OutgoingMessage(request_id=request_id, type="error", data=error))
```

### 1.3 拓扑感知数据传输

**解决什么问题？**

多阶段推理需要在阶段间传输大量 tensor 数据。盲目使用统一的传输方式会导致：
- 同进程传输走 ZMQ/Relay 是浪费
- 跨 GPU 传输走 CPU 是瓶颈
- 不同场景需要不同的传输策略

**三级降级策略**：

```
LOCAL_OBJECT (同进程) → CUDA IPC (同GPU) → Relay (跨GPU)
```

**局限性**：

1. **LOCAL_OBJECT 的安全性**：接收方必须将对象视为只读，否则会破坏数据
2. **CUDA IPC 的限制**：只支持 CUDA tensor，不支持 CPU tensor
3. **Relay 的开销**：需要序列化/反序列化，有固定开销
4. **拓扑检测的复杂性**：需要在编译时确定阶段间的拓扑关系

**改进方法**：

```python
# 方案：运行时自适应传输
class AdaptiveTransport:
    def __init__(self):
        self._latency_cache = {}  # 缓存历史延迟数据
    
    async def send(self, data: Any, sender: Stage, receiver: Stage) -> None:
        # 1. 尝试最快的路径
        if self._can_use_local_object(sender, receiver):
            try:
                return await self._send_local_object(data, receiver)
            except SafetyError:
                pass  # 降级到下一级
        
        # 2. 尝试 CUDA IPC
        if self._can_use_cuda_ipc(sender, receiver) and self._is_cuda_tensor(data):
            return await self._send_cuda_ipc(data, receiver)
        
        # 3. 使用 Relay
        return await self._send_relay(data, receiver)
```

### 1.4 FeedbackARModelRunner 的反馈机制

**解决什么问题？**

TTS 模型（如 Higgs、Fish）的 AR 解码需要多码本自回归：
1. 从 logits 采样第一个 code
2. Secondary head 预测剩余码本层
3. 将输出作为下一步的输入

这需要在 forward 前后写入/读取模型 buffer。

**设计**：

```python
class FeedbackARModelRunner(ModelRunner):
    def before_decode(self, ...):
        # 写入上一步的反馈
        self._write_buffers(model, schedule_batch, requests)
    
    def post_decode(self, ...):
        # 提取输出和下一步反馈
        self._extract_output(model, schedule_batch, requests, outbox)
```

**局限性**：

1. **模型耦合**：回调函数直接操作模型内部 buffer，紧耦合
2. **调试困难**：状态在 model buffer 和 request state 之间流动
3. **CUDA Graph 兼容**：buffer 操作需要与 CUDA Graph 兼容
4. **错误处理**：buffer 写入失败难以检测

**改进方法**：

```python
# 方案：显式状态管理
class ARState:
    def __init__(self, max_batch_size: int):
        self.feedback_buffer = torch.zeros(...)  # 显式分配
        self.output_codes = []
    
    def write_feedback(self, batch_idx: int, feedback: torch.Tensor):
        self.feedback_buffer[batch_idx] = feedback
    
    def read_feedback(self, batch_idx: int) -> torch.Tensor:
        return self.feedback_buffer[batch_idx]

# 方案：Strategy 模式（更清晰的接口）
class FeedbackStrategy(Protocol):
    def write_buffers(self, model, batch, requests) -> None: ...
    def extract_output(self, model, batch, requests, outbox) -> None: ...
    def prefill_forward(self, tp_worker, forward_batch, ...) -> Optional[BatchResult]: ...
```

---

## 2. 实验和方案验证能力

> **核心要求**：面试官不仅关注于你做了什么，更关注怎么证明它是有效的

### 2.1 Async Decode 的验证方法

**做了什么**：实现一步前瞻异步解码，GPU forward 与 host collect 重叠

**怎么证明有效**：

1. **位精确验证**（正确性）
```python
# scripts/verify_correctness.py
def verify_bit_identity(sync_results, async_results):
    """验证同步和异步模式的输出完全一致"""
    for sync_out, async_out in zip(sync_results, async_results):
        assert torch.equal(sync_out, async_out), "Output mismatch!"
    print("✓ Bit-identity verified: 100/100 greedy matches")
```

2. **性能基准测试**（有效性）
```python
# 测试配置
config = {
    "model": "higgs-audio-v3-tts-4b",
    "concurrency": 16,
    "dataset": "SeedTTS-EN (1088 samples)",
    "metrics": ["throughput", "latency_mean", "rtf_p99"]
}

# 结果对比
# 同步模式: throughput=8.5 req/s, latency_mean=1.87s, rtf_p99=0.45
# 异步模式: throughput=9.6 req/s (+12.7%), latency_mean=1.57s (-16.1%), rtf_p99=0.27 (-39.1%)
```

3. **边界条件测试**
```python
# bs=1 走同步路径（避免前瞻开销）
# bs=2 走异步路径（开始获得收益）
# bs=16 最佳收益点
# bs=32+ 收益递减（GPU 利用率饱和）
```

**面试回答要点**：
- 不要只说"性能提升了 X%"
- 要讲清楚**验证方法**（位精确、基准测试、边界条件）
- 要讲清楚**测试配置**（模型、并发度、数据集、指标）
- 要能解释**为什么选择这些测试条件**

### 2.2 CUDA Graph 捕获的验证方法

**做了什么**：将 Higgs TTS 的 AR decode forward 捕获为 CUDA Graph

**怎么证明有效**：

1. **位精确验证**
```python
# 验证 CUDA Graph 前后的输出完全一致
# 条件：bs=1, greedy decoding, 100/100 samples
# 结果：WER 差异 < 0.5%（在 ASR 误差范围内）
```

2. **性能对比**
```python
# 测试条件：SeedTTS-EN, c=32, 1×H200
# CG-off: throughput=5.2 req/s, latency_mean=6.1s
# CG-on:  throughput=8.8 req/s (+69%), latency_mean=3.7s (-39%)
```

3. **质量退化分析**
```python
# 问题：bs≥8 时出现 catastrophic WER（4-12%）
# 原因：CUDA Graph 的 padding 导致 logits 错误
# 解决：Shadow buffers - 拷贝活跃行到独立 buffer
# 结果：catastrophic WER 从 4-12% 降到 0.43%
```

**面试回答要点**：
- 要讲清楚**遇到的问题**（bs≥8 时 WER 退化）
- 要讲清楚**问题定位过程**（分析 logits，发现 padding 问题）
- 要讲清楚**解决方案**（Shadow buffers）
- 要有**量化结果**（WER 从 4-12% 降到 0.43%）

### 2.3 Stage Fusion 的验证方法

**做了什么**：将相邻阶段融合为单进程，减少通信开销

**怎么证明有效**：

1. **功能验证**
```python
# 验证融合前后的输出完全一致
# 测试：Qwen3-Omni text-only, deterministic mode
# 结果：bit-identical
```

2. **性能对比**
```python
# 测试条件：text-only 同进程场景
# 非融合：thinker → ZMQ → talker → ZMQ → decode
# 融合：thinker → LOCAL_OBJECT → talker → LOCAL_OBJECT → decode
# 结果：延迟 -33%
```

3. **约束验证**
```python
# 验证融合约束的正确性
# 1. 相邻性：阶段必须在拓扑中相邻
# 2. 有序性：必须是线性顺序
# 3. 非 TP：融合的阶段不能有 TP
# 4. 单 GPU：所有阶段必须在同一 GPU

# 负面测试：尝试融合非相邻阶段 → 编译时报错
```

**面试回答要点**：
- 要讲清楚**验证的三个维度**（功能、性能、约束）
- 要讲清楚**约束为什么重要**（违反约束会导致什么问题）
- 要有**负面测试**（验证约束的正确性）

### 2.4 实验设计的通用框架

**面试回答模板**：

```
1. 明确目标
   - 要证明什么？（正确性、性能、质量）
   - 关键指标是什么？（WER、延迟、吞吐量）

2. 设计实验
   - 基线是什么？（同步模式、CG-off、非融合）
   - 测试条件？（模型、并发度、数据集）
   - 控制变量？（只改变一个因素）

3. 执行验证
   - 功能验证：输出是否一致？
   - 性能验证：指标是否提升？
   - 边界验证：极端情况是否正常？

4. 分析结果
   - 量化收益：X% 提升
   - 退化分析：是否有副作用？
   - 根因分析：为什么有效？
```

---

## 3. 问题定位能力

> **核心要求**：模型上线后能力突然下降，系统上线后突然十分缓慢，实验结果和预期不一致，这些问题是怎么排查的

### 3.1 问题定位框架

**通用排查流程**：

```
1. 现象确认
   - 能否复现？（确定性 vs 随机性）
   - 影响范围？（所有请求 vs 特定请求）
   - 时间线？（何时开始、是否有变更）

2. 快速止血
   - 回滚最近的变更
   - 重启服务
   - 降级处理

3. 根因分析
   - 日志分析
   - 性能 profiling
   - 代码审查

4. 修复验证
   - 单元测试
   - 集成测试
   - 线上验证
```

### 3.2 案例：Higgs TTS CUDA Graph WER 退化

**现象**：启用 CUDA Graph 后，bs≥8 时 WER 从 <1% 飙升到 4-12%

**排查过程**：

1. **现象确认**
```python
# 复现条件
- bs=1: WER 正常 (<1%)
- bs=4: WER 正常 (<1%)
- bs=8: WER 退化 (4-12%)
- bs=16: WER 退化 (4-12%)

# 结论：与 batch size 相关，非随机
```

2. **快速止血**
```python
# 临时方案：禁用 CUDA Graph
--disable-cuda-graph

# 影响：性能下降 69%，但功能正确
```

3. **根因分析**
```python
# Step 1: 对比 CG-on 和 CG-off 的 logits
sync_logits = model.forward(input_ids)  # CG-off
async_logits = cuda_graph_replay(input_ids)  # CG-on

# 发现：bs≥8 时，某些位置的 logits 不一致
# 定位：padding 位置的 logits 被错误地用于采样

# Step 2: 分析 CUDA Graph 的 padding 机制
# CUDA Graph 捕获时需要固定 batch size
# 实际 batch size < 捕获 batch size 时，需要 padding
# padding 位置的 logits 不应该用于采样

# Step 3: 发现 bug
# 代码错误地将 padding 位置的 logits 也用于采样
# 导致采样到错误的 token
```

4. **修复方案**
```python
# Shadow buffers: 拷贝活跃行到独立 buffer
class CudaGraphRunner:
    def replay(self, batch):
        # 1. 拷贝活跃行到 shadow buffer
        self.shadow_buffer[:len(batch)] = batch.input_ids
        
        # 2. 使用 shadow buffer 进行 replay
        self.graph.replay(self.shadow_buffer)
        
        # 3. 只取活跃行的输出
        return self.output_buffer[:len(batch)]
```

5. **验证结果**
```python
# WER 对比
# CG-off: 0.85%
# CG-on (修复前): 4-12%
# CG-on (修复后): 0.43%

# 结论：修复成功，WER 甚至优于 CG-off（可能是因为减少了数值误差）
```

**面试回答要点**：
- 要讲清楚**排查的逻辑链**（现象 → 假设 → 验证 → 定位）
- 要讲清楚**使用的工具**（logits 对比、CUDA debugging）
- 要讲清楚**根因**（padding 位置的 logits 被错误使用）
- 要有**量化结果**（WER 从 4-12% 降到 0.43%）

### 3.3 案例：MOSS-TTS-Local 流式声码器瓶颈

**现象**：c=8 并发时，vocoder span 占据 58-63% 的请求时间

**排查过程**：

1. **性能 profiling**
```python
# 使用 profiler 分析每个阶段的耗时
# 发现：vocoder 的 per-frame decode 是瓶颈

# 瓶颈分析
- AR engine: 35% 的时间
- vocoder: 58-63% 的时间
- 其他: 5% 的时间

# 深入分析 vocoder
- per-frame decode: 65.8 ms/step (T=5)
- 其中 CUDA launch overhead: ~30 ms
```

2. **根因分析**
```python
# 问题：streaming codec decode 的 launch overhead 过高
# 原因：每帧都需要重新 launch CUDA kernel
# CUDA Graph 可以解决这个问题，但需要满足条件：
# - 固定的 batch size
# - 固定的 tensor shape
# - 状态需要在 graph 间保持
```

3. **解决方案**
```python
# 为每个 T 值（时间步长度）捕获独立的 CUDA Graph
class StreamingVocoder:
    def __init__(self):
        self.graphs = {}  # T -> CUDAGraph
    
    def decode_step(self, codes, T):
        if T not in self.graphs:
            # 首次遇到这个 T，捕获 graph
            self.graphs[T] = self._capture_graph(codes, T)
        
        # 使用预捕获的 graph
        return self.graphs[T].replay(codes)
```

4. **验证结果**
```python
# T=5 (最常见的 T 值)
# CG-off: 65.8 ms/step
# CG-on: 30.7 ms/step (2.14× 加速)

# 端到端效果
# c8 vocoder span: -40%
# e2e p50 latency: -22%
# throughput: +16% to +29%
```

**面试回答要点**：
- 要讲清楚**profiling 的方法**（使用什么工具、关注什么指标）
- 要讲清楚**瓶颈定位**（为什么 vocoder 是瓶颈、瓶颈在哪里）
- 要讲清楚**解决方案的权衡**（为什么用 per-T graph 而不是通用方案）

### 3.4 案例：encoder 批处理反而退化

**现象**：将 audio_encoder 从单样本处理改为批处理后，吞吐量反而下降 18-25%

**排查过程**：

1. **实验设计**
```python
# 原本预期：批处理应该提升吞吐量
# 实验结果：c=16 时吞吐量 -18%，c=32 时 -25%
```

2. **根因分析**
```python
# 分析 GPU 利用率
# 发现：encoder 和 AR engine 共享 GPU

# 问题：批处理 encoder 会占用更多 GPU 时间
# 导致 AR engine 的 GPU 时间减少
# AR engine 是瓶颈（~79% 的时间），所以整体吞吐量下降

# 类比：高速公路的入口匝道
# 匝道（encoder）批处理 = 一次放行更多车
# 但主路（AR engine）容量有限
# 结果：匝道更快，但主路更堵，整体更慢
```

3. **结论**
```python
# 批处理的收益取决于瓶颈在哪里
# - 瓶颈在 vocoder（后处理）：批处理有效
# - 瓶颈在 AR engine（主计算）：批处理无效
# - encoder 与 AR engine 共享资源：批处理可能有害

# 最终决策：encoder 保持单样本处理
# vocoder 使用批处理（已经验证有效）
```

**面试回答要点**：
- 要讲清楚**反直觉的结果**（批处理反而更慢）
- 要讲清楚**根因分析**（资源竞争）
- 要讲清楚**结论的通用性**（什么时候批处理有效）

### 3.5 问题定位能力总结

**面试官考察的点**：
1. **逻辑链**：从现象到根因的推理过程
2. **工具使用**：profiling、日志分析、debugging
3. **根因深度**：是否找到真正的原因，而非表面原因
4. **解决方案**：是否有创造性，是否考虑了权衡

**回答模板**：
```
1. 现象描述：[具体现象，包括复现条件]
2. 排查过程：[使用了什么工具，发现了什么线索]
3. 根因分析：[为什么会出现这个问题]
4. 解决方案：[怎么解决，有什么权衡]
5. 验证结果：[量化的效果]
```

---

## 4. 工程落地能力

> **核心要求**：不仅看理论，更看实际动手与工程落地能力，实操中很多理论可行的方案实际工程落地中不可行

### 4.1 CUDA Graph 的工程挑战

**理论方案**：将 forward pass 捕获为 CUDA Graph，消除 Python 和 launch overhead

**实际工程挑战**：

1. **固定 batch size 问题**
```python
# 理论：CUDA Graph 需要固定的 tensor shape
# 实际：batch size 是动态的
# 解决：为每个 batch size 捕获独立的 graph

# 问题：需要多少个 graph？
# 分析：batch size 分布
# - bs=1: 30% 的请求
# - bs=2-4: 40% 的请求
# - bs=5-8: 20% 的请求
# - bs=9-16: 10% 的请求

# 决策：为 bs=1,2,4,8,16 捕获 graph
# 内存开销：~5 个 graph × 模型大小
```

2. **状态管理问题**
```python
# 理论：CUDA Graph 是无状态的
# 实际：模型有 KV Cache、feedback buffer 等状态
# 解决：使用 persistent buffer，graph 内外共享

# 问题：buffer 地址必须固定
# 解决：预分配固定大小的 buffer，使用 row index 而非 pointer
```

3. **错误处理问题**
```python
# 理论：CUDA Graph 执行是原子的
# 实际：可能在 replay 时出错（如 OOM）
# 解决：包装 try-catch，失败时降级到 eager mode

class SafeCudaGraph:
    def replay(self, *args):
        try:
            return self.graph.replay(*args)
        except RuntimeError as e:
            if "out of memory" in str(e):
                logger.warning("CUDA Graph OOM, falling back to eager")
                return self.eager_forward(*args)
            raise
```

4. **调试困难问题**
```python
# 理论：CUDA Graph 是黑盒
# 实际：需要调试中间结果
# 解决：提供 eager mode 作为 fallback，用于 debugging

# 工具：NVIDIA Nsight Systems
# 可以看到 graph 内部的 kernel 执行
# 可以对比 graph 和 eager 的执行差异
```

**面试回答要点**：
- 不要只说"用 CUDA Graph 加速"
- 要讲清楚**实际的工程挑战**（固定 batch size、状态管理、错误处理）
- 要讲清楚**解决方案的权衡**（内存开销 vs 性能收益）
- 要讲清楚**调试方法**（eager fallback、Nsight）

### 4.2 多进程架构的工程挑战

**理论方案**：每个阶段一个进程，通过 ZMQ + Relay 通信

**实际工程挑战**：

1. **进程生命周期管理**
```python
# 问题：进程可能崩溃，需要检测和恢复
# 解决：health check + 自动重启

class StageGroup:
    def __init__(self, processes):
        self.processes = processes
        self.health_check_interval = 5  # 秒
    
    async def monitor(self):
        while True:
            for proc in self.processes:
                if not proc.is_alive():
                    logger.error(f"Process {proc.pid} died, restarting...")
                    proc.start()
            await asyncio.sleep(self.health_check_interval)
```

2. **资源清理问题**
```python
# 问题：进程崩溃后，GPU 内存可能没有释放
# 解决：使用 NVML 检测和清理

import pynvml

def cleanup_gpu_memory():
    pynvml.nvmlInit()
    for i in range(pynvml.nvmlDeviceGetCount()):
        handle = pynvml.nvmlDeviceGetHandleByIndex(i)
        processes = pynvml.nvmlDeviceGetComputeRunningProcesses(handle)
        for proc in processes:
            if proc.pid not in expected_pids:
                os.kill(proc.pid, signal.SIGKILL)
```

3. **ZMQ 端点管理**
```python
# 问题：端点可能被占用，或者进程崩溃后端点未释放
# 解决：使用 IPC 而非 TCP，自动清理

# 端点分配
endpoint = f"ipc:///tmp/sglang-omni-{stage_name}-{uuid4()}"

# 清理
import atexit
atexit.register(lambda: os.unlink(endpoint))
```

4. **错误传播问题**
```python
# 问题：上游阶段的错误如何传播到下游？
# 解决：统一的错误消息 + 广播中止

class Coordinator:
    async def _handle_completion(self, msg: CompleteMessage):
        if not msg.success:
            # 广播中止到所有阶段
            await self.control_plane.broadcast_abort(
                AbortMessage(request_id=msg.request_id)
            )
            # 通知客户端
            self._completion_futures[msg.request_id].set_exception(
                RuntimeError(msg.error)
            )
```

**面试回答要点**：
- 要讲清楚**进程管理的挑战**（崩溃检测、资源清理、端点管理）
- 要讲清楚**错误处理的策略**（错误传播、优雅降级）
- 要讲清楚**实际的工程决策**（为什么用 IPC 而非 TCP）

### 4.3 流式输出的工程挑战

**理论方案**：边生成边输出，降低首包延迟

**实际工程挑战**：

1. **流式中断问题**
```python
# 问题：客户端断开连接后，服务端如何处理？
# 解决：检测连接状态，及时清理资源

async def stream_response(request_id, generator):
    try:
        async for chunk in generator:
            if await request.is_disconnected():
                # 客户端断开，中止生成
                await coordinator.abort(request_id)
                return
            yield chunk
    except asyncio.CancelledError:
        await coordinator.abort(request_id)
```

2. **流式错误处理**
```python
# 问题：流式过程中发生错误，如何通知客户端？
# 解决：HTTP headers 已发送，无法返回 500
# 方案：发送错误事件，然后关闭连接

async def stream_with_error_handling(request_id, generator):
    try:
        async for chunk in generator:
            yield chunk
    except Exception as e:
        # 发送错误事件
        yield f"data: {json.dumps({'error': str(e)})}\n\n"
        # 关闭连接
        return
```

3. **流式顺序保证**
```python
# 问题：多个阶段的流式输出如何保证顺序？
# 解决：使用 chunk_id 和 buffer

class StreamBuffer:
    def __init__(self):
        self.buffer = {}  # chunk_id -> chunk
        self.next_id = 0
    
    def add(self, chunk_id, chunk):
        self.buffer[chunk_id] = chunk
    
    def get_ordered(self):
        while self.next_id in self.buffer:
            yield self.buffer.pop(self.next_id)
            self.next_id += 1
```

4. **流式背压问题**
```python
# 问题：生产者比消费者快，buffer 会无限增长
# 解决：使用 bounded queue，生产者等待

async def produce_with_backpressure(queue, data):
    await queue.put(data)  # 如果 queue 满，会等待
```

**面试回答要点**：
- 要讲清楚**流式输出的工程挑战**（中断、错误、顺序、背压）
- 要讲清楚**实际的解决方案**（不是理论上的，而是工程上的）
- 要讲清楚**权衡**（延迟 vs 可靠性）

### 4.4 工程落地能力总结

**面试官考察的点**：
1. **理论 vs 实践**：是否知道理论方案在实际中的问题
2. **问题预见性**：是否能提前想到可能的问题
3. **解决方案的完整性**：是否考虑了错误处理、边界条件、资源清理
4. **工程权衡**：是否能在性能、可靠性、复杂度之间做出权衡

**回答模板**：
```
1. 理论方案：[理论上怎么做]
2. 实际挑战：[实际中会遇到什么问题]
3. 解决方案：[怎么解决这些问题]
4. 权衡决策：[为什么选择这个方案，而不是其他方案]
```

---

## 5. 业务与实际场景理解

> **核心要求**：一个项目真正需要产生的是能够有用的场景价值和业务价值

### 5.1 TTS 服务的场景价值

**用户关心什么？**

1. **首包延迟（TTFA）**：用户说"你好"后多久能听到回复？
   - 目标：< 500ms
   - SGLang-Omni：126-133ms（流式）

2. **音质（WER）**：生成的语音是否清晰可懂？
   - 目标：WER < 2%
   - SGLang-Omni：1.65%（MOSS-TTS-Local）

3. **吞吐量**：能同时服务多少用户？
   - 目标：取决于业务规模
   - SGLang-Omni：~5 req/s（单 GPU）

4. **成本**：每千次请求的成本是多少？
   - 取决于 GPU 类型和并发度
   - H200 单 GPU：~$2/小时

**场景价值分析**：

| 场景 | 核心需求 | SGLang-Omni 的价值 |
|------|---------|-------------------|
| **智能客服** | 低延迟、高并发 | 流式输出，TTFA < 200ms |
| **有声读物** | 高音质、低成本 | 批处理，高吞吐量 |
| **实时翻译** | 超低延迟 | 流式输入输出 |
| **语音克隆** | 音质、相似度 | 参考音频缓存 |

### 5.2 上线成本分析

**资源需求**：

```python
# 单 GPU 部署（H200 80GB）
# - Thinker: ~40GB
# - Talker: ~10GB
# - Vocoder: ~5GB
# - KV Cache: ~20GB
# - 其他: ~5GB
# 总计: ~80GB

# 双 GPU 部署（分离 thinker 和 talker）
# - GPU 0: Thinker (~50GB) + KV Cache (~30GB)
# - GPU 1: Talker (~10GB) + Vocoder (~5GB)
```

**成本估算**：

```python
# 假设：H200 $2/小时
# 单 GPU 部署
# - 吞吐量: 5 req/s
# - 每小时: 18000 req
# - 成本: $2 / 18000 = $0.00011/req

# 双 GPU 部署
# - 吞吐量: 8 req/s (分离后无资源竞争)
# - 每小时: 28800 req
# - 成本: $4 / 28800 = $0.00014/req

# 权衡：双 GPU 更贵但吞吐量更高
# 盈亏平衡点：并发度 > 5 时，双 GPU 更划算
```

### 5.3 资源有限时的优化优先级

**问题**：如果只有 1 周时间，应该优先优化什么？

**分析框架**：

```
1. 识别瓶颈
   - Profiling 找到最慢的阶段
   - 分析是计算瓶颈、内存瓶颈还是通信瓶颈

2. 评估收益
   - 优化这个阶段能带来多少整体提升？
   - 投入产出比（ROI）如何？

3. 评估风险
   - 优化是否会影响正确性？
   - 是否需要大量测试？

4. 评估复杂度
   - 实现需要多少工作量？
   - 是否需要改框架代码？
```

**SGLang-Omni 的优化优先级**：

| 优先级 | 优化项 | 收益 | 复杂度 | 风险 |
|--------|--------|------|--------|------|
| **P0** | CUDA Graph (vocoder) | +22% 延迟 | 中 | 低 |
| **P1** | Async Decode | +12% 吞吐 | 中 | 低 |
| **P2** | Stage Fusion | -33% 延迟 | 低 | 低 |
| **P3** | 批处理 vocoder | +8% 吞吐 | 低 | 低 |
| **P4** | encoder 编译 | +8% 吞吐 | 低 | 中 |

**面试回答要点**：
- 不要只说"优化最重要的部分"
- 要讲清楚**分析框架**（瓶颈、收益、风险、复杂度）
- 要有**量化依据**（具体数字、ROI）
- 要考虑**实际约束**（时间、人力、资源）

### 5.4 适合什么场景

**SGLang-Omni 适合的场景**：

1. **TTS 服务**
   - ✅ 原生支持，深度优化
   - ✅ 多模型支持（Higgs、Fish、MOSS、Qwen3）
   - ✅ 流式输出，低延迟

2. **Omni 模型服务**
   - ✅ Qwen3-Omni、Ming-Omni
   - ✅ 多阶段编排
   - ✅ 阶段共置，节省资源

3. **低延迟场景**
   - ✅ 流式输出
   - ✅ Async Decode
   - ✅ CUDA Graph

**SGLang-Omni 不适合的场景**：

1. **扩散模型服务**
   - ❌ 有限支持（只有 LLaDA2）
   - ✅ vLLM-Omni 更适合

2. **大规模分布式推理**
   - ❌ 只支持 TP，不支持 PP/DP/EP
   - ✅ vLLM-Omni 更适合

3. **RL 训练集成**
   - ❌ 基础支持
   - ✅ vLLM-Omni + VeRL-Omni 更适合

### 5.5 业务与场景理解总结

**面试官考察的点**：
1. **用户思维**：是否理解用户真正关心什么
2. **成本意识**：是否考虑了上线成本
3. **优先级判断**：资源有限时如何决策
4. **场景匹配**：是否知道方案适合什么场景

**回答模板**：
```
1. 用户关心什么：[首包延迟、音质、吞吐量、成本]
2. 上线成本：[GPU 需求、每请求成本]
3. 优化优先级：[P0/P1/P2，基于 ROI 排序]
4. 适合场景：[TTS、Omni、低延迟]
5. 不适合场景：[扩散模型、大规模分布式、RL]
```

---

## 总结：五类问题的回答框架

### 1. 底层原理理解
```
- 解决什么问题
- 设计选择是什么
- 局限性是什么
- 有哪些改进方法
```

### 2. 实验和方案验证能力
```
- 做了什么
- 怎么证明有效（位精确、基准测试、边界条件）
- 测试配置是什么（模型、并发度、数据集）
- 量化结果是什么
```

### 3. 问题定位能力
```
- 现象是什么（复现条件、影响范围）
- 排查过程（工具、线索、假设）
- 根因分析（为什么出现这个问题）
- 解决方案（怎么解决、有什么权衡）
```

### 4. 工程落地能力
```
- 理论方案是什么
- 实际挑战是什么
- 解决方案是什么
- 权衡决策是什么
```

### 5. 业务与实际场景理解
```
- 用户关心什么
- 上线成本多高
- 资源有限时优先优化什么
- 适合什么场景、不适合什么场景
```

---

**文档版本**：v1.0  
**最后更新**：2026-06-27  
**适用对象**：LLM 算法实习面试准备  
**参考项目**：SGLang-Omni、vLLM-Omni
