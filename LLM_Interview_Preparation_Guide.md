# LLM 算法实习面试准备指南

> 基于 SGLang-Omni 项目深度分析，结合 vLLM-Omni 对比，为 LLM & Agent 应用及后训练方向的面试准备

---

## 目录

1. [项目概述与核心定位](#1-项目概述与核心定位)
2. [SGLang-Omni 核心架构深度解析](#2-sglang-omni-核心架构深度解析)
3. [SGLang-Omni vs vLLM-Omni 对比分析](#3-sglang-omni-vs-vllm-omni-对比分析)
4. [关键技术深挖点](#4-关键技术深挖点)
5. [系统设计面试题](#5-系统设计面试题)
6. [算法相关深挖点](#6-算法相关深挖点)
7. [面试常见问题与回答要点](#7-面试常见问题与回答要点)
8. [自我介绍与项目经验展示](#8-自我介绍与项目经验展示)
9. [扩展知识点与论文推荐](#9-扩展知识点与论文推荐)

---

## 1. 项目概述与核心定位

### 1.1 SGLang-Omni 项目定位

**SGLang-Omni** 是基于 [SGLang](https://github.com/sgl-project/sglang) 构建的**高性能全模态模型推理与服务框架**，专注于：

- **多阶段流水线编排**：将复杂多模态模型分解为独立阶段，每个阶段有独立调度器
- **计算中心化设计**：每个阶段根据其计算瓶颈（compute-bound / memory-bound / latency-sensitive）独立调优
- **零拷贝共享内存通信**：阶段间通过 inbox/outbox 抽象和零拷贝共享内存传输 tensor

### 1.2 核心技术亮点（面试亮点）

| 技术亮点 | 面试价值 | 关键点 |
|---------|---------|--------|
| **Composition over Inheritance** | 设计模式理解 | 通过 `__getattr__` 委托复用 SGLang Scheduler，避免继承耦合 |
| **多调度器架构** | 系统设计能力 | OmniScheduler (AR) + SimpleScheduler (非AR) + Code2WavScheduler (流式) |
| **拓扑感知数据传输** | 性能优化思维 | LOCAL_OBJECT (同进程) → CUDA IPC (同GPU) → Relay (跨GPU) 三级降级 |
| **Stage Fusion** | 资源管理理解 | 相邻阶段可融合为单进程，减少通信开销 |
| **Async Decode** | 异步编程能力 | 一步前瞻异步解码，GPU forward 与 host collect 重叠 |

### 1.3 支持模型

| 模型 | 类型 | 管道阶段数 | 特点 |
|------|------|-----------|------|
| Qwen3-Omni | Omni | 8 (语音) / 6 (文本) | 文本+图像+音频+视频 → 文本+音频 |
| Fish Audio S2-Pro | TTS | 3 | 语音克隆，流式输出 |
| Higgs Audio v3 | TTS | 3 | 多码本 AR，102 种语言 |
| Qwen3-TTS / Voxtral | TTS | 3 | 语音克隆，多语言 |
| MOSS-TTS | TTS | 3 | 延迟模式，31 种语言 |
| Ming-Omni | Omni | 3+ | 100B MoE，视频+音频输出 |
| LLaDA2.0-Uni | 多模态 | 4 | 扩散 LLM，并行去码 |

---

## 2. SGLang-Omni 核心架构深度解析

### 2.1 四层架构设计

```
┌─────────────────────────────────────────────────────────────────┐
│                    Serving Layer (HTTP/WebSocket)                 │
│   /v1/chat/completions  /v1/audio/speech  /v1/realtime          │
└───────────────────────────┬─────────────────────────────────────┘
                            │
┌───────────────────────────▼─────────────────────────────────────┐
│                    Client Layer (Endpoint Adapter)                │
│   将 HTTP/WS 请求转换为 Coordinator 调用                          │
└───────────────────────────┬─────────────────────────────────────┘
                            │
┌───────────────────────────▼─────────────────────────────────────┐
│                    Coordinator Layer (Lifecycle Manager)          │
│   请求路由、多终端合并、广播中止、状态追踪                          │
└───────────────────────────┬─────────────────────────────────────┘
                            │ ZMQ (控制面) + Relay (数据面)
┌───────────────────────────▼─────────────────────────────────────┐
│                    Pipeline Layer (Stage × N)                     │
│   ┌─────────┐    ┌─────────┐    ┌─────────┐    ┌─────────┐     │
│   │ Stage A │───▶│ Stage B │───▶│ Stage C │───▶│ Stage D │     │
│   │Scheduler│    │Scheduler│    │Scheduler│    │Scheduler│     │
│   │ Runner  │    │ Runner  │    │ Runner  │    │ Runner  │     │
│   └─────────┘    └─────────┘    └─────────┘    └─────────┘     │
└─────────────────────────────────────────────────────────────────┘
```

### 2.2 核心组件职责

| 组件 | 职责 | 模型感知? |
|------|------|----------|
| **Coordinator** | 请求生命周期、路由、多终端合并、广播中止 | 否 |
| **Stage** | IO Shell — ZMQ 控制面、Relay 数据面、流路由 | 否 |
| **Scheduler** | 批次选择、KV Cache 管理、计算调度 | 部分 |
| **ModelRunner** | 前向传播、采样、模型特定 hook | 是 |

### 2.3 调度器体系（核心设计）

**三种调度器，统一接口**：

```python
# 统一接口：inbox, outbox, start(), stop(), abort()
class Scheduler(Protocol):
    inbox: Queue[IncomingMessage]
    outbox: Queue[OutgoingMessage]
    def start(self) -> None: ...
    def stop(self) -> None: ...
    def abort(self, request_id: str) -> None: ...
```

| 调度器 | 用途 | 特点 |
|--------|------|------|
| **OmniScheduler** | AR 阶段 (Thinker, Talker) | 复用 SGLang 的 KV Cache、连续批处理、RadixAttention |
| **SimpleScheduler** | 非 AR 阶段 (预处理、编码器) | 无 KV Cache，简单 inbox→fn→outbox |
| **Code2WavScheduler** | 流式声码器 | 处理 stream_chunk/stream_done 消息 |

### 2.4 OmniScheduler 的 Composition 设计（高频考点）

**面试问题**：为什么不直接继承 SGLang Scheduler，而要用 Composition？

**深挖点 1：`__getattr__` 委托机制**

```python
class OmniScheduler:
    def __getattr__(self, name: str):
        """Look up methods on the upstream SGLang Scheduler class.
        
        This gives us access to the full scheduling MRO (batch selection,
        result processing, memory checks, etc.) without inheriting.
        """
        try:
            attr = getattr(_Upstream, name)
        except AttributeError:
            raise AttributeError(
                f"'{type(self).__name__}' has no attribute {name!r}"
            ) from None

        # Bind unbound methods to this instance so they use our state
        if callable(attr):
            return types.MethodType(attr, self)
        return attr
```

**设计优势**：
1. **避免继承耦合**：不侵入 SGLang Scheduler 的 MRO
2. **选择性复用**：只复用 `get_next_batch_to_run()`、`process_batch_result()` 等核心方法
3. **独立演化**：SGLang 上游更新不会破坏 OmniScheduler
4. **显式覆盖**：需要定制的方法（如 `recv_requests`、`run_batch`）直接定义在本类

**深挖点 2：Composition 边界规则**

```python
# 三条规则控制 Composition 边界
# 1. Pin, don't track — 固定 SGLang 版本，不追踪 main
# 2. Minimize reuse surface — 只用公开方法，不读写内部属性
# 3. Upstream-first, when affordable — 需要新功能优先提 PR 到上游
```

**深挖点 3：覆盖的方法**

| 方法 | 原因 |
|------|------|
| `recv_requests()` | 从 inbox 队列取消息，而非 ZMQ |
| `process_input_requests()` | 使用 request_builder 转换 StagePayload |
| `run_batch()` | 委托给 ModelRunner，而非直接调 tp_worker |
| `stream_output()` | 将结果放入 outbox，而非发送到 detokenizer |
| `send_to_tokenizer()` | No-op，结果通过 stage outbox 路由 |

### 2.5 数据面与控制面分离（架构亮点）

```
控制面 (ZMQ): 小消息 — Submit, DataReady, Abort, Shutdown
    ↓
数据面 (Relay): 大 tensor — SHM / NCCL / NixL / Mooncake
    ↓
快速路径: LOCAL_OBJECT (同进程) / CUDA IPC (同GPU)
```

**为什么不用 Ray？**

> Thinker→Talker 是生产者-消费者模式，结构上与 RL 训练相同。但 Ray 的额外运行时、对象存储语义和运维开销对我们的场景（少量固定阶段、少量 GPU、无自动扩缩容）过于沉重。ZMQ + Relay 以更少的基础设施覆盖了我们的拓扑需求。

---

## 3. SGLang-Omni vs vLLM-Omni 对比分析

### 3.1 架构设计对比

| 维度 | SGLang-Omni | vLLM-Omni |
|------|-------------|-----------|
| **基座框架** | SGLang | vLLM |
| **架构模式** | Composition (委托) | Mixin (混入) |
| **调度器设计** | 多调度器 (Omni/Simple/Code2Wav) | 统一 Orchestrator + StagePool |
| **阶段间通信** | ZMQ (控制) + Relay (数据) | ZMQ (控制) + ZMQ/IPC (数据) |
| **KV Cache 管理** | 复用 SGLang RadixAttention | 复用 vLLM PagedAttention |
| **进程模型** | MultiProcessRunner (每阶段独立进程) | Orchestrator + Worker (进程池) |
| **配置方式** | 声明式 PipelineConfig | 注册式 OMNI_MODELS + OMNI_PIPELINES |

### 3.2 核心设计哲学对比

**SGLang-Omni 的"计算中心化"设计**：

> 现代 omni 模型分解为具有根本不同计算特征的异构阶段：compute-bound thinker、memory-bound talker、latency-sensitive codec。每个阶段运行自己的独立调度器，根据其瓶颈调优。

**vLLM-Omni 的"完全解耦"设计**：

> 将复杂多模态推理分解为独立阶段，通过 StagePool 统一编排。每个阶段是逻辑副本，支持负载均衡和亲和性。

### 3.3 技术实现对比

| 技术点 | SGLang-Omni | vLLM-Omni |
|--------|-------------|-----------|
| **KV Cache 传输** | 通过 Relay 直接传输 | OmniKVTransferManager + 异构 TP 路由 |
| **流式优化** | Async Decode (一步前瞻) | Async Chunk 流水线 |
| **步缓存** | 无 (专注 AR 模型) | TeaCache (扩散模型) |
| **并行策略** | TP + 同 GPU 共置 | TP/SP/PP/DP/CFG/EP 六维并行 |
| **量化支持** | 通过 SGLang 上游 | Per-Component 量化 (GGUF/Int8/MXFP4) |
| **CUDA Graph** | 模型级捕获 (Higgs, MOSS) | 引擎级 CudaGraphRunner |

### 3.4 性能特点对比

| 指标 | SGLang-Omni | vLLM-Omni |
|------|-------------|-----------|
| **优化重点** | AR TTS 延迟优化 | 首包延迟 (TTFP) 优化 |
| **典型优化** | Async Decode: +12.7% 吞吐，-16.1% 延迟 | Async Chunk: TTFP -99.7% |
| **CUDA Graph** | 模型级捕获，位精确 | 引擎级捕获 |
| **内存管理** | 阶段共置 + mem_fraction_static | Per-stage memory budgeting |

### 3.5 适用场景对比

| 场景 | SGLang-Omni | vLLM-Omni |
|------|-------------|-----------|
| **TTS 服务** | ✅ 原生支持，深度优化 | ✅ 支持，通用方案 |
| **Omni 模型** | ✅ 支持，阶段可共置 | ✅ 支持，完全解耦 |
| **扩散模型** | ⚠️ 有限支持 (LLaDA2) | ✅ 深度支持 (DiT/Flow-Matching) |
| **大规模部署** | ✅ Router 支持多副本 | ✅ 原生分布式 |
| **RL 集成** | ⚠️ 基础支持 | ✅ VeRL-Omni 集成 |

---

## 4. 关键技术深挖点

### 4.1 Stage Fusion（阶段融合）

**面试问题**：如何优化多阶段推理的通信开销？

**核心思想**：将相邻的逻辑阶段融合为单个运行时进程

```python
# 声明式配置
stages = [
    StageConfig(name="thinker", factory="...", gpu=0),
    StageConfig(name="talker_ar", factory="...", gpu=0),
    StageConfig(name="code2wav", factory="...", gpu=0),
]

# 融合配置
fused_stages = [["thinker", "talker_ar", "code2wav"]]
```

**融合约束**：
- 相邻性：阶段必须在拓扑中相邻
- 有序性：必须是线性顺序
- 非 TP：融合的阶段不能有 TP
- 单 GPU：所有阶段必须在同一 GPU

**性能收益**：
- 文本-only 同进程场景：延迟 -33%
- 减少 ZMQ/Relay 开销

### 4.2 Async Decode（异步解码）

**面试问题**：如何优化 AR 模型的解码延迟？

**核心思想**：一步前瞻 (one-step lookahead)，GPU forward 与 host collect 重叠

```
时间 ─────────────────────────────────────────────▶

同步模式:
Step N:  [==GPU forward==][==host collect==]
Step N+1:                                    [==GPU forward==][==host collect==]

异步模式:
Step N:  [==GPU forward==]
                      [==host collect==]
Step N-1:             [==GPU forward==]
                                    [==host collect (N-1)==]
Step N+1:                           [==GPU forward==]
```

**实现机制**：

```python
class OmniScheduler:
    def _event_loop_async_decode(self):
        """一步前瞻异步解码循环"""
        while self._running:
            # 1. 获取下一个批次
            batch = self.get_next_batch_to_run()
            
            # 2. Launch: 启动 GPU forward，不等待
            sched_output, pending_step = self._run_batch_launch(batch)
            
            # 3. 同时处理上一步的结果
            if self._async_pending:
                prev_result = self._run_batch_resolve(
                    *self._async_pending
                )
                self.process_batch_result(prev_result)
            
            # 4. 保存当前步为 pending
            self._async_pending = (batch, sched_output, pending_step)
```

**性能收益**：
- Higgs TTS: +12.7% 吞吐，-16.1% 延迟，-39.1% RTF p99
- 位精确验证：bs=1 和 bs=4 下 100/100 贪心解码一致

### 4.3 拓扑感知数据传输

**面试问题**：多阶段推理中，如何选择最优的数据传输方式？

**三级降级策略**：

```python
# 1. LOCAL_OBJECT (同进程最快)
if same_process and is_safe_edge:
    # 直接传递 Python 对象引用，零拷贝
    target_stage.receive_local_object(payload)

# 2. CUDA IPC (同 GPU 次快)
elif same_gpu and is_cuda_tensor:
    # 跨进程 CUDA IPC，不经过 relay
    cuda_ipc_send(tensor, target_rank)

# 3. Relay (跨 GPU 通用)
else:
    # 通过 relay 后端传输 (SHM/NCCL/NixL/Mooncake)
    relay.write_payload(payload, target_stage)
```

**LOCAL_OBJECT 的安全约束**：
- 接收方必须将对象视为只读
- 投影扇出时，每个投影必须返回独立的 data 容器
- 不可用于流式 chunk（需要 CUDA IPC 或 Relay）

### 4.4 FeedbackARModelRunner（反馈 AR 模型运行器）

**面试问题**：如何实现带反馈的 AR 模型（如 TTS 的多码本解码）？

**核心设计**：三个回调函数 + 一个模型

```python
class FeedbackARModelRunner(ModelRunner):
    def __init__(self, tp_worker, output_processor, outbox, *,
                 write_buffers_fn, extract_output_fn, prefill_forward_fn=None):
        self._write_buffers = write_buffers_fn
        self._extract_output = extract_output_fn
        self._prefill_forward = prefill_forward_fn

    def before_decode(self, forward_batch, schedule_batch, requests):
        """在 decode 前写入上一步的反馈"""
        if schedule_batch.is_decode:
            self._write_buffers(self.model, schedule_batch, requests)

    def custom_prefill_forward(self, forward_batch, schedule_batch, requests):
        """自定义 prefill forward（可选）"""
        if self._prefill_forward and schedule_batch.is_prefill:
            return self._prefill_forward(self.tp_worker, forward_batch, ...)
        return None

    def post_decode(self, batch_result, schedule_batch, requests):
        """在 decode 后提取输出和反馈"""
        self._extract_output(self.model, schedule_batch, requests, self.outbox)
```

**模型内部的一步 decode**：

```
1. 从 model buffer 读取上一步反馈 (由 FeedbackARModelRunner 写入)
2. AR backbone → hidden states → logits
3. 从 logits 采样第一个 code
4. Secondary head 自回归预测剩余码本层
5. 存储组合输出到 buffer，供下一步使用
6. 输出：多层 codes + feedback
```

### 4.5 同 GPU 共置（Colocation）

**面试问题**：如何在单 GPU 上运行多阶段模型？

**核心思想**：per-stage memory budgeting

```python
# 配置示例
config = Qwen3OmniSpeechColocatedPipelineConfig(
    total_gpu_memory_fraction=0.95,  # 总 GPU 内存预算
    stages={
        "thinker": {"mem_fraction_static": 0.7},
        "talker_ar": {"mem_fraction_static": 0.2},
        "code2wav": {"mem_fraction_static": 0.05},
    }
)
```

**KV Cache 头空间计算**：

```python
available_kv_bytes = total_gpu_memory_bytes * fraction - accounted_stage_memory_bytes
```

**内存语义澄清**：
- vLLM 的 `gpu_memory_utilization`：总 VRAM 的分数
- SGLang 的 `mem_fraction_static`：权重加载后剩余 VRAM 的分数

---

## 5. 系统设计面试题

### 5.1 设计一个 TTS 模型服务系统

**面试问题**：请设计一个支持语音克隆、流式输出的 TTS 服务系统。

**回答框架**：

1. **需求分析**
   - 输入：文本 + 参考音频（可选）
   - 输出：流式音频 PCM / SSE
   - 性能：低首包延迟，高吞吐
   - 功能：语音克隆、多语言、流式

2. **架构设计**
   ```
   ┌─────────────────┐
   │   HTTP API      │  /v1/audio/speech
   └────────┬────────┘
            │
   ┌────────▼────────┐
   │   Coordinator   │  请求生命周期管理
   └────────┬────────┘
            │
   ┌────────▼────────┐    ┌──────────────┐    ┌──────────────┐
   │  Preprocessing  │───▶│  TTS Engine  │───▶│   Vocoder    │
   │  (SimpleSched)  │    │ (OmniSched)  │    │ (Code2Wav)   │
   └─────────────────┘    └──────────────┘    └──────────────┘
   ```

3. **关键设计点**
   - **流式输出**：thinker 边生成 token 边传给 vocoder
   - **语音克隆**：参考音频经 codec 编码后作为 prompt
   - **CUDA Graph**：AR decode 步骤捕获为 CUDA Graph
   - **Async Decode**：GPU forward 与 host collect 重叠

4. **扩展讨论**
   - 如何支持多模型？→ 声明式 PipelineConfig
   - 如何水平扩展？→ Router + 多副本
   - 如何保证音质？→ 位精确 CUDA Graph

### 5.2 设计一个支持反馈的 AR 推理引擎

**面试问题**：如何设计一个支持多码本 TTS 的 AR 推理引擎？

**回答要点**：

1. **反馈机制设计**
   - 模型内部维护 feedback buffer
   - ModelRunner 在 forward 前写入上一步反馈
   - forward 后提取输出和下一步反馈

2. **调度器设计**
   - 复用 SGLang 的 KV Cache 和连续批处理
   - 覆盖 run_batch 委托给自定义 ModelRunner
   - 支持流式输出到下游阶段

3. **CUDA Graph 捕获**
   - 捕获 AR decode forward
   - 处理变长批次（padding）
   - 位精确验证

4. **性能优化**
   - Async Decode：一步前瞻
   - 批量 vocoder decode
   - torch.compile 编码器

### 5.3 设计一个阶段融合系统

**面试问题**：如何设计一个支持阶段融合的多阶段推理系统？

**回答要点**：

1. **融合约束**
   - 相邻性、有序性、线性、非 TP、单 GPU

2. **实现机制**
   - 编译时验证融合约束
   - 运行时进程拓扑规划器合并进程组
   - 融合阶段复用 same-process local dispatch

3. **配置方式**
   ```python
   PipelineConfig(
       stages=[...],
       fused_stages=[["thinker", "talker_ar"]]
   )
   ```

4. **性能收益**
   - 减少 ZMQ/Relay 开销
   - 同进程场景延迟 -33%

---

## 6. 算法相关深挖点

### 6.1 自回归 vs 非自回归 vs 扩散模型

**面试问题**：请比较自回归、非自回归和扩散模型的生成方式。

| 特性 | 自回归 (AR) | 非自回归 (NAR) | 扩散模型 |
|------|------------|---------------|----------|
| 生成方式 | 逐 token 生成 | 并行生成所有 token | 迭代去噪 |
| 典型模型 | GPT, LLaMA, Qwen3 | CTC, Mask-CTC | DDPM, Flow-Matching, DiT |
| 优势 | 质量高、连贯性好 | 速度快 | 质量高、多样性好 |
| 劣势 | 速度慢 | 质量可能下降 | 速度慢、需要多步 |
| 适用场景 | 文本、TTS codec | 语音识别 | 图像/视频/语音合成 |

**SGLang-Omni 的处理**：
- AR 阶段：OmniScheduler + RadixAttention
- NAR 阶段：SimpleScheduler (无 KV Cache)
- 扩散阶段：DllmScheduler (LLaDA2 的并行去码)

### 6.2 KV Cache 管理（高频考点）

**面试问题**：请解释 KV Cache 的工作原理和优化策略。

**核心概念**：

```python
# KV Cache 的作用：避免重复计算
# 在自回归生成中，每一步只需要计算新 token 的 K, V
# 之前 token 的 K, V 可以缓存起来复用

# 内存占用计算
kv_cache_size = (
    2 *  # K 和 V
    num_layers *
    num_kv_heads *  # GQA 下可能小于 num_heads
    head_dim *
    seq_len *
    dtype_size  # float16 = 2 bytes
)
```

**SGLang 的 RadixAttention**：
- 使用 Radix Tree 管理 KV Cache
- 支持前缀共享和自动回收
- 与 PagedAttention 结合，高效内存管理

**优化策略**：
1. **PagedAttention**：分页管理，减少碎片
2. **GQA/MQA**：减少 KV head 数量
3. **量化**：KV Cache 量化 (FP8, INT8)
4. **前缀共享**：Radix Tree 自动共享公共前缀

### 6.3 注意力机制变体

**面试问题**：有哪些高效的注意力机制变体？

| 机制 | 原理 | 复杂度 | 适用场景 |
|------|------|--------|---------|
| **标准注意力** | Q·K^T / √d | O(n²) | 短序列 |
| **Flash Attention** | 分块 + 重计算 | O(n²) 但常数小 | 通用 |
| **Ring Attention** | 环形传递 KV | O(n²/p) | 长序列 + 多 GPU |
| **Ulysses Attention** | All-to-All 重分布 | O(n²/p) | 中等序列 + 多 GPU |
| **PagedAttention** | 分页 KV Cache | O(n²) 但内存高效 | 高吞吐服务 |
| **RadixAttention** | Radix Tree 前缀共享 | O(n²) 但缓存高效 | 多轮对话 |

### 6.4 MoE 架构理解

**面试问题**：请解释 MoE（Mixture of Experts）架构。

**核心概念**：

```python
# 门控机制
G(x) = Softmax(TopK(x · W_g, k))

# 专家计算
y = Σ_i G(x)_i * E_i(x)

# 负载均衡损失
L_aux = α * Σ_i (f_i * P_i)
# f_i: 路由到专家 i 的 token 比例
# P_i: 门控概率的平均值
```

**SGLang-Omni 中的 MoE**：
- Qwen3-Omni：30B 总参数，3B 激活参数
- Ming-Omni：100B MoE，TP=4

### 6.5 扩散模型原理

**面试问题**：请解释扩散模型的工作原理。

**核心概念**：

1. **前向过程（加噪）**
   ```
   x_t = √(α_t) * x_{t-1} + √(1-α_t) * ε
   ```

2. **反向过程（去噪）**
   ```
   x_{t-1} = (1/√α_t) * (x_t - (1-α_t)/√(1-ᾱ_t) * ε_θ(x_t, t))
   ```

3. **Flow-Matching（现代方法）**
   ```
   dx = v_θ(x, t) * dt
   ```

**SGLang-Omni 中的扩散支持**：
- LLaDA2.0-Uni：扩散 LLM，并行去码
- Ming-Omni：CFM + DiT 扩散 talker

### 6.6 量化技术

**面试问题**：有哪些模型量化方法？

| 方法 | 精度 | 原理 | 优缺点 |
|------|------|------|--------|
| **INT8** | W8A8 | 线性量化 | 简单高效，精度损失小 |
| **FP8** | W8A8 | 浮点量化 | 动态范围大，硬件支持好 |
| **MXFP4** | W4A4 | 微浮点 | 压缩率高，需要校准 |
| **GGUF** | 灵活 | 混合精度 | 通用性好，工具链成熟 |
| **GPTQ** | W4/W8 | 逐层量化 | 精度高，需要校准数据 |
| **AWQ** | W4 | 激活感知 | 保护重要权重 |

---

## 7. 面试常见问题与回答要点

### 7.1 项目相关问题

**Q1：请介绍一下你参与的这个项目。**

**回答要点**（2-3 分钟）：
1. **项目定位**：SGLang-Omni 是基于 SGLang 的全模态模型推理框架
2. **核心问题**：现有系统只支持单一范式，缺乏对 any-to-any 管道的支持
3. **解决方案**：多调度器架构 + Composition 设计 + 拓扑感知传输
4. **关键创新**：Async Decode、Stage Fusion、FeedbackARModelRunner
5. **性能成果**：Higgs TTS +12.7% 吞吐，-16.1% 延迟

**Q2：你在这个项目中遇到了什么挑战？如何解决的？**

**回答示例**：
> 最大的挑战是如何在不侵入 SGLang 上游代码的前提下复用其调度能力。我设计了基于 `__getattr__` 的 Composition 机制，通过绑定未绑定方法到当前实例，实现了选择性复用。同时制定了三条边界规则：Pin don't track、Minimize reuse surface、Upstream-first，确保长期可维护性。

**Q3：SGLang-Omni 和 vLLM-Omni 有什么区别？**

**回答要点**：
1. **基座框架**：SGLang vs vLLM，不同的 KV Cache 管理策略
2. **架构模式**：Composition vs Mixin，不同的代码复用方式
3. **调度器设计**：多调度器 vs 统一编排器
4. **优化重点**：AR TTS 延迟 vs 首包延迟 (TTFP)
5. **适用场景**：TTS 深度优化 vs 扩散模型广泛支持

### 7.2 技术深挖问题

**Q4：为什么选择 Composition 而不是继承？**

**回答要点**：
1. **解耦**：避免侵入 SGLang Scheduler 的 MRO
2. **选择性**：只复用需要的方法，覆盖需要定制的方法
3. **独立性**：SGLang 上游更新不会破坏 OmniScheduler
4. **可测试性**：更容易 mock 和单元测试

**Q5：Async Decode 如何保证正确性？**

**回答要点**：
1. **位精确验证**：bs=1 和 bs=4 下 100/100 贪心解码一致
2. **停止边界处理**：使用 GPU→GPU 设备快照，防止 lagged resolve 丢失 EOS
3. **最小批次阈值**：bs=1 走同步路径，避免开销
4. **前瞻开销**：固定开销在低并发时不划算

**Q6：如何处理多阶段推理中的错误传播？**

**回答要点**：
1. **统一错误捕获**：`run_batch()` 统一捕获异常，通过 outbox 传播
2. **模型代码禁止 broad catch**：特定异常捕获特定类型
3. **禁止 fallback**：要么成功，要么交给 Scheduler，不返回"假成功"
4. **CI 故障注入**：注入 OOM 验证正确的失败信号

### 7.3 系统设计问题

**Q7：如果要支持一个新的 TTS 模型，需要做什么？**

**回答步骤**：
1. **创建模型目录**：`models/<name>/` 包含 config.py、stages.py、callbacks.py
2. **定义 PipelineConfig**：声明阶段、路由、GPU 放置
3. **实现 Stage 工厂**：返回 SimpleScheduler 或 OmniScheduler
4. **实现 Callbacks**（如果是 AR + 码本）：write_buffers、extract_output、prefill_forward
5. **测试验证**：单元测试 + 集成测试 + 性能测试

**Q8：如何评估系统的性能瓶颈？**

**回答框架**：
1. **Profiling 工具**：PyTorch Profiler、NVIDIA Nsight
2. **关键指标**：TTFA、延迟、RTF、吞吐量
3. **瓶颈分析**：
   - 计算瓶颈：GPU 利用率低 → 优化算子
   - 内存瓶颈：OOM → 量化/卸载
   - 通信瓶颈：传输慢 → 优化连接器
4. **优化策略**：Async Decode、CUDA Graph、Stage Fusion

---

## 8. 自我介绍与项目经验展示

### 8.1 技术背景介绍模板

```
面试官您好，我是[姓名]，[学校]的[专业]硕士在读。

我的研究方向是[方向]，在[实验室/导师]指导下从事[研究内容]。

在实习期间，我参与了 SGLang-Omni 项目，这是一个面向全模态模型的推理服务框架。
我主要负责[具体工作]，解决了[具体问题]，取得了[具体成果]。

这个经历让我深入理解了[技术点]，也锻炼了我的[能力]。
```

### 8.2 项目经验介绍模板

```
项目名称：SGLang-Omni - 全模态模型推理服务框架

项目背景：
- 现有推理系统只支持单一范式（AR 或 DiT）
- 缺乏对 any-to-any 多模态管道的统一支持
- 多阶段推理的通信开销和延迟过高

我的工作：
1. [具体任务1]：设计并实现了[功能]，解决了[问题]
2. [具体任务2]：优化了[模块]，性能提升了[X]%
3. [具体任务3]：参与了[设计/评审]，贡献了[想法/代码]

技术亮点：
- [亮点1]：Composition over Inheritance，避免继承耦合
- [亮点2]：Async Decode，GPU forward 与 host collect 重叠
- [亮点3]：拓扑感知传输，三级降级策略

成果：
- 性能：Higgs TTS +12.7% 吞吐，-16.1% 延迟
- 代码：提交了 X 个 PR，合并了 Y 个
- 文档：撰写了 Z 篇技术文档
```

### 8.3 技术能力展示

**可展示的技术栈**：
- **框架理解**：SGLang、vLLM、PyTorch、Transformers
- **分布式系统**：CUDA、NCCL、ZMQ、多进程
- **优化技术**：CUDA Graph、Async Decode、Stage Fusion
- **工程能力**：Python、性能调优、系统设计

**可提及的算法知识**：
- 自回归模型：GPT、LLaMA、Qwen3
- 扩散模型：DDPM、Flow-Matching、DiT
- 注意力机制：Flash Attention、RadixAttention
- 并行策略：TP、PP、DP、EP

---

## 9. 扩展知识点与论文推荐

### 9.1 相关论文推荐

| 论文 | 主题 | 与项目的关联 |
|------|------|-------------|
| **SGLang** (OSDI '24) | RadixAttention | 项目基座 |
| **vLLM** (SOSP '23) | PagedAttention | 对比项目基座 |
| **Flash Attention** (NeurIPS '22) | 高效注意力 | AR 阶段优化 |
| **Ring Attention** (ICLR '24) | 长序列并行 | vLLM-Omni SP 并行 |
| **DiT** (ICCV '23) | Diffusion Transformer | 扩散模型架构 |
| **Flow-Matching** (ICLR '23) | 连续流生成 | 现代扩散方法 |
| **CUDA Graph** | GPU 图捕获 | 性能优化 |
| **MoE** (JMLR '22) | 混合专家 | Qwen3-Omni 架构 |

### 9.2 面试官可能追问的方向

1. **系统设计追问**
   - 如何处理故障恢复？
   - 如何实现动态扩缩容？
   - 如何保证多租户隔离？

2. **算法深挖追问**
   - KV Cache 的内存占用如何计算？
   - 连续批处理和静态批处理有什么区别？
   - 如何评估量化后的模型质量？

3. **工程实践追问**
   - 如何进行性能调优？
   - 如何设计测试用例？
   - 如何处理代码冲突？

### 9.3 准备建议

1. **深入理解架构**：能够画出四层架构图，解释每个组件的作用
2. **掌握核心创新**：Composition、Async Decode、拓扑感知传输的原理和实现
3. **准备具体案例**：能够描述一个你解决的技术问题及解决方案
4. **了解性能数据**：记住关键性能指标（吞吐、延迟、RTF）
5. **练习表达**：用 2-3 分钟清晰介绍项目，突出技术亮点
6. **对比分析**：能够对比 SGLang-Omni 和 vLLM-Omni 的设计差异

---

## 附录：关键代码位置索引

| 模块 | 文件路径 | 关键类/函数 |
|------|---------|------------|
| Coordinator | `sglang_omni/pipeline/coordinator.py` | `Coordinator` |
| Stage | `sglang_omni/pipeline/stage/runtime.py` | `Stage` |
| OmniScheduler | `sglang_omni/scheduling/omni_scheduler.py` | `OmniScheduler` |
| SimpleScheduler | `sglang_omni/scheduling/simple_scheduler.py` | `SimpleScheduler` |
| ModelRunner | `sglang_omni/model_runner/` | `ModelRunner`, `FeedbackARModelRunner` |
| Relay | `sglang_omni/relay/` | `SHMRelay`, `NCCLRelay`, `NixLRelay` |
| Qwen3-Omni | `sglang_omni/models/qwen3_omni/` | 多个文件 |
| Higgs TTS | `sglang_omni/models/higgs_tts/` | 多个文件 |
| MOSS TTS | `sglang_omni/models/moss_tts_local/` | 多个文件 |
| PipelineConfig | `sglang_omni/config/` | `StageConfig`, `PipelineConfig` |
| MultiProcessRunner | `sglang_omni/pipeline/mp_runner.py` | `MultiProcessRunner` |

---

**文档版本**：v1.0  
**最后更新**：2026-06-27  
**适用对象**：LLM 算法实习面试准备  
**参考项目**：SGLang-Omni、vLLM-Omni
