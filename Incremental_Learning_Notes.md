# SGLang-Omni 学习增量笔记

> 基于 8×4090 复现过程中对比前两轮面试准备文档，增量新学到的关键知识点

---

## 目录

1. [架构层面的增量理解](#1-架构层面的增量理解)
2. [优化技术的增量理解](#2-优化技术的增量理解)
3. [工程落地的增量理解](#3-工程落地的增量理解)
4. [分布式推理的增量理解](#4-分布式推理的增量理解)
5. [问题定位的增量理解](#5-问题定位的增量理解)
6. [业务场景的增量理解](#6-业务场景的增量理解)

---

## 1. 架构层面的增量理解

### 1.1 Pipeline 架构的深层理解

**前两轮的理解**：
- 知道有 Coordinator、Stage、Scheduler、ModelRunner 四层
- 知道使用 ZMQ + Relay 通信

**增量理解**：

1. **Stage 的 IO Shell 设计**
   ```
   Stage 不仅仅是"一个阶段"，而是一个通用的 IO Shell
   - 控制面 IO：ZMQ 接收 SubmitMessage、DataReadyMessage、AbortMessage
   - 数据面 IO：Relay 读写 tensor，CUDA IPC 快速路径
   - 流路由：接收/发送 streaming chunks
   - 调度器桥接：将消息推入 scheduler.inbox，从 scheduler.outbox 拉取结果
   
   关键洞察：Stage 是模型无关的，它不关心里面跑的是什么模型
   ```

2. **数据面与控制面分离的实际意义**
   ```
   之前理解：知道是分离的
   现在理解：
   - ZMQ 只携带小消息（SubmitMessage ~100B，DataReadyMessage ~1KB）
   - Relay 携带大 tensor（hidden states ~10MB，audio codes ~1MB）
   - 这种分离使得控制面可以快速响应（abort、shutdown），不受数据面阻塞
   
   实际问题：如果不分离，大 tensor 传输会阻塞 abort 信号
   ```

3. **LOCAL_OBJECT 快速路径的约束**
   ```
   之前理解：知道有 LOCAL_OBJECT 快速路径
   现在理解：
   - 只能用于同进程的单目标路由
   - 扇出时需要投影函数返回独立的 data 容器
   - 接收方必须将对象视为只读
   - 不可用于流式 chunk（需要 CUDA IPC 或 Relay）
   
   实际影响：Stage Fusion 后，阶段间通信走 LOCAL_OBJECT，延迟 -33%
   ```

### 1.2 多调度器设计的深层理解

**前两轮的理解**：
- 知道有 OmniScheduler、SimpleScheduler、Code2WavScheduler
- 知道 OmniScheduler 复用 SGLang 的调度逻辑

**增量理解**：

1. **OmniScheduler 的 Composition 机制细节**
   ```python
   # 之前理解：用 __getattr__ 委托
   # 现在理解：绑定未绑定方法到当前实例
   
   def __getattr__(self, name: str):
       attr = getattr(_Upstream, name)  # 从上游类获取
       if callable(attr):
           return types.MethodType(attr, self)  # 绑定到当前实例
       return attr
   
   # 关键洞察：这样上游方法内部调用 self.xxx 时，
   # 会使用 OmniScheduler 的属性，而非上游 Scheduler 的属性
   ```

2. **覆盖方法的原因**
   ```
   之前理解：知道要覆盖一些方法
   现在理解：
   
   recv_requests(): 从 inbox 队列取消息，而非 ZMQ
   - 原因：Stage 通过 inbox/outbox 通信，不直接用 ZMQ
   
   process_input_requests(): 使用 request_builder 转换
   - 原因：StagePayload 需要转换为 SGLang 的请求格式
   
   run_batch(): 委托给 ModelRunner
   - 原因：需要自定义 forward 逻辑（多模态注入、反馈机制）
   
   stream_output(): 结果放入 outbox
   - 原因：结果需要通过 Stage 路由到下游，而非发送到 detokenizer
   
   send_to_tokenizer(): No-op
   - 原因：tokenizer 初始化在 Stage 层面处理
   ```

3. **SimpleScheduler 的批处理机制**
   ```python
   # 之前理解：简单的 inbox→fn→outbox
   # 现在理解：支持批处理和并发
   
   class SimpleScheduler:
       def __init__(self, compute_fn, batch_compute_fn=None, 
                    max_batch_size=1, max_concurrency=1):
           # max_batch_size: 收集多个请求一起处理
           # max_concurrency: 多个 worker 并发处理
           # batch_compute_fn: 批量处理函数
   
   # 实际应用：vocoder 使用 batch_compute_fn 批量解码
   # 收益：+7.8% 吞吐，-20% p99 延迟
   ```

### 1.3 ModelRunner 的 Hook 设计

**前两轮的理解**：
- 知道有 before_prefill、before_decode、custom_prefill_forward、custom_decode_forward、post_prefill、post_decode

**增量理解**：

1. **Hook 的执行顺序**
   ```
   ForwardBatch → before_*() → custom_*_forward() or standard forward → post_*() → sample/output
                     ↑ hook             ↑ explicit custom path          ↑ hook
   
   关键洞察：
   - before_*: 修改 batch（注入多模态 embeds、写入反馈）
   - custom_*_forward: 自定义 forward（可选，返回 None 则走标准 forward）
   - post_*: 提取输出（写入 outbox、更新 feedback buffer）
   ```

2. **FeedbackARModelRunner 的三个回调**
   ```python
   # 之前理解：知道有三个回调
   # 现在理解：每个回调的具体职责
   
   write_buffers_fn: 在 decode 前写入上一步的反馈
   - 写入 feedback_embeds 到 model._feedback_buffer
   - 处理 trailing/padding
   
   extract_output_fn: 在 decode 后提取输出
   - 从 model._output_codes 提取到 outbox
   - 从 model._output_embeds 提取到 feedback
   
   prefill_forward_fn: 自定义 prefill forward（可选）
   - 投影 input_embeds
   - 处理多模态输入
   ```

---

## 2. 优化技术的增量理解

### 2.1 CUDA Graph 的工程细节

**前两轮的理解**：
- 知道 CUDA Graph 可以消除 Python overhead
- 知道需要固定的 batch size

**增量理解**：

1. **Shadow buffers 解决 padding 问题**
   ```python
   # 问题：CUDA Graph 捕获时需要固定的 batch size
   # 实际 batch size < 捕获 batch size 时，需要 padding
   # padding 位置的 logits 不应该用于采样
   
   # 解决：Shadow buffers
   class CudaGraphRunner:
       def replay(self, batch):
           # 1. 拷贝活跃行到 shadow buffer
           self.shadow_buffer[:len(batch)] = batch.input_ids
           
           # 2. 使用 shadow buffer 进行 replay
           self.graph.replay(self.shadow_buffer)
           
           # 3. 只取活跃行的输出
           return self.output_buffer[:len(batch)]
   
   # 效果：catastrophic WER 从 4-12% 降到 0.43%
   ```

2. **per-T CUDA Graph 用于流式声码器**
   ```python
   # 问题：streaming codec decode 的 launch overhead 过高
   # 原因：每帧都需要重新 launch CUDA kernel
   
   # 解决：为每个 T 值（时间步长度）捕获独立的 CUDA Graph
   class StreamingVocoder:
       def __init__(self):
           self.graphs = {}  # T -> CUDAGraph
       
       def decode_step(self, codes, T):
           if T not in self.graphs:
               # 首次遇到这个 T，捕获 graph
               self.graphs[T] = self._capture_graph(codes, T)
           
           # 使用预捕获的 graph
           return self.graphs[T].replay(codes)
   
   # 效果：T=5 时 65.8→30.7 ms/step (2.14× 加速)
   ```

3. **CUDA Graph 的状态管理**
   ```
   问题：模型有 KV Cache、feedback buffer 等状态
   解决：使用 persistent buffer，graph 内外共享
   
   关键：buffer 地址必须固定
   方案：预分配固定大小的 buffer，使用 row index 而非 pointer
   ```

### 2.2 Async Decode 的工程细节

**前两轮的理解**：
- 知道是一步前瞻，GPU forward 与 host collect 重叠

**增量理解**：

1. **bs=1 的特殊处理**
   ```python
   # 问题：bs=1 时前瞻开销大于收益
   # 原因：固定开销（CUDA event、同步）相对于计算时间占比大
   
   # 解决：bs=1 走同步路径
   if batch_size < self.async_decode_min_batch_size:
       return self._run_batch_sync(batch)
   else:
       return self._run_batch_async(batch)
   
   # 默认 async_decode_min_batch_size=2
   ```

2. **位精确验证的重要性**
   ```python
   # 验证方法：对比同步和异步模式的输出
   # 条件：bs=1 和 bs=4，greedy decoding，100/100 samples
   # 结果：完全一致
   
   # 为什么重要：
   # - 保证优化不改变模型行为
   # - 便于调试（可以用同步模式复现问题）
   # - 建立信任（用户可以验证）
   ```

3. **停止边界处理**
   ```python
   # 问题：lagged resolve 可能丢失 EOS token
   # 解决：使用 GPU→GPU 设备快照
   
   def _run_batch_resolve(self, batch, sched_output, pending_step):
       # 1. 从 GPU 快照获取 stop id
       stop_id_snapshot = pending_step.stop_id_snapshot
       
       # 2. 检查是否应该停止
       if stop_id_snapshot == EOS_TOKEN:
           # 标记请求完成
           pass
       
       # 3. 继续处理
       return self._process_batch_result(batch)
   ```

### 2.3 Stage Fusion 的工程细节

**前两轮的理解**：
- 知道可以将相邻阶段融合为单进程

**增量理解**：

1. **融合约束的实际意义**
   ```
   相邻性：阶段必须在拓扑中相邻
   - 原因：非相邻阶段之间可能有其他阶段，融合会破坏拓扑
   
   有序性：必须是线性顺序
   - 原因：非线性顺序（如分支）无法简单融合
   
   非 TP：融合的阶段不能有 TP
   - 原因：TP 需要 NCCL 通信，融合后无法保证同步
   
   单 GPU：所有阶段必须在同一 GPU
   - 原因：跨 GPU 融合没有意义，通信开销不变
   ```

2. **融合的实现机制**
   ```python
   # 编译时验证融合约束
   def validate_fusion_groups(stages, fusion_groups):
       for group in fusion_groups:
           # 检查相邻性
           for i in range(len(group) - 1):
               if group[i+1] not in stages[group[i]].next:
                   raise ValueError(f"Stages {group[i]} and {group[i+1]} are not adjacent")
           
           # 检查非 TP
           for stage_name in group:
               if stages[stage_name].tp_size > 1:
                   raise ValueError(f"Stage {stage_name} has TP={stages[stage_name].tp_size}")
           
           # 检查单 GPU
           gpus = set(stages[stage_name].gpu for stage_name in group)
           if len(gpus) > 1:
               raise ValueError(f"Stages in group span multiple GPUs: {gpus}")
   
   # 运行时进程拓扑规划
   def merge_process_groups(groups, fusion_groups):
       for group in fusion_groups:
           # 找到包含这些阶段的进程组
           # 合并为单个进程组
           pass
   ```

---

## 3. 工程落地的增量理解

### 3.1 错误处理的工程实践

**前两轮的理解**：
- 知道需要统一的错误处理

**增量理解**：

1. **统一错误捕获在 Scheduler 层**
   ```python
   # 问题：之前不同模型的错误处理不一致
   # S2-Pro 和 Voxtral 返回 HTTP 500（正确）
   # Ming-Omni 返回 HTTP 200 with waveform=None（静默失败）
   # Qwen3-Omni 返回 HTTP 200 with zero tensor（看起来有效）
   
   # 解决：在 Scheduler.run_batch() 统一捕获
   class OmniScheduler:
       def run_batch(self, batch):
           try:
               return self._run_batch(batch)
           except Exception as exc:
               self._handle_batch_failure(batch, exc)
               return _FAILED_BATCH_RESULT
   ```

2. **模型代码禁止 broad catch**
   ```python
   # 规则：模型代码不允许 except Exception
   # 原因：会吞掉错误，导致静默失败
   
   # 正确做法：只捕获特定异常
   try:
       result = model.forward(input)
   except RuntimeError as e:
       if "out of memory" in str(e):
           # 处理 OOM
           pass
       else:
           raise  # 重新抛出其他异常
   ```

3. **流式错误处理的挑战**
   ```python
   # 问题：HTTP headers 已发送，无法返回 500
   # 解决：发送错误事件，然后关闭连接
   
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

### 3.2 资源管理的工程实践

**前两轮的理解**：
- 知道需要管理 GPU 内存、KV Cache 等

**增量理解**：

1. **mem_fraction_static 的语义**
   ```
   vLLM 的 gpu_memory_utilization：总 VRAM 的分数
   SGLang 的 mem_fraction_static：权重加载后剩余 VRAM 的分数
   
   实际影响：
   - 4090 24GB，权重 8GB，剩余 16GB
   - mem_fraction_static=0.8 → KV Cache = 16GB * 0.8 = 12.8GB
   - 实际可用 KV Cache = 12.8GB - CUDA Graph - 其他开销
   ```

2. **阶段共置的内存预算**
   ```python
   # 问题：多个阶段共享 GPU，如何分配内存？
   # 解决：per-stage memory budgeting
   
   config = Qwen3OmniSpeechColocatedPipelineConfig(
       total_gpu_memory_fraction=0.95,  # 总 GPU 内存预算
       stage_overrides={
           "thinker": {"total_gpu_memory_fraction": 0.7},
           "talker_ar": {"total_gpu_memory_fraction": 0.15},
           "code2wav": {"total_gpu_memory_fraction": 0.02},
       }
   )
   
   # 关键洞察：需要为 CUDA Graph、backend workspaces、activation peaks 留空间
   ```

3. **进程崩溃后的资源清理**
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

### 3.3 测试策略的工程实践

**前两轮的理解**：
- 知道需要单元测试和集成测试

**增量理解**：

1. **测试分层**
   ```
   unit_test/: 快速，不需要模型下载或 GPU
   - pipeline/: 框架契约测试
   - qwen3_omni/: 模型特定行为
   - fishaudio_s2_pro/: 模型特定行为
   
   test_model/: 慢速，需要真实模型和 GPU
   - 基准测试
   - WER 测试
   - 端到端测试
   ```

2. **位精确验证**
   ```python
   # 验证优化不改变模型行为
   def verify_bit_identity(sync_results, async_results):
       for sync_out, async_out in zip(sync_results, async_results):
           assert torch.equal(sync_out, async_out), "Output mismatch!"
       print("✓ Bit-identity verified")
   
   # 使用场景：
   # - CUDA Graph 开启/关闭
   # - Async Decode 开启/关闭
   # - Stage Fusion 前后
   ```

3. **故障注入测试**
   ```python
   # 验证错误处理的正确性
   def test_oom_handling():
       # 1. 注入 OOM
       with mock.patch('torch.cuda.empty_cache', side_effect=RuntimeError("CUDA OOM")):
           # 2. 发送请求
           response = send_request(...)
           
           # 3. 验证返回 500
           assert response.status_code == 500
           assert "CUDA OOM" in response.text
   ```

---

## 4. 分布式推理的增量理解

### 4.1 Tensor Parallelism 的工程细节

**前两轮的理解**：
- 知道 TP 将模型权重分片到多个 GPU

**增量理解**：

1. **4090 的 TP 挑战**
   ```
   问题：4090 没有 NVLink，使用 PCIe 4.0 通信
   PCIe 4.0 带宽：32 GB/s (x16)
   NVLink 带宽：900 GB/s (H200)
   
   影响：
   - TP 通信开销大，效率低
   - 需要减少 TP degree，或者使用 CPU offload
   ```

2. **NCCL 端点管理**
   ```python
   # 问题：每个 TP stage 需要独立的 NCCL port
   # 解决：使用 _NcclPortAllocator
   
   class MultiProcessRunner:
       def __init__(self):
           self._nccl_port_allocator = NcclPortAllocator()
       
       def _build_stage_groups(self, config):
           for stage in config.stages:
               if stage.tp_size > 1:
                   # 为每个 TP stage 分配独立的 NCCL port
                   port = self._nccl_port_allocator.allocate()
                   # 创建 TP group
                   pass
   ```

3. **rank 0 的特殊角色**
   ```
   在 TP group 中：
   - rank 0 接收控制面消息
   - rank 0 广播给其他 rank
   - 所有 rank 做相同的调度决策
   - 只有 rank 0 发送结果到下游
   
   原因：保证所有 rank 的状态一致
   ```

### 4.2 Router 的工程细节

**前两轮的理解**：
- 知道 Router 可以负载均衡到多个 worker

**增量理解**：

1. **Worker 选择策略**
   ```python
   # 支持的策略：
   # - round_robin: 轮询
   # - least_request: 最少请求
   # - random: 随机
   
   class Router:
       def select_worker(self, request):
           if self.policy == "round_robin":
               return self._round_robin()
           elif self.policy == "least_request":
               return self._least_request()
           elif self.policy == "random":
               return self._random()
   ```

2. **健康检查和故障转移**
   ```python
   class Router:
       def __init__(self):
           self._worker_health = {}  # worker_id -> health status
           self._health_check_interval = 5  # 秒
       
       async def _health_check_loop(self):
           while True:
               for worker in self.workers:
                   try:
                       response = await self._check_health(worker)
                       self._worker_health[worker.id] = response.ok
                   except Exception:
                       self._worker_health[worker.id] = False
               
               await asyncio.sleep(self._health_check_interval)
       
       def select_worker(self, request):
           # 只选择健康的 worker
           healthy_workers = [w for w in self.workers if self._worker_health[w.id]]
           return self._select_from_healthy(healthy_workers, request)
   ```

3. **Router 不分割请求**
   ```
   关键洞察：Router 将整个请求路由到一个 worker
   不会将请求分割到多个 worker（如 thinker 在 worker1，talker 在 worker2）
   
   原因：
   - 简化实现
   - 避免跨 worker 通信
   - 保证请求的局部性
   ```

---

## 5. 问题定位的增量理解

### 5.1 性能 Profiling 的工程实践

**前两轮的理解**：
- 知道需要 profiling 来定位瓶颈

**增量理解**：

1. **Request-level Profiler**
   ```python
   # SGLang-Omni 内置的 profiler
   from sglang_omni.profiler.event_recorder import emit, get_active_stage
   
   # 记录事件
   emit(
       request_id=request_id,
       stage=get_active_stage(),
       event_name="scheduler_prefill_start",
       metadata={"batch_size": len(batch)}
   )
   
   # 分析结果
   # - 每个阶段的耗时
   # - 每个请求的生命周期
   # - 瓶颈分析
   ```

2. **关键指标**
   ```
   TTFA (Time to First Audio): 首包延迟
   - 目标：< 500ms
   - 影响因素：prefill 时间、调度延迟
   
   RTF (Real-Time Factor): 实时因子
   - 目标：< 1.0
   - 计算：生成时间 / 音频时长
   - RTF < 1.0 表示可以实时生成
   
   Throughput: 吞吐量
   - 单位：req/s 或 audio_s/s
   - 影响因素：并发度、批处理大小、GPU 利用率
   ```

3. **瓶颈分析方法**
   ```
   1. 使用 profiler 获取每个阶段的耗时
   2. 识别最慢的阶段（瓶颈）
   3. 分析瓶颈的类型：
      - Compute-bound: GPU 利用率高，计算密集
      - Memory-bound: 内存带宽受限
      - Communication-bound: 通信受限
   4. 针对性优化：
      - Compute-bound: 优化算子、使用 FP8
      - Memory-bound: 减少内存访问、使用 CUDA Graph
      - Communication-bound: 减少通信、使用 Stage Fusion
   ```

### 5.2 实际问题定位案例

**前两轮的理解**：
- 知道 CUDA Graph WER 退化、encoder 批处理退化等案例

**增量理解**：

1. **CUDA Graph WER 退化的排查过程**
   ```
   现象：bs≥8 时 WER 从 <1% 飙升到 4-12%
   
   排查步骤：
   1. 确认复现条件：与 batch size 相关，非随机
   2. 快速止血：禁用 CUDA Graph
   3. 对比 logits：发现 bs≥8 时某些位置 logits 不一致
   4. 分析 padding：CUDA Graph 的 padding 导致 logits 错误
   5. 定位 bug：padding 位置的 logits 被错误地用于采样
   6. 解决方案：Shadow buffers
   7. 验证结果：WER 从 4-12% 降到 0.43%
   
   关键洞察：问题定位需要系统性排查，不能只看表面
   ```

2. **Encoder 批处理退化的排查过程**
   ```
   现象：批处理 encoder 后吞吐量反而下降 18-25%
   
   排查步骤：
   1. 确认实验结果：确实是下降
   2. 分析 GPU 利用率：encoder 和 AR engine 共享 GPU
   3. 分析资源竞争：批处理 encoder 占用更多 GPU 时间
   4. 分析瓶颈：AR engine 是瓶颈（~79% 的时间）
   5. 结论：批处理 encoder 会抢占 AR engine 的 GPU 时间
   
   关键洞察：批处理的收益取决于瓶颈在哪里
   - 瓶颈在 vocoder（后处理）：批处理有效
   - 瓶颈在 AR engine（主计算）：批处理无效
   - encoder 与 AR engine 共享资源：批处理可能有害
   ```

---

## 6. 业务场景的增量理解

### 6.1 TTS 服务的场景价值

**前两轮的理解**：
- 知道 TTS 服务有首包延迟、音质、吞吐量等指标

**增量理解**：

1. **不同场景的需求差异**
   ```
   智能客服：
   - 核心需求：低延迟（<200ms TTFA）
   - 并发度：中等（10-100 并发）
   - 音质要求：中等（WER < 2%）
   - 优化重点：Async Decode、流式输出
   
   有声读物：
   - 核心需求：高音质、低成本
   - 并发度：低（1-10 并发）
   - 音质要求：高（WER < 1%）
   - 优化重点：批处理、CUDA Graph
   
   实时翻译：
   - 核心需求：超低延迟（<100ms TTFA）
   - 并发度：低（1-5 并发）
   - 音质要求：中等
   - 优化重点：流式输入输出、Stage Fusion
   ```

2. **成本分析**
   ```
   4090 部署：
   - 单卡成本：~$1.5/小时（云服务）
   - 吞吐量：~3 req/s（Higgs TTS）
   - 每请求成本：$1.5 / (3 * 3600) = $0.00014/req
   
   H200 部署：
   - 单卡成本：~$3/小时（云服务）
   - 吞吐量：~8 req/s（Higgs TTS）
   - 每请求成本：$3 / (8 * 3600) = $0.00010/req
   
   结论：H200 单位成本更低，但 4090 总成本更低（适合小规模）
   ```

3. **资源有限时的优化优先级**
   ```
   P0: CUDA Graph (vocoder)
   - 收益：+22% 延迟
   - 复杂度：中
   - 风险：低
   
   P1: Async Decode
   - 收益：+12% 吞吐
   - 复杂度：中
   - 风险：低
   
   P2: Stage Fusion
   - 收益：-33% 延迟
   - 复杂度：低
   - 风险：低
   
   P3: 批处理 vocoder
   - 收益：+8% 吞吐
   - 复杂度：低
   - 风险：低
   ```

### 6.2 场景匹配的增量理解

**前两轮的理解**：
- 知道 SGLang-Omni 适合 TTS、Omni 模型

**增量理解**：

1. **SGLang-Omni 的优势场景**
   ```
   TTS 服务：
   - 原生支持，深度优化
   - 多模型支持（Higgs、Fish、MOSS、Qwen3）
   - 流式输出，低延迟
   
   Omni 模型服务：
   - Qwen3-Omni、Ming-Omni
   - 多阶段编排
   - 阶段共置，节省资源
   
   低延迟场景：
   - 流式输出
   - Async Decode
   - CUDA Graph
   ```

2. **SGLang-Omni 的劣势场景**
   ```
   扩散模型服务：
   - 有限支持（只有 LLaDA2）
   - vLLM-Omni 更适合（TeaCache、多种并行策略）
   
   大规模分布式推理：
   - 只支持 TP，不支持 PP/DP/EP
   - vLLM-Omni 更适合（六维并行）
   
   RL 训练集成：
   - 基础支持
   - vLLM-Omni + VeRL-Omni 更适合
   ```

3. **4090 部署的特殊考虑**
   ```
   显存限制：
   - 小模型（<4B）：单卡可运行
   - 中等模型（4-8B）：单卡 + 量化
   - 大模型（>30B）：需要 TP
   
   互连限制：
   - PCIe 4.0 通信效率低
   - TP 开销大
   - 建议：尽量使用单卡，减少 TP
   
   优化重点：
   - CUDA Graph：减少 Python overhead
   - Async Decode：重叠计算和通信
   - Stage Fusion：减少通信开销
   ```

---

## 总结：增量学习的关键点

### 架构层面
1. **Stage 是 IO Shell**：模型无关，只负责通信和调度
2. **Composition 的细节**：绑定未绑定方法，覆盖方法的原因
3. **SimpleScheduler 的批处理**：支持 batch_compute_fn 和 max_concurrency

### 优化技术
1. **Shadow buffers**：解决 CUDA Graph padding 问题
2. **per-T CUDA Graph**：为流式声码器的每个 T 值捕获独立 graph
3. **bs=1 的特殊处理**：走同步路径，避免前瞻开销

### 工程落地
1. **统一错误捕获**：在 Scheduler 层捕获，模型禁止 broad catch
2. **mem_fraction_static 语义**：权重加载后剩余 VRAM 的分数
3. **测试分层**：unit_test (快速) + test_model (慢速)

### 分布式推理
1. **4090 的 TP 挑战**：PCIe 通信效率低
2. **Router 不分割请求**：整个请求路由到一个 worker
3. **rank 0 的特殊角色**：接收控制面，广播给其他 rank

### 问题定位
1. **Request-level Profiler**：内置的性能分析工具
2. **瓶颈分析方法**：识别瓶颈类型，针对性优化
3. **系统性排查**：不能只看表面，需要深入分析

### 业务场景
1. **不同场景的需求差异**：智能客服 vs 有声读物 vs 实时翻译
2. **成本分析**：4090 vs H200 的单位成本对比
3. **优化优先级**：基于 ROI 排序

---

**文档版本**：v1.0  
**最后更新**：2026-06-27  
**适用对象**：基于 8×4090 复现 SGLang-Omni 的学习者  
**参考文档**：LLM_Interview_Preparation_Guide.md、Technical_Interview_Five_Categories.md
