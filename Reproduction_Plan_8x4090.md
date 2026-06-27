# SGLang-Omni 8×4090 复现计划

> 基于 8 卡 RTX 4090 (24GB 显存) 的完整复现方案

---

## 目录

1. [硬件资源评估](#1-硬件资源评估)
2. [复现阶段规划](#2-复现阶段规划)
3. [Phase 1: 环境搭建与小模型复现](#3-phase-1-环境搭建与小模型复现)
4. [Phase 2: 中等模型复现与优化](#4-phase-2-中等模型复现与优化)
5. [Phase 3: 大模型复现与分布式](#5-phase-3-大模型复现与分布式)
6. [Phase 4: 性能优化与对比](#6-phase-4-性能优化与对比)
7. [学习路径与关键知识点](#7-学习路径与关键知识点)
8. [常见问题与解决方案](#8-常见问题与解决方案)

---

## 1. 硬件资源评估

### 1.1 4090 硬件规格

| 参数 | 规格 |
|------|------|
| 显存 | 24GB GDDR6X |
| 架构 | Ada Lovelace |
| FP16 性能 | 82.6 TFLOPS |
| FP8 性能 | 165.2 TFLOPS |
| 内存带宽 | 1008 GB/s |
| 互连 | PCIe 4.0 (无 NVLink) |

### 1.2 各模型显存需求评估

| 模型 | 参数量 | FP16 显存 | INT8 显存 | 4090 可行性 |
|------|--------|----------|----------|------------|
| **Qwen3-TTS 0.6B** | 0.6B | ~1.2GB | ~0.6GB | ✅ 单卡轻松运行 |
| **Qwen3-TTS 1.7B** | 1.7B | ~3.4GB | ~1.7GB | ✅ 单卡运行 |
| **Fish Audio S2-Pro** | ~1B | ~2GB | ~1GB | ✅ 单卡运行 |
| **Voxtral TTS 4B** | 4B | ~8GB | ~4GB | ✅ 单卡运行 |
| **Higgs TTS 4B** | 4B | ~8GB | ~4GB | ✅ 单卡运行 |
| **MOSS-TTS-Local 4B** | 4B | ~8GB | ~4GB | ✅ 单卡运行 |
| **MOSS-TTS 8B** | 8B | ~16GB | ~8GB | ✅ 单卡运行 |
| **Qwen3-Omni 30B (MoE)** | 30B (3B active) | ~60GB | ~30GB | ⚠️ 需要 4 卡 TP |
| **Ming-Omni 100B (MoE)** | 100B | ~200GB | ~100GB | ⚠️ 需要 8 卡 + CPU offload |

### 1.3 显存预算分配

**单卡部署（24GB）**：
```
模型权重: 8-16GB
KV Cache: 4-8GB
CUDA Graph: 2-4GB
其他开销: 2-4GB
总计: 16-24GB
```

**多卡部署（TP=4, 4×24GB=96GB）**：
```
模型权重: 60GB (分片到 4 卡)
KV Cache: 20GB
CUDA Graph: 8GB
其他开销: 8GB
总计: 96GB
```

### 1.4 与 H200 的对比

| 指标 | 4090 | H200 | 差距 |
|------|------|------|------|
| 显存 | 24GB | 141GB | 5.9× |
| 内存带宽 | 1008 GB/s | 4800 GB/s | 4.8× |
| FP16 性能 | 82.6 TFLOPS | 989 TFLOPS | 12× |
| 互连 | PCIe 4.0 | NVLink | - |

**影响**：
- **显存限制**：大模型需要 TP，增加通信开销
- **带宽限制**：memory-bound 任务（如 AR decode）性能下降
- **计算限制**：compute-bound 任务（如 prefill）性能下降
- **互连限制**：TP 通信效率低，需要优化

---

## 2. 复现阶段规划

### 2.1 总体时间线

| 阶段 | 时间 | 目标 | 关键产出 |
|------|------|------|----------|
| **Phase 1** | 第 1-2 周 | 环境搭建 + 小模型复现 | 能跑通 TTS demo |
| **Phase 2** | 第 3-4 周 | 中等模型复现 + 优化 | 完成性能基准测试 |
| **Phase 3** | 第 5-6 周 | 大模型复现 + 分布式 | 完成 Qwen3-Omni 复现 |
| **Phase 4** | 第 7-8 周 | 性能优化 + 对比 | 完成优化对比报告 |

### 2.2 详细任务分解

**Phase 1: 环境搭建与小模型复现（第 1-2 周）**

```
Week 1:
├── Day 1-2: 环境搭建
│   ├── 安装 CUDA 12.x、PyTorch 2.x
│   ├── 安装 SGLang-Omni 依赖
│   └── 验证 GPU 可用性
├── Day 3-4: 小模型复现 (Qwen3-TTS 0.6B)
│   ├── 下载模型
│   ├── 启动服务
│   └── 测试 TTS 功能
└── Day 5: 小模型复现 (Fish S2-Pro)
    ├── 下载模型
    ├── 启动服务
    └── 测试语音克隆

Week 2:
├── Day 1-2: 流式输出测试
│   ├── 测试流式 TTS
│   ├── 测量首包延迟
│   └── 分析性能瓶颈
├── Day 3-4: 代码走读
│   ├── 理解 Pipeline 架构
│   ├── 理解 Scheduler 设计
│   └── 理解数据流
└── Day 5: 文档整理
    ├── 记录复现过程
    ├── 记录问题和解决方案
    └── 准备 Phase 2
```

**Phase 2: 中等模型复现与优化（第 3-4 周）**

```
Week 3:
├── Day 1-2: Higgs TTS 复现
│   ├── 下载模型 (4B)
│   ├── 配置 CUDA Graph
│   └── 测试多码本解码
├── Day 3-4: MOSS-TTS-Local 复现
│   ├── 下载模型 (4B)
│   ├── 配置流式输出
│   └── 测试 48kHz 输出
└── Day 5: 性能基准测试
    ├── 测试不同并发度
    ├── 测量延迟和吞吐量
    └── 对比不同模型

Week 4:
├── Day 1-2: 优化实验
│   ├── 测试 CUDA Graph 效果
│   ├── 测试 Async Decode 效果
│   └── 测试批处理效果
├── Day 3-4: 问题定位练习
│   ├── 故意制造 OOM
│   ├── 分析错误日志
│   └── 练习问题排查
└── Day 5: 文档整理
    ├── 记录优化效果
    ├── 记录问题和解决方案
    └── 准备 Phase 3
```

**Phase 3: 大模型复现与分布式（第 5-6 周）**

```
Week 5:
├── Day 1-2: Qwen3-Omni TP 配置
│   ├── 配置 TP=4 (4 卡)
│   ├── 测试 Thinker 阶段
│   └── 测试 Talker 阶段
├── Day 3-4: 完整管道测试
│   ├── 测试文本输入 → 文本输出
│   ├── 测试文本输入 → 音频输出
│   └── 测试多模态输入
└── Stage Fusion 测试
    ├── 测试阶段共置
    ├── 测量通信开销
    └── 对比不同配置

Week 6:
├── Day 1-2: Ming-Omni 复现 (如果显存足够)
│   ├── 配置 TP=8
│   ├── 测试 MoE 推理
│   └── 测试多模态输出
├── Day 3-4: Router 测试
│   ├── 配置多副本
│   ├── 测试负载均衡
│   └── 测试故障转移
└── Day 5: 文档整理
    ├── 记录分布式经验
    ├── 记录问题和解决方案
    └── 准备 Phase 4
```

**Phase 4: 性能优化与对比（第 7-8 周）**

```
Week 7:
├── Day 1-2: 性能优化
│   ├── 优化 CUDA Graph 配置
│   ├── 优化 KV Cache 大小
│   └── 优化批处理策略
├── Day 3-4: 对比测试
│   ├── 对比不同模型
│   ├── 对比不同配置
│   └── 对比 4090 vs H200
└── Day 5: 优化报告
    ├── 总结优化效果
    ├── 分析瓶颈
    └── 提出改进建议

Week 8:
├── Day 1-2: 文档完善
│   ├── 整理复现文档
│   ├── 整理优化文档
│   └── 整理面试准备文档
├── Day 3-4: 面试准备
│   ├── 准备项目介绍
│   ├── 准备技术深挖
│   └── 准备问题回答
└── Day 5: 总结与回顾
    ├── 回顾学习过程
    ├── 总结关键知识点
    └── 准备下一步计划
```

---

## 3. Phase 1: 环境搭建与小模型复现

### 3.1 环境搭建

**Step 1: 检查 GPU 状态**

```bash
# 检查 GPU 信息
nvidia-smi

# 检查 CUDA 版本
nvcc --version

# 检查 GPU 显存
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv
```

**Step 2: 安装 CUDA 和 cuDNN**

```bash
# 安装 CUDA 12.x (如果未安装)
wget https://developer.download.nvidia.com/compute/cuda/12.4.0/local_installers/cuda_12.4.0_550.54.14_linux.run
sudo sh cuda_12.4.0_550.54.14_linux.run

# 设置环境变量
export PATH=/usr/local/cuda-12.4/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-12.4/lib64:$LD_LIBRARY_PATH
```

**Step 3: 安装 PyTorch**

```bash
# 安装 PyTorch 2.x with CUDA 12.4
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

# 验证安装
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
```

**Step 4: 安装 SGLang-Omni**

```bash
# 克隆仓库
git clone https://github.com/sgl-project/sglang-omni.git
cd sglang-omni

# 创建虚拟环境
python -m venv .venv
source .venv/bin/activate

# 安装依赖
pip install -e .

# 或者使用 Docker (推荐)
docker pull lmsysorg/sglang-omni:dev
docker run -it --gpus all --ipc host --network host -v $(pwd):/workspace lmsysorg/sglang-omni:dev
```

**Step 5: 验证安装**

```bash
# 检查 SGLang-Omni 版本
python -c "import sglang_omni; print(sglang_omni.__version__)"

# 检查 GPU 可用性
python -c "import torch; print(f'GPUs: {torch.cuda.device_count()}'); print(f'GPU 0: {torch.cuda.get_device_name(0)}')"
```

### 3.2 小模型复现: Qwen3-TTS 0.6B

**Step 1: 下载模型**

```bash
# 下载模型 (~1.2GB)
huggingface-cli download Qwen/Qwen3-TTS-12Hz-0.6B-Base

# 或者使用 Python
from huggingface_hub import snapshot_download
snapshot_download("Qwen/Qwen3-TTS-12Hz-0.6B-Base")
```

**Step 2: 启动服务**

```bash
# 启动 TTS 服务
sgl-omni serve \
  --model-path Qwen/Qwen3-TTS-12Hz-0.6B-Base \
  --config examples/configs/qwen3_tts_0_6b.yaml \
  --port 8000

# 等待服务启动（首次启动可能需要几分钟捕获 CUDA Graph）
```

**Step 3: 测试 TTS 功能**

```bash
# 基础 TTS
curl -X POST http://localhost:8000/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{"input": "Hello, how are you?"}' \
  --output output.wav

# 语音克隆
curl -X POST http://localhost:8000/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{
    "input": "Have a nice day!",
    "references": [{
      "audio_path": "docs/_static/audio/male-voice.wav",
      "text": "Hey, Adam here."
    }]
  }' \
  --output output_clone.wav

# 流式 TTS
curl -N -X POST http://localhost:8000/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{
    "input": "Hello, how are you?",
    "stream": true,
    "response_format": "pcm"
  }' \
  --output output.pcm
```

**Step 4: 测量性能**

```python
import requests
import time

def measure_tts_latency(text, num_runs=10):
    latencies = []
    for _ in range(num_runs):
        start = time.time()
        resp = requests.post(
            "http://localhost:8000/v1/audio/speech",
            json={"input": text}
        )
        latencies.append(time.time() - start)
    
    print(f"Average latency: {sum(latencies)/len(latencies):.3f}s")
    print(f"Min latency: {min(latencies):.3f}s")
    print(f"Max latency: {max(latencies):.3f}s")

measure_tts_latency("Hello, how are you?")
```

### 3.3 小模型复现: Fish Audio S2-Pro

**Step 1: 下载模型**

```bash
huggingface-cli download fishaudio/s2-pro
```

**Step 2: 启动服务**

```bash
sgl-omni serve \
  --model-path fishaudio/s2-pro \
  --config examples/configs/s2pro_tts.yaml \
  --port 8001
```

**Step 3: 测试功能**

```bash
# 基础 TTS
curl -X POST http://localhost:8001/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{"input": "Hello, how are you?"}' \
  --output output_fish.wav

# 语音克隆
curl -X POST http://localhost:8001/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{
    "input": "Have a nice day!",
    "references": [{
      "audio_path": "docs/_static/audio/male-voice.wav",
      "text": "Hey, Adam here."
    }]
  }' \
  --output output_fish_clone.wav
```

### 3.4 学习要点记录

**架构理解**：
- Pipeline 架构：preprocessing → tts_engine → vocoder
- Scheduler 设计：OmniScheduler (AR) + SimpleScheduler (非AR)
- 数据流：HTTP → Client → Coordinator → Stage → Scheduler → ModelRunner

**关键代码位置**：
- `sglang_omni/pipeline/coordinator.py`: 请求生命周期管理
- `sglang_omni/scheduling/omni_scheduler.py`: AR 调度器
- `sglang_omni/models/qwen3_tts/`: Qwen3-TTS 模型实现

**问题和解决方案**：
- 问题1：CUDA Graph 捕获失败
  - 原因：显存不足
  - 解决：减小 `mem_fraction_static` 或禁用 CUDA Graph

- 问题2：模型加载慢
  - 原因：首次下载模型
  - 解决：提前下载模型到本地

---

## 4. Phase 2: 中等模型复现与优化

### 4.1 Higgs TTS 复现

**显存需求**：
- FP16: ~8GB
- INT8: ~4GB
- 4090 单卡可运行

**Step 1: 下载模型**

```bash
huggingface-cli download bosonai/higgs-audio-v3-tts-4b
```

**Step 2: 启动服务**

```bash
sgl-omni serve \
  --model-path bosonai/higgs-audio-v3-tts-4b \
  --port 8002
```

**Step 3: 测试多码本解码**

```bash
# 测试 TTS
curl -X POST http://localhost:8002/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{"input": "Hello, how are you?"}' \
  --output output_higgs.wav
```

**Step 4: 性能测试**

```python
import requests
import time
import concurrent.futures

def test_concurrent_tts(concurrency, text="Hello, how are you?"):
    def single_request():
        start = time.time()
        resp = requests.post(
            "http://localhost:8002/v1/audio/speech",
            json={"input": text}
        )
        return time.time() - start
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as executor:
        futures = [executor.submit(single_request) for _ in range(concurrency)]
        latencies = [f.result() for f in futures]
    
    print(f"Concurrency: {concurrency}")
    print(f"Average latency: {sum(latencies)/len(latencies):.3f}s")
    print(f"Throughput: {concurrency / sum(latencies):.2f} req/s")

# 测试不同并发度
for c in [1, 2, 4, 8]:
    test_concurrent_tts(c)
```

### 4.2 MOSS-TTS-Local 复现

**显存需求**：
- FP16: ~8GB
- 4090 单卡可运行

**Step 1: 下载模型**

```bash
huggingface-cli download OpenMOSS-Team/MOSS-TTS-Local-Transformer-v1.5
```

**Step 2: 启动服务**

```bash
sgl-omni serve \
  --model-path OpenMOSS-Team/MOSS-TTS-Local-Transformer-v1.5 \
  --port 8003
```

**Step 3: 测试 48kHz 输出**

```bash
curl -X POST http://localhost:8003/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{"input": "Hello, how are you?"}' \
  --output output_moss.wav
```

### 4.3 优化实验

**CUDA Graph 效果测试**：

```bash
# 禁用 CUDA Graph
sgl-omni serve \
  --model-path bosonai/higgs-audio-v3-tts-4b \
  --disable-cuda-graph \
  --port 8004

# 对比性能
python benchmark.py --port 8002 --output results_cg_on.json
python benchmark.py --port 8004 --output results_cg_off.json
```

**Async Decode 效果测试**：

```bash
# 启用 Async Decode
sgl-omni serve \
  --model-path bosonai/higgs-audio-v3-tts-4b \
  --async-decode on \
  --port 8005

# 对比性能
python benchmark.py --port 8002 --output results_async_off.json
python benchmark.py --port 8005 --output results_async_on.json
```

### 4.4 学习要点记录

**多码本解码**：
- Higgs 使用 8 个码本，延迟模式
- 每个 decode step 需要预测 8 个 code
- FeedbackARModelRunner 负责写入/读取反馈

**CUDA Graph 优化**：
- 消除 Python 和 launch overhead
- 需要固定的 batch size
- 使用 Shadow buffers 解决 padding 问题

**Async Decode**：
- 一步前瞻，GPU forward 与 host collect 重叠
- bs=1 走同步路径，bs>=2 走异步路径
- 需要位精确验证

---

## 5. Phase 3: 大模型复现与分布式

### 5.1 Qwen3-Omni TP 配置

**显存需求**：
- FP16: ~60GB
- INT8: ~30GB
- 需要 TP=4 (4×24GB=96GB)

**Step 1: 下载模型**

```bash
huggingface-cli download Qwen/Qwen3-Omni-30B-A3B-Instruct
```

**Step 2: 配置 TP=4**

```bash
# 使用 4 张 GPU
CUDA_VISIBLE_DEVICES=0,1,2,3 sgl-omni serve \
  --model-path Qwen/Qwen3-Omni-30B-A3B-Instruct \
  --tp 4 \
  --port 8006
```

**Step 3: 测试完整管道**

```bash
# 文本输入 → 文本输出
curl -X POST http://localhost:8006/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [{"role": "user", "content": "Hello!"}],
    "modalities": ["text"]
  }'

# 文本输入 → 音频输出
curl -X POST http://localhost:8006/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [{"role": "user", "content": "Hello!"}],
    "modalities": ["text", "audio"]
  }'
```

### 5.2 Stage Fusion 测试

```bash
# 配置 Stage Fusion
cat > config_fused.yaml << EOF
config_cls: Qwen3OmniSpeechColocatedPipelineConfig
model_path: Qwen/Qwen3-Omni-30B-A3B-Instruct
relay_backend: shm
fused_stages:
  - ["thinker", "talker_ar", "code2wav"]
EOF

sgl-omni serve \
  --model-path Qwen/Qwen3-Omni-30B-A3B-Instruct \
  --config config_fused.yaml \
  --tp 4 \
  --port 8007
```

### 5.3 Router 测试

```bash
# 启动多个 worker
CUDA_VISIBLE_DEVICES=0,1,2,3 sgl-omni serve \
  --model-path Qwen/Qwen3-Omni-30B-A3B-Instruct \
  --tp 4 \
  --port 8006 &

CUDA_VISIBLE_DEVICES=4,5,6,7 sgl-omni serve \
  --model-path Qwen/Qwen3-Omni-30B-A3B-Instruct \
  --tp 4 \
  --port 8008 &

# 启动 Router
sgl-omni-router \
  --worker-urls http://localhost:8006,http://localhost:8008 \
  --port 8000
```

### 5.4 Ming-Omni 复现 (如果显存足够)

**显存需求**：
- FP16: ~200GB
- 需要 TP=8 + CPU offload

```bash
# 使用 8 张 GPU + CPU offload
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 sgl-omni serve \
  --model-path inclusionAI/Ming-flash-omni-2.0 \
  --tp 8 \
  --cpu-offload-gb 50 \
  --mem-fraction-static 0.7 \
  --port 8009
```

### 5.5 学习要点记录

**Tensor Parallelism**：
- 将模型权重分片到多个 GPU
- 每个 GPU 计算部分结果，通过 NCCL 通信
- 4090 没有 NVLink，PCIe 通信效率较低

**Stage Fusion**：
- 将相邻阶段融合为单进程
- 减少 ZMQ/Relay 通信开销
- 需要满足约束：相邻、有序、非 TP、单 GPU

**Router**：
- 负载均衡到多个 worker
- 支持 round_robin、least_request、random 策略
- 健康检查和故障转移

---

## 6. Phase 4: 性能优化与对比

### 6.1 性能基准测试

**测试脚本**：

```python
#!/usr/bin/env python3
"""SGLang-Omni 性能基准测试"""

import requests
import time
import json
import concurrent.futures
from typing import List, Dict

def benchmark_tts(
    port: int,
    text: str = "Hello, how are you?",
    concurrency: int = 1,
    num_requests: int = 10
) -> Dict:
    """测试 TTS 性能"""
    
    def single_request() -> float:
        start = time.time()
        resp = requests.post(
            f"http://localhost:{port}/v1/audio/speech",
            json={"input": text}
        )
        resp.raise_for_status()
        return time.time() - start
    
    # 预热
    for _ in range(3):
        single_request()
    
    # 测试
    with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as executor:
        futures = [executor.submit(single_request) for _ in range(num_requests)]
        latencies = [f.result() for f in futures]
    
    return {
        "concurrency": concurrency,
        "num_requests": num_requests,
        "latency_mean": sum(latencies) / len(latencies),
        "latency_min": min(latencies),
        "latency_max": max(latencies),
        "throughput": concurrency / (sum(latencies) / len(latencies))
    }

def run_benchmark(port: int, output_file: str):
    """运行完整基准测试"""
    results = []
    
    for c in [1, 2, 4, 8, 16]:
        print(f"Testing concurrency={c}...")
        result = benchmark_tts(port, concurrency=c)
        results.append(result)
        print(f"  Latency: {result['latency_mean']:.3f}s")
        print(f"  Throughput: {result['throughput']:.2f} req/s")
    
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    return results

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--output", type=str, default="benchmark_results.json")
    args = parser.parse_args()
    
    run_benchmark(args.port, args.output)
```

### 6.2 优化对比

**CUDA Graph 对比**：

```bash
# 测试 CUDA Graph ON
sgl-omni serve --model-path bosonai/higgs-audio-v3-tts-4b --port 8002
python benchmark.py --port 8002 --output results_cg_on.json

# 测试 CUDA Graph OFF
sgl-omni serve --model-path bosonai/higgs-audio-v3-tts-4b --disable-cuda-graph --port 8004
python benchmark.py --port 8004 --output results_cg_off.json

# 对比
python compare_results.py results_cg_on.json results_cg_off.json
```

**Async Decode 对比**：

```bash
# 测试 Async Decode ON
sgl-omni serve --model-path bosonai/higgs-audio-v3-tts-4b --async-decode on --port 8005
python benchmark.py --port 8005 --output results_async_on.json

# 测试 Async Decode OFF
python benchmark.py --port 8002 --output results_async_off.json

# 对比
python compare_results.py results_async_on.json results_async_off.json
```

### 6.3 4090 vs H200 对比

**预期差距**：
- **显存**：4090 (24GB) vs H200 (141GB)，差距 5.9×
- **带宽**：4090 (1008 GB/s) vs H200 (4800 GB/s)，差距 4.8×
- **计算**：4090 (82.6 TFLOPS) vs H200 (989 TFLOPS)，差距 12×

**实际影响**：
- **小模型**：差距不大，主要受带宽限制
- **大模型**：需要 TP，通信开销大
- **并发度**：4090 受显存限制，并发度较低

### 6.4 学习要点记录

**性能优化策略**：
- **CUDA Graph**：消除 Python overhead，适合固定 batch size
- **Async Decode**：GPU forward 与 host collect 重叠
- **Stage Fusion**：减少通信开销
- **批处理**：提高 GPU 利用率

**瓶颈分析**：
- **Compute-bound**：prefill 阶段，受计算能力限制
- **Memory-bound**：decode 阶段，受内存带宽限制
- **Communication-bound**：TP 通信，受互连带宽限制

---

## 7. 学习路径与关键知识点

### 7.1 学习路径

```
Week 1-2: 基础
├── 理解 Pipeline 架构
├── 理解 Scheduler 设计
├── 理解数据流
└── 能跑通 TTS demo

Week 3-4: 进阶
├── 理解多码本解码
├── 理解 CUDA Graph
├── 理解 Async Decode
└── 能进行性能优化

Week 5-6: 高级
├── 理解 Tensor Parallelism
├── 理解 Stage Fusion
├── 理解 Router
└── 能进行分布式部署

Week 7-8: 专家
├── 能分析性能瓶颈
├── 能进行深度优化
├── 能解决复杂问题
└── 能进行面试准备
```

### 7.2 关键知识点

**架构层面**：
- Pipeline 架构：Coordinator → Stage → Scheduler → ModelRunner
- 多调度器设计：OmniScheduler (AR) + SimpleScheduler (非AR)
- 数据面与控制面分离：ZMQ (控制) + Relay (数据)

**优化层面**：
- CUDA Graph：消除 Python overhead
- Async Decode：GPU forward 与 host collect 重叠
- Stage Fusion：减少通信开销
- 拓扑感知传输：LOCAL_OBJECT → CUDA IPC → Relay

**分布式层面**：
- Tensor Parallelism：模型权重分片
- Pipeline Parallelism：模型层分片
- Router：负载均衡和故障转移

### 7.3 面试准备要点

**项目介绍**：
- SGLang-Omni 是基于 SGLang 的全模态模型推理框架
- 核心创新：Composition over Inheritance、多调度器架构、拓扑感知传输
- 性能优化：CUDA Graph、Async Decode、Stage Fusion

**技术深挖**：
- 为什么用 Composition 而不是继承？
- Async Decode 如何保证位精确？
- 拓扑感知传输的三级降级策略？

**问题定位**：
- CUDA Graph WER 退化：padding 问题 → Shadow buffers
- Encoder 批处理退化：资源竞争 → 保持单样本处理
- Vocoder 瓶颈：profiling → per-T CUDA Graph

---

## 8. 常见问题与解决方案

### 8.1 显存不足 (OOM)

**问题**：模型加载或推理时 OOM

**解决方案**：

```bash
# 方案 1: 减小 mem_fraction_static
sgl-omni serve \
  --model-path <model> \
  --mem-fraction-static 0.7 \
  --port 8000

# 方案 2: 使用 INT8 量化
sgl-omni serve \
  --model-path <model> \
  --quantization int8 \
  --port 8000

# 方案 3: 使用 CPU offload
sgl-omni serve \
  --model-path <model> \
  --cpu-offload-gb 10 \
  --port 8000

# 方案 4: 使用 TP
CUDA_VISIBLE_DEVICES=0,1,2,3 sgl-omni serve \
  --model-path <model> \
  --tp 4 \
  --port 8000
```

### 8.2 CUDA Graph 捕获失败

**问题**：CUDA Graph 捕获时 OOM 或失败

**解决方案**：

```bash
# 方案 1: 禁用 CUDA Graph
sgl-omni serve \
  --model-path <model> \
  --disable-cuda-graph \
  --port 8000

# 方案 2: 减小 CUDA Graph batch size
sgl-omni serve \
  --model-path <model> \
  --cuda-graph-bs 1,2,4 \
  --port 8000
```

### 8.3 模型加载慢

**问题**：首次启动时模型加载很慢

**解决方案**：

```bash
# 方案 1: 提前下载模型
huggingface-cli download <model>

# 方案 2: 使用本地模型路径
sgl-omni serve \
  --model-path /path/to/local/model \
  --port 8000

# 方案 3: 使用 Docker 预构建镜像
docker pull lmsysorg/sglang-omni:dev
```

### 8.4 TP 通信效率低

**问题**：4090 使用 PCIe 通信，TP 效率低

**解决方案**：

```bash
# 方案 1: 减小 TP degree
# 使用 TP=2 而不是 TP=4
CUDA_VISIBLE_DEVICES=0,1 sgl-omni serve \
  --model-path <model> \
  --tp 2 \
  --port 8000

# 方案 2: 使用 Stage Fusion
# 将相邻阶段融合，减少通信
sgl-omni serve \
  --model-path <model> \
  --config config_fused.yaml \
  --port 8000

# 方案 3: 使用 CPU offload
# 减少 GPU 显存使用，避免 TP
sgl-omni serve \
  --model-path <model> \
  --cpu-offload-gb 20 \
  --port 8000
```

### 8.5 流式输出延迟高

**问题**：流式输出的首包延迟高

**解决方案**：

```bash
# 方案 1: 启用 Async Decode
sgl-omni serve \
  --model-path <model> \
  --async-decode on \
  --port 8000

# 方案 2: 减小初始 chunk 大小
curl -X POST http://localhost:8000/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{
    "input": "Hello",
    "stream": true,
    "initial_codec_chunk_frames": 1
  }'

# 方案 3: 使用流式 PCM 输出
curl -N -X POST http://localhost:8000/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{
    "input": "Hello",
    "stream": true,
    "response_format": "pcm"
  }'
```

---

## 附录: 关键配置文件

### A.1 Qwen3-TTS 0.6B 配置

```yaml
# examples/configs/qwen3_tts_0_6b.yaml
config_cls: Qwen3TTSPipelineConfig
model_path: Qwen/Qwen3-TTS-12Hz-0.6B-Base
relay_backend: shm
```

### A.2 Higgs TTS 配置

```yaml
# examples/configs/higgs_tts.yaml
config_cls: HiggsTTSPipelineConfig
model_path: bosonai/higgs-audio-v3-tts-4b
relay_backend: shm
```

### A.3 Qwen3-Omni Colocated 配置

```yaml
# examples/configs/qwen3_omni_colocated_h200.yaml
config_cls: Qwen3OmniSpeechColocatedPipelineConfig
model_path: Qwen/Qwen3-Omni-30B-A3B-Instruct
relay_backend: shm
stage_overrides:
  thinker:
    runtime:
      resources:
        total_gpu_memory_fraction: 0.769
  talker_ar:
    runtime:
      resources:
        total_gpu_memory_fraction: 0.123
  code2wav:
    runtime:
      resources:
        total_gpu_memory_fraction: 0.014
```

### A.4 4090 优化配置

```yaml
# 4090 优化配置
config_cls: Qwen3OmniSpeechColocatedPipelineConfig
model_path: Qwen/Qwen3-Omni-30B-A3B-Instruct
relay_backend: shm
stage_overrides:
  thinker:
    runtime:
      resources:
        total_gpu_memory_fraction: 0.6  # 减小以留更多空间
  talker_ar:
    runtime:
      resources:
        total_gpu_memory_fraction: 0.15
  code2wav:
    runtime:
      resources:
        total_gpu_memory_fraction: 0.02
```

---

**文档版本**：v1.0  
**最后更新**：2026-06-27  
**适用硬件**：8× RTX 4090 (24GB)  
**参考项目**：SGLang-Omni
