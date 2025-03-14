---
title: Distributed training frameworks
date: 2025-02-21
categories: [Tech, Frameworks]
tags: [bert, machine learning, NLP, training, distributed training]     # TAG names should always be lowercase
math: true
---

什么是分布式训练？

分布式训练大语言模型（LLM, 如 LLaMA-3 65B）需要**多个 GPU 共同协作**，因为单张 GPU 无法存储和计算如此庞大的参数量。核心思路是**将模型、数据、优化器的计算任务拆分到多个 GPU 进行并行计算**。

### 显存分析

分析tranformer参数量、计算量、中间激活: [blog](https://zhuanlan.zhihu.com/p/624740065)

大模型也分为**不同的规格**，一般模型的规格会体现在模型的名称上，例如 LLaMA2-13b，13b 就是其模型参数量的大小，意思是 130亿的参数量。大模型的文件大小与其参数量有关，通常大模型是以半精度存储的， Xb 的模型文件大概是 2X GB多一些，例如 13b 的模型文件大小大约是 27GB 左右。

1 billion bytes 约等于 1 GB

一般来说**推理模型需要的显存约等于模型文件大小，全参训练需要的显存约为推理所需显存的三倍到四倍**，正常来说，在不量化的情况下4张 v100 显卡推理 65b 的模型都会有一些吃力，无法进行训练，需要通过 **LoRA 或者\**\**QLoRA** 采用低秩分解的方式才可以训练。



在一次训练迭代中，模型参数（或梯度）占用的显存大小只与模型参数量和参数数据类型有关，与输入数据的大小是没有关系的。优化器状态占用的显存大小也是一样，与优化器类型有关，与模型参数量有关，但与输入数据的大小无关。而**中间激活值与输入数据的大小（批次大小 b 和序列长度 s ）是成正相关的**，随着批次大小 b 和序列长度 s 的增大，中间激活占用的显存会同步增大。当我们训练神经网络遇到显存不足OOM（Out Of Memory）问题时，通常会尝试减小批次大小来避免显存不足的问题，这种方式减少的其实是中间激活占用的显存，而不是模型参数、梯度和优化器的显存。

以GPT3-175B为例，我们来直观地对比下模型参数与中间激活的显存大小。GPT3的模型配置如下。我们假设采用混合精度训练，模型参数和中间激活都采用float16数据类型，每个元素占2个bytes。

| 模型名 | 参数量 | 层数 | 隐藏维度 | 注意力头数 |
| ------ | ------ | ---- | -------- | ---------- |
| GPT3   | 175B   | 96   | 12288    | 96         |

GPT3的模型参数量为175B，占用的显存大小为 2×175×109bytes=350GB 。GPT3模型需要占用350GB的显存。

GPT3的序列长度 s 为 2048 。对比不同的批次大小 b 占用的中间激活：

当 b=1 时，中间激活占用显存为 275GB ，大约是模型参数显存的0.79倍。

当 b=64 时，中间激活占用显存为17.6TB ，大约是模型参数显存的50倍。

当 b=128 时，中间激活占用显存为35.3TB ，大约是模型参数显存的101倍。

可以看到随着批次大小 b 的增大，中间激活占用的显存远远超过了模型参数显存。通常会采用**激活重计算**技术来减少中间激活，理论上可以将中间激活显存从 O(n) 减少到 O(n) ，代价是增加了一次额外前向计算的时间，本质上是“时间换空间”。



**训练过程中如何解决OOM的问题**：

- 减小batch size：减少中间激活占用的显存，对模型本身占用的显存没有影响
- 混合精度训练：训练时使用不同精度的树数值(FP32, FP16), 前向传播和激活值使用FP16，梯度，权重更新使用FP32
- 梯度检查点：在前向传播时丢弃部分激活值，在反向传播时重新计算
- PEFT



### 数据并行

**方法**：每个 GPU 训练 **相同的模型**，但使用 **不同的 mini-batch**。

**核心框架**：PyTorch DDP (`torch.nn.parallel.DistributedDataParallel`)。

**适用场景**：中小规模模型（< 13B），数据可复制到多个 GPU。

**缺点**：对于超大模型（>65B），**单个 GPU 无法存储完整模型**，DP 不能单独使用。

![dist](/assets/images/dist.png)

在数据并行训练中，数据集被分割成几个碎片，每个碎片被分配到一个设备上。这相当于**沿批次（Batch）维度对训练过程进行并行化**。每个设备将持有一个完整的模型副本，并在分配的数据集碎片上进行训练。在反向传播之后，模型的梯度将被全部减少，以便在不同设备上的模型参数能够保持同步。典型的数据并行实现：PyTorch DDP。

```python
import torch
import torch.distributed dist
import torch.multiprocessing as mp
import torch.nn as nn
import torch.optim as optim
from torch.nn.parallel import DistributedDataParallel as DDP

def init_distributed_mode(args):
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        args.rank = int(os.environ["RANK"])
        args.world_size = int(os.environ['WORLD_SIZE'])
        args.gpu = int(os.environ['LOCAL_RANK'])
    elif 'SLURM_PROCID' in os.environ:
        args.rank = int(os.environ['SLURM_PROCID'])
        args.gpu = args.rank % torch.cuda.device_count()
    else:
        print('Not using distributed mode')
        args.distributed = False
        return

    args.distributed = True

    torch.cuda.set_device(args.gpu)
    args.dist_backend = 'nccl'
    print('| distributed init (rank {}, word {}): {}'.format(
        args.rank, args.world_size, args.dist_url), flush=True)
    torch.distributed.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                         world_size=args.world_size, rank=args.rank)
    torch.distributed.barrier()
    setup_for_distributed(args.rank == 0) 

def example(rank, world_size):
    # create default process group
    dist.init_process_group("gloo", rank=rank, world_size=world_size)
    # create local model
    model = nn.Linear(10, 10).to(rank)
    # construct DDP model
    ddp_model = DDP(model, device_ids=[rank])
    # define loss function and optimizer
    loss_fn = nn.MSELoss()
    optimizer = optim.SGD(ddp_model.parameters(), lr=0.001)

    # forward pass
    outputs = ddp_model(torch.randn(20, 10).to(rank))
    labels = torch.randn(20, 10).to(rank)
    # backward pass
    loss_fn(outputs, labels).backward()
    # update parameters
    optimizer.step()

def main():
    world_size = 2
    mp.spawn(example,
        args=(world_size,),
        nprocs=world_size,
        join=True)

if __name__=="__main__":
    # Environment variables which need to be
    # set when using c10d's default "env"
    # initialization mode.
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29500"
    main()
```

DP数据传输过程：

1. 前向传播得到的输出结果gather到主cuda计算loss
2. scatter上述loss到各个cuda
3. 各个cuda反向传播计算得到梯度后gather到主cuda后，主cuda的模型参数被更新。
4. 主cuda将模型参数broadcast到其它cuda设备上，至此，完成权重参数值的同步。

综上，DP大概是有4次输出传输。

DDP数据传输过程：

1. 前向传播的输出和loss的计算都是在每个cuda独立计算的，梯度all-reduce到所有的CUDA(传输梯度)，这样初始参数相同，para.grad也相同，反向传播后参数就还是保持一致的，其他没有数据传输了。



### 模型并行

可分为张量并行和流水线并行

**核心框架**：

- **Megatron-LM**
- **DeepSpeed Tensor Slicing**
- **DeepSpeed Pipeline Parallel**

**适用场景**：超大模型（如 GPT-4, LLaMA-3 65B），需要多个 GPU **共同存储参数**。



### DeepSpeed

![zero](/assets/images/zero.png)

**用 3D 并行化实现万亿参数模型训练**：DeepSpeed 实现了三种并行方法的灵活组合：ZeRO 支持的数据并行，流水线并行和张量切片模型并行。3D 并行性适应了不同工作负载的需求，以支持具有**万亿**参数的**超大型模型**，同时实现了近乎完美的显存扩展性和吞吐量扩展效率。

`ZeRO-0`：禁用所有类型的分片，仅使用 DeepSpeed 作为 DDP (Distributed Data Parallel)

`ZeRO-1`：分割Optimizer States，减少了4倍的内存，通信容量与数据并行性相同

`ZeRO-2`：分割Optimizer States与Gradients，8x内存减少，通信容量与数据并行性相同

`ZeRO-3`：分割Optimizer States、Gradients与Parameters，内存减少与数据并行度和复杂度成线性关系。

`ZeRO-Infinity`是ZeRO-3的拓展。允许通过使用 NVMe 固态硬盘扩展 GPU 和 CPU 内存来训练大型模型。ZeRO-Infinity 需要启用 ZeRO-3。

在deepspeed中通过zero_optimization.stage=0/1/2/3 设置，

```json
{
  "train_batch_size": 16,
  "fp16": {
    "enabled": true
  },
  "zero_optimization": {
    "stage": 3,
    "offload_optimizer": {
      "device": "cpu"
    },
    "offload_param": {
      "device": "cpu"
    }
  },
  "gradient_accumulation_steps": 4,
  "gradient_checkpointing": true
}
```



```python
import os
import torch
import deepspeed
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from torch.utils.data import DataLoader
from torch.optim import AdamW

# ---- Configuration ---- #
MODEL_NAME = "meta-llama/Llama-2-7b-hf"  # Change for different models
DATASET_NAME = "tatsu-lab/alpaca"
BATCH_SIZE = 4  # Adjust based on available GPU memory
LR = 3e-5  # Learning rate
NUM_EPOCHS = 3
GRAD_ACCUM_STEPS = 4
OUTPUT_DIR = "./output"

# ---- Initialize DeepSpeed ---- #
deepspeed.init_distributed()

# ---- Load Model & Tokenizer ---- #
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.float16)

# ---- Wrap Model with DeepSpeed ---- #
ds_config = "ds_config.json"
model, optimizer, _, _ = deepspeed.initialize(
    model=model,
    model_parameters=model.parameters(),
    optimizer=AdamW(model.parameters(), lr=LR),
    config_params=ds_config
)

# ---- Load Dataset ---- #
dataset = load_dataset(DATASET_NAME)

# Tokenization Function
def tokenize_function(examples):
    return tokenizer(examples["instruction"], truncation=True, padding="max_length", max_length=512)

# Apply tokenization
dataset = dataset.map(tokenize_function, batched=True)
dataset.set_format(type="torch", columns=["input_ids", "attention_mask"])

# Create DataLoader
train_loader = DataLoader(dataset["train"], batch_size=BATCH_SIZE)

# ---- Training Loop ---- #
for epoch in range(NUM_EPOCHS):
    model.train()
    for step, batch in enumerate(train_loader):
        inputs = {k: v.cuda() for k, v in batch.items()}

        outputs = model(**inputs)
        loss = outputs.loss

        model.backward(loss)  # DeepSpeed Backward
        model.step()  # DeepSpeed Optimizer Step

        if step % 10 == 0:
            print(f"Epoch {epoch} Step {step} Loss: {loss.item()}")

# ---- Save Model ---- #
model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)

```



### MoE并行



