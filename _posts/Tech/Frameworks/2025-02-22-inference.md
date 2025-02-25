---
title: Inference/Decoding
date: 2025-01-25
categories: [Tech, Frameworks]
tags: [bert, machine learning, NLP]     # TAG names should always be lowercase
math: true
---

### Deocding Strategies

解码参数如下：

```json
{
 "top_k": 10,
 "temperature": 0.95,
 "num_beams": 1,
 "top_p": 0.8,
 "repetition_penalty": 1.5,
 "max_tokens": 30000,
 "message": [
        {
 "content": "你好！",
 "role": "user"
        }
    ]
}
```

- **贪心解码**（Greedy Decoding）：直接选择概率最高的单词。这种方法简单高效，但是可能会导致生成的文本过于单调和重复。
- **随机采样**（Random Sampling）：按照概率分布随机选择一个单词。这种方法可以增加生成的多样性，但是可能会导致生成的文本不连贯和无意义。
- **Beam Search**：维护一个大小为 k 的候选序列集合，每一步从每个候选序列的概率分布中选择概率最高的 k 个单词，然后保留总概率最高的 k 个候选序列。这种方法可以平衡生成的质量和多样性，但是可能会导致生成的文本过于保守和不自然。

#### Top-k

**top-k 采样**的思路是，在每一步，只从概率最高的 k 个单词中进行随机采样，而不考虑其他低概率的单词。这样可以避免采样到一些不合适或不相关的单词，同时也可以保留一些有趣或有创意的单词。

```python
import torch
from labml_nn.sampling import Sampler

# Top-k Sampler
class TopKSampler(Sampler):
    # k is the number of tokens to pick
    # sampler is the sampler to use for the top-k tokens
    # sampler can be any sampler that takes a logits tensor as input and returns a token tensor; e.g. `TemperatureSampler`.
    def __init__(self, k: int, sampler: Sampler):
        self.k = k
        self.sampler = sampler

    # Sample from logits
    def __call__(self, logits: torch.Tensor):
        # New logits filled with −∞; i.e. zero probability
        zeros = logits.new_ones(logits.shape) * float('-inf')
        # Pick the largest k logits and their indices
        values, indices = torch.topk(logits, self.k, dim=-1)
        # Set the values of the top-k selected indices to actual logits.
        # Logits of other tokens remain −∞
        zeros.scatter_(-1, indices, values)
        # Sample from the top-k logits with the specified sampler.
        return self.sampler(zeros)
```

一般来说，**k 越大，生成的多样性越高，但是生成的质量越低；k 越小，生成的质量越高，但是生成的多样性越低**。



#### Top-p(nucleus sampling)

top-p 采样的思路是，在每一步，**只从累积概率超过某个阈值 p 的最小单词集合中进行随机采样，而不考虑其他低概率的单词**。这种方法也被称为**核采样（nucleus sampling）**，因为它只关注概率分布的核心部分，而忽略了尾部部分。例如，如果 p=0.9，那么我们只从累积概率达到 0.9 的最小单词集合中选择一个单词，而不考虑其他累积概率小于 0.9 的单词。这样可以避免采样到一些不合适或不相关的单词，同时也可以保留一些有趣或有创意的单词。

```python
import torch
from torch import nn

from labml_nn.sampling import Sampler

class NucleusSampler(Sampler):
    """
    ## Nucleus Sampler
    """
    def __init__(self, p: float, sampler: Sampler):
        """
        :param p: is the sum of probabilities of tokens to pick $p$
        :param sampler: is the sampler to use for the selected tokens
        """
        self.p = p
        self.sampler = sampler
        # Softmax to compute $P(x_i | x_{1:i-1})$ from the logits
        self.softmax = nn.Softmax(dim=-1)

    def __call__(self, logits: torch.Tensor):
        """
        Sample from logits with Nucleus Sampling
        """

        # Get probabilities $P(x_i | x_{1:i-1})$
        probs = self.softmax(logits)

        # Sort probabilities in descending order
        sorted_probs, indices = torch.sort(probs, dim=-1, descending=True)

        # Get the cumulative sum of probabilities in the sorted order
        cum_sum_probs = torch.cumsum(sorted_probs, dim=-1)

        # Find the cumulative sums less than $p$.
        nucleus = cum_sum_probs < self.p

        # Prepend ones so that we add one token after the minimum number
        # of tokens with cumulative probability less that $p$.
        nucleus = torch.cat([nucleus.new_ones(nucleus.shape[:-1] + (1,)), nucleus[..., :-1]], dim=-1)

        # Get log probabilities and mask out the non-nucleus
        sorted_log_probs = torch.log(sorted_probs)
        sorted_log_probs[~nucleus] = float('-inf')

        # Sample from the sampler
        sampled_sorted_indexes = self.sampler(sorted_log_probs)

        # Get the actual indexes
        res = indices.gather(-1, sampled_sorted_indexes.unsqueeze(-1))

        #
        return res.squeeze(-1)
```



#### Beam search

**计算所有当前 Beam 候选的下一个 Token 概率**

**计算扩展序列的累积概率**

对于 每个 beam计算当前 **beam_score + log(新 token 概率)**, 生成所有可能的新候选序列。

**保留前 `K` 个最高概率的序列**, 计算所有扩展序列的 `log_prob` 并排序。

选取 **Top K 个最优的 Beam**，作为下一步的候选序列。



### KV Cache

When decoding, the transformer will recompute the exact same key-query similarities over and over.

At time t query t will interact with all keys at 1, 2,..., t

queries until t - 1 are not needed

Similarly, all past values should be retained in the cache to compute final attention output

The decoder-only architecture is causal.



### Flash Attention

![flash](/assets/images/flash-att.png)

Accelerate attetion computation druing inference and training.

Given long seq, GPUs spend more time in writing/reading data (from HBM to SRAM), insted of matrix multiplication (on SRAM)

通过减少writing/reading time来提高GPU利用率

1. Tiling: Restructure the algorithm to load block by block from HBM to SRAM to compute attetion; Load inputs from HBM to SRAM by blocks,; compute attetion within that block on chip; update output in HBM by scaling. Softmax can be exactly decomposed into different blocks of the sequence by scaling.
2. Recomputation: attention maps should be stored to compute gradients during back-propagation, which take memory O(n^2).  To reduce memory cost, only store the scaling factors for each block instead of storing the whole attetion matrix during forward pass. During back-propagation, recompute the attention using SRAM.



### Paged Attention

解决长序列解码时KV cache占用GPU内存过大的问题

这是一种受操作系统中虚拟内存和分页经典思想启发的注意力算法。与传统的注意力算法不同，**PagedAttention 允许在非连续的内存空间中存储连续的键和值**。具体来说，**PagedAttention 将每个序列的 KV 缓存划分为块，每个块包含固定数量 token 的键和值**。在注意力计算期间，PagedAttention 内核可以有效地识别和获取这些块。

PagedAttention 算法能让 KV 块存储在非相邻连续的物理内存中，从而让 vLLM 实现更为灵活的分页内存管理。

![paged](/assets/images/paged-att.png)



### vllm 推理框架

**vLLM是一个大模型推理服务框架**，声称

- 最牛的serving 吞吐量
- **PagedAttention**对kv cache的有效管理
- 传入请求的**continus batching**，而不是static batching
- 高性能CUDA kernel
- 流行的HuggingFace模型无缝集成
- 有各种decoder算法的高吞吐量服务，包括parallel sampling和beam search等
- tensor parallel
- 兼容OpenAI的API服务器

vllm优化推理技术：

1. 连续批处理：简单来说，**一旦一个batch中的某个seq完成生成，发射了一个end-of-seq token，就可以在其位置插入新的seq继续生成token**，从而达到比static batching更高的GPU利用率。
2. Paged Attetion：KV缓存的分页高效管理



### 蒸馏

缩小模型大小的另一种方法是通过称为蒸馏的过程**将其知识转移到较小的模型**。此过程涉及训练较小的模型（称为学生）来模仿较大模型（教师）的行为。

DistilBERT

![distill](/assets/images/distill.png)



### Speculative Decoding

![spec-samping](/assets/images/spec-sampling.png)

