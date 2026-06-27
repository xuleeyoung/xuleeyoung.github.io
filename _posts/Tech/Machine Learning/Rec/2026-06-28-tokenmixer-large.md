---
title: TokenMixer-Large
date: 2026-06-28
categories: [Tech, Machine Learning]
tags: [rec, machine learning]
description: TokenMixer-Large技术调研：字节跳动将RankMixer扩展至7B/15B参数的超大规模排序模型架构
math: true
---

## 背景与动机

### RankMixer的成功与局限

RankMixer作为字节跳动在排序模型领域的重要创新，首次将Token混合器（TokenMixer）架构成功应用于工业推荐系统。该架构通过模拟Transformer中的自注意力机制，实现了对用户行为序列的有效建模。然而，当团队尝试将RankMixer扩展至超大规模参数（7B/15B）时，遇到了四个关键瓶颈。

### 四大核心瓶颈

#### 1. 残差设计不合理

RankMixer采用标准的残差连接方式，将混合前后的Token直接相加：

$$
\mathbf{y} = \mathbf{x} + \text{Mixer}(\mathbf{x})
$$

这种设计在浅层模型中表现良好，但当模型层数增加时，混合后的Token与原始Token的语义空间发生错位，导致深层网络的梯度传递效率低下。

#### 2. 模型架构不纯净

原始RankMixer保留了LHUC（Learning Hidden Unit Contributions）、DCNv2等历史算子，形成了一种"碎片化"架构。这种混合设计虽然继承了历史经验，但降低了模型的计算效率（MFU，Model FLOPs Utilization）。

#### 3. 深层模型梯度更新不足

RankMixer仅使用2层结构，缺乏针对深层模型的设计。当扩展至数十层时，梯度消失问题变得尤为严重，浅层网络难以获得有效的监督信号。

#### 4. MoE稀疏化不充分

RankMixer采用DTSI（Dual Training Sparse Inference）范式，仅在推理阶段稀疏化，训练阶段仍是密集计算。这无法从根本上降低超大规模模型的训练成本。同时，ReLU-MoE存在激活动态性问题，导致负载均衡不稳定。

---

## 核心架构

### 1. Mixing & Reverting操作

针对残差设计不合理的问题，TokenMixer-Large引入了**混合-还原（Mixing & Reverting）**机制：

```
原始Token → Mixing → 混合Token → Reverting → 还原Token + 新Token
```

数学表达式为：

$$
\mathbf{x}' = \text{Concat}(\mathbf{x}, \text{Mixer}(\mathbf{x}))
$$

$$
\mathbf{y} = \mathbf{x} + \text{Revert}(\mathbf{x}')
$$

其中 `Revert` 操作确保输出Token数量与输入保持一致（当 $T' \neq T$ 时），通过可学习的投影矩阵实现：

$$
\text{Revert}(\mathbf{x}') = \mathbf{x}' \cdot \mathbf{W}_{revert}
$$

**关键优势**：
- 保留了原始Token的语义信息
- 混合后的Token获得增强表达
- 兼容不同长度的序列输入

### 2. 层间残差连接 + 辅助损失

为了解决深层模型的梯度消失问题，TokenMixer-Large设计了双管齐下的策略：

#### 2.1 Inter-layer Residual Connection

跨层直接传递信息，形成"高速公路"结构：

$$
\mathbf{h}_l^{(k)} = \mathbf{h}_{l-1}^{(k)} + \alpha \cdot \text{Block}_l(\mathbf{h}_{l-1}^{(k)})
$$

其中 $\alpha$ 为可学习的缩放因子，$k$ 表示第 $k$ 个专家。

#### 2.2 Auxiliary Loss

引入辅助损失函数，帮助浅层网络学习：

$$
\mathcal{L}_{aux} = \lambda \cdot \sum_{l=1}^{L-1} \text{BCE}(\hat{\mathbf{y}}, \text{Head}_l(\mathbf{h}_l))
$$

总损失为：

$$
\mathcal{L}_{total} = \mathcal{L}_{main} + \beta \cdot \mathcal{L}_{aux}
$$

### 3. Per-token-SwiGLU

将标准FFN升级为SwiGLU激活函数：

```python
class SwiGLU(nn.Module):
    def __init__(self, d_model, d_ffn):
        super().__init__()
        self.w1 = nn.Linear(d_model, d_ffn, bias=False)
        self.w2 = nn.Linear(d_ffn, d_model, bias=False)
        self.w3 = nn.Linear(d_model, d_ffn, bias=False)
    
    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))
```

SwiGLU的数学表达式：

$$
\text{SwiGLU}(x) = x \cdot \sigma(x) \odot \text{GLU}(x)
$$

其中 $\sigma$ 为Sigmoid函数，$\odot$ 表示逐元素乘法。

### 4. Sparse Per-Token MoE

从DTSI范式升级为"稀疏训练+稀疏推理"的完整稀疏化方案：

#### 4.1 激活动态性优化

传统ReLU-MoE的问题：

$$
\text{ReLU}(x) = \max(0, x)
$$

ReLU的硬阈值导致激活模式不稳定。TokenMixer-Large采用Softmax-based路由：

$$
P(e_k | \mathbf{x}_i) = \text{Softmax}\left(\frac{\mathbf{g}_k^T \mathbf{x}_i}{\sqrt{d_k}}\right)
$$

#### 4.2 负载均衡正则

$$
\mathcal{L}_{load} = \alpha \cdot \sum_{k=1}^{K} \left| \frac{n_k}{N} - \frac{1}{K} \right|
$$

#### 4.3 工程优化

- **FP8量化训练**：降低30%显存占用
- **Token并行**：支持超长序列处理
- **动态Top-K**：根据输入动态调整激活专家数

### 5. Down-matrix小初始化策略

对于深层模型的收敛问题，采用Down-matrix小初始化：

```python
class TokenMixerBlock(nn.Module):
    def __init__(self, d_model, n_layers):
        super().__init__()
        self.blocks = nn.ModuleList([MixerLayer(d_model) for _ in range(n_layers)])
        self.down_proj = nn.Linear(d_model, d_model)
        
        # 小初始化策略
        nn.init.normal_(self.down_proj.weight, std=0.02)
```

初始化时将Down-projection权重的标准差设置较小，有助于深层模型稳定训练。

---

## Scaling Law

### 离线实验结果

在**抖音电商场景**进行参数扩展实验：

| 模型规模 | 参数量 | AUC提升 | 收敛步数 |
|:--------:|:------:|:-------:|:--------:|
| Base | 1B | 基准 | 100K |
| Large | 7B | +1.2% | 150K |
| XLarge | 15B | +2.1% | 200K |

实验展示了清晰的对数线性扩展趋势：

$$
\text{AUC} = \alpha \cdot \log(\text{Params}) + \beta
$$

### 线上推理模型

在**抖音广告场景**，线上推理模型扩展至7B参数，通过知识蒸馏技术压缩模型体积。

---

## 工业部署

### 业务指标提升

| 场景 | 核心指标 | 提升幅度 |
|:----:|:--------:|:--------:|
| 抖音电商 | 订单量 | **+1.66%** |
| 抖音电商 | 人均GMV | **+2.98%** |
| 抖音广告 | ADSS | **+2.0%** |
| 抖音直播 | 营收 | **+1.4%** |

### 系统效率

- **MFU提升至60%**：相比RankMixer提升显著
- **推理延迟**：通过FlashAttention优化，控制在10ms以内
- **显存占用**：FP8量化后降低30%

---

## 与RankMixer的对比

### 架构改进对比

| 特性 | RankMixer | TokenMixer-Large |
|:----:|:----------:|:----------------:|
| **残差设计** | 直接相加 | Mixing & Reverting |
| **模型架构** | 含LHUC、DCNv2等 | 纯净统一架构 |
| **网络深度** | 2层 | 数十层 |
| **训练范式** | DTSI（密集训练） | 稀疏训练+稀疏推理 |
| **MoE激活** | ReLU-MoE | Softmax-based MoE |
| **FFN激活** | GELU | SwiGLU |
| **参数量级** | 1B | 7B / 15B |
| **MFU** | ~45% | ~60% |

### 核心差异总结

1. **架构纯粹性**：TokenMixer-Large移除了历史碎片化算子，采用统一的Mixer架构
2. **稀疏化完整性**：从仅推理稀疏升级为训练推理双稀疏
3. **深度扩展能力**：通过残差连接和辅助损失，解决了深层训练难题
4. **工程优化**：FP8量化、Token并行等系统级优化

---

## 技术洞见

### 1. 语义对齐的重要性

Mixing & Reverting机制的本质是解决**语义空间对齐**问题。在深层网络中，混合操作会改变Token的语义分布，直接残差连接会导致信息混淆。通过显式的还原操作，可以有效保持语义一致性。

### 2. 稀疏化是Scaling的关键

从DTSI到完整稀疏化的升级，证明了**训练侧稀疏**的重要性。单纯推理稀疏无法根本降低训练成本，只有稀疏训练+稀疏推理才能支撑超大规模模型。

### 3. 辅助监督的必要性

深层排序模型中，浅层网络的监督信号衰减严重。辅助损失提供额外的梯度路径，有助于浅层学习用户行为的多层次模式。

### 4. 工程与算法的协同

TokenMixer-Large的成功离不开FP8量化、Token并行等工程优化。这提示我们：**工业级大模型的突破需要算法与系统的协同设计**。

---

## 参考文献

1. **TokenMixer-Large: Scaling Up Large Ranking Models in Industrial Recommenders**  
   ByteDance, arXiv:2602.06563, February 2026  
   https://arxiv.org/abs/2602.06563

2. **RankMixer: Token Mixer for Industrial Recommenders**  
   ByteDance, 2024

3. **Deep & Cross Network for Ad Click Predictions** (DCNv2)  
   Wang et al., arXiv:2008.13535, 2020

4. **LHUC: Learning Hidden Unit Contributions**  
   Staff et al., Interspeech, 2020

5. **SwiGLU: Swish-Gated Linear Unit**  
   Shazeer et al., arXiv:2002.05202, 2020

6. **FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness**  
   Dao et al., NeurIPS 2022

7. **Mixture-of-Experts with Softmax Router**  
   Shazeer et al., 2017

---

*本文为技术调研笔记，内容基于arXiv论文2602.06563整理，细节请以原论文为准。*
