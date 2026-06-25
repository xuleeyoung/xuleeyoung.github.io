---
title: RankMixer
date: 2026-06-24
categories: [Tech, Machine Learning]
tags: [rec, machine learning]     # TAG names should always be lowercase
description: RankMixer技术调研：字节跳动工业级推荐排序模型的规模化扩展架构
math: true
---

## 背景与动机

推荐系统的排序模型（Ranking Model）是信息分发链路的核心环节。随着大语言模型（LLM）在参数量扩展上取得巨大成功，业界也开始探索推荐系统领域的 Scaling Law。然而，工业级推荐系统的规模化扩展面临两个独特挑战：

1. **严苛的延迟与 QPS 约束**：与自然语言处理或计算机视觉任务不同，在线推荐必须在毫秒级延迟内完成对大量候选的排序，计算预算极为紧张。
2. **GPU 利用率极低**：现有排序模型中大量人工设计的特征交叉模块（如 DCN、CIN、LHUC 等）继承自 CPU 时代，核心算子多为内存受限型（memory-bound），无法充分利用现代 GPU 的并行计算能力，模型浮点运算利用率（Model FLOPs Utilization, MFU）通常仅为个位数百分比。

传统排序模型的典型架构往往堆叠多种异构模块（DCNv2、LHUC、各类手工交叉层），每个模块结构不同、算子碎片化，导致：
- GPU 无法高效执行大规模矩阵运算，大量算力浪费在零碎的小算子上
- 参数量难以扩展：CPU 时代模型的计算成本与参数量大致成正比，扩展参数量意味着直接增加推理延迟
- 不同特征子空间的信息被共享参数混合，高频特征主导梯度，长尾特征信号被淹没

在此背景下，字节跳动算法团队提出了 **RankMixer**——一种面向硬件感知（hardware-aware）设计的全新排序模型架构，旨在构建统一且可扩展的特征交互范式。

> 论文：[RankMixer: Scaling Up Ranking Models in Industrial Recommenders](https://arxiv.org/abs/2507.15551)（arXiv:2507.15551, ByteDance, 2025）

![RankMixer 模型架构](/assets/images/rankmixer.png)

---

## 核心架构

RankMixer 采用类似 Transformer 的堆叠结构，数据流经过以下四个核心模块：

### 1. 特征 Token 化（Automatic Feature Tokenization）

推荐系统的输入是高度异构的特征集合：用户画像（年龄、城市）、物品属性（标签、类目）、行为序列（历史点击、购买记录）、交叉统计特征等，维度和语义各不相同。

传统做法是将所有特征 embedding 直接拼接，导致计算碎片化（大量不同维度的小矩阵运算），GPU 效率极低。RankMixer 提出三步 Token 化流程：

1. **语义分组**：按语义将特征分为若干组（如用户组、物品组、行为组、交叉组），组内特征语义相近，交互价值高。
2. **组内拼接 + 切分**：将同一组的 embedding 拼接后，按固定维度 $D$ 切分为若干 Token。若某组拼接后维度为 $d_g$，则生成 $\lceil d_g / D \rceil$ 个 Token。
3. **统一映射**：将各组 Token 通过独立的线性投影映射到统一的隐层维度 $H$，并附加可学习的位置编码。

最终输出 $T$ 个维度对齐的特征 Token：$\mathbf{S} = [s_1, s_2, \dots, s_T] \in \mathbb{R}^{T \times H}$。

这一步的关键价值在于：
- 消除计算碎片化，将异构特征转化为规整的矩阵运算，为后续高效并行计算奠定基础
- 按语义分组保留了不同特征空间的异质性，避免语义信息被混合稀释

### 2. 多头 Token Mixing（Multi-head Token Mixing）

这是 RankMixer 最核心的创新之一——用**无参数**的操作替代 Transformer 中的自注意力（Self-Attention），实现跨 Token 的信息交互。

具体步骤如下：

1. **Head Splitting**：将每个 Token 向量 $s_i \in \mathbb{R}^H$ 等分为 $K$ 个子头，每个头维度为 $H/K$。
2. **Cross-Token Shuffling**：将所有 Token 的同一位置子头拼接为矩阵 $M_k \in \mathbb{R}^{T \times H/K}$，即沿 Token 维度拼接同一 head 的信息。
3. **Fusion**：对 $M_k$ 施加 Linear + GELU 变换，得到混合后的 $M'_k$，再拼接所有 head 并通过投影层输出。

```python
# Token Mixing 伪代码
def token_mixing(S, K):
    # S: (T, H), K 个头
    T, H = S.shape
    head_dim = H // K

    # Step 1: reshape 为 (T, K, H/K)
    heads = S.reshape(T, K, head_dim)

    # Step 2: 转置为 (K, T, H/K)，沿 Token 维度拼接
    # 即对每个 head_k，把所有 token 的信息放在一起
    shuffled = heads.permute(1, 0, 2)  # (K, T, H/K)

    # Step 3: 对每个 head 做线性变换 + 激活
    mixed = []
    for k in range(K):
        mk = shuffled[k]           # (T, H/K)
        mk = F.gelu(Linear(mk))   # Linear + GELU
        mixed.append(mk)

    # 拼接所有 head 并投影
    output = torch.cat(mixed, dim=-1)  # (T, H)
    output = Linear(output)             # 投影层
    return output
```

**为什么 Token Mixing 优于 Self-Attention？**

| 维度 | Self-Attention | Token Mixing |
|------|---------------|-------------|
| 参数量 | $O(H^2)$（Q/K/V/O 矩阵） | 仅投影层参数，Shuffling 本身无参数 |
| 计算复杂度 | $O(T^2 \cdot H)$（Attention Matrix） | $O(T \cdot H)$（矩阵乘法） |
| 内存访问 | 需要存储 $T \times T$ 的注意力权重矩阵 | 无中间注意力矩阵 |
| GPU 适配 | 注意力计算涉及不规则内存访问 | 全部为规整的矩阵乘法，Tensor Core 友好 |
| 推荐场景效果 | 注意力权重在异构特征上难以学习 | 无参数 shuffling 天然适合异构特征交互 |

关键洞察：在推荐场景中，不同特征 Token 来自完全不同的语义空间（如"用户年龄"和"视频标签"），Self-Attention 试图在它们之间学习注意力权重，但这类跨空间的相似度信号本身就不稳定。Token Mixing 通过简单的 reshape + 线性变换实现信息混合，避免了在异构空间中建模注意力的困难。

### 3. Per-Token FFN（逐 Token 前馈网络）

传统 Transformer 中，所有 Token 共享同一个 FFN（Feed-Forward Network）。但在推荐场景中，不同特征 Token 代表完全不同的语义子空间，共享参数会导致高频特征（如热门视频 ID）主导梯度，长尾特征信号被稀释。

RankMixer 引入 **Per-Token FFN**：每个 Token 拥有独立参数的前馈网络。

$$
v_t = f_{t,2}^{\text{pffn}}\Big(\text{Gelu}\big(f_{t,1}^{\text{pffn}}(s_t)\big)\Big)
$$

其中 $f_{t,i}^{\text{pffn}}(x) = x W_{t,i}^{\text{pffn}} + b_{t,i}^{\text{pffn}}$，$W_{t,1}^{\text{pffn}} \in \mathbb{R}^{D \times kD}$，$W_{t,2}^{\text{pffn}} \in \mathbb{R}^{kD \times D}$，$k$ 为隐藏层维度缩放系数。

整体表示为：

$$
v_1, v_2, \dots, v_T = \text{PFFN}(s_1, s_2, \dots, s_T)
$$

**Per-Token FFN 与传统设计的本质区别：**

| 设计 | 输入 | 参数 |
|------|------|------|
| Transformer FFN | 所有 Token 共享同一 FFN | 参数共享 |
| MMoE | 所有专家共享同一输入 | 多专家、同一输入 |
| **Per-Token FFN** | **每个 Token 独立 FFN** | **输入和参数同时拆分** |

Per-Token FFN 在**计算复杂度不变**的前提下，通过引入更多参数增强了模型对不同特征子空间的差异化建模能力。

### 4. 稀疏 MoE 扩展（Sparse MoE in RankMixer）

为进一步提升模型的投入产出比（ROI），RankMixer 将 Per-Token FFN 扩展为 Sparse Mixture-of-Experts（MoE）结构，在计算开销基本不变的情况下显著提升模型容量。

但直接将标准 Sparse-MoE 应用于 RankMixer 会出现性能退化，原因有二：
1. **均匀 Top-K 路由问题**：传统 Top-K 对所有 Token 一视同仁，浪费了计算预算在低信息量 Token 上
2. **专家训练不足**：Per-Token FFN 已使参数随 Token 数量线性增长，再叠加 MoE 导致专家数量爆炸，路由严重不均衡

RankMixer 采用两种互补策略解决：

**ReLU 路由**：将 Top-K + Softmax 替换为 ReLU 门控 + 自适应 $\ell_1$ 正则：

$$
G_{i,j} = \text{ReLU}(h(s_i)_j), \quad v_i = \sum_{j=1}^{N_e} G_{i,j} \cdot e_{i,j}(s_i)
$$

ReLU 路由为信息量大的 Token 自动激活更多专家，低信息量 Token 则只激活少量专家，实现自适应的稀疏激活。

**Dense-Training / Sparse-Inference（DTSI-MoE）**：训练时使用两个路由器 $h_{\text{train}}$ 和 $h_{\text{infer}}$，$\ell_1$ 正则仅作用于 $h_{\text{infer}}$。训练时两个路由器同时更新，推理时仅用 $h_{\text{infer}}$，确保专家在训练阶段得到充分学习。

---

## Scaling Law

RankMixer 的参数量和计算量可以沿四个正交维度扩展：

- **Token 数量 $T$**：更多特征组 / 更细粒度的切分
- **模型宽度 $D$**：隐层维度
- **层数 $L$**：堆叠的 RankMixer Block 数量
- **专家数量 $E$**：MoE 中的专家数

对于全稠密（Dense）版本，单样本的参数量和前向 FLOPs 为：

$$
\#\text{Param} \approx 2kLTD^2, \quad \text{FLOPs} \approx 4kLTD^2
$$

其中 $k$ 为 Per-Token FFN 的隐藏层缩放比。

在 Sparse-MoE 版本中，有效参数量随专家数增长，但实际计算 FLOPs 仅与激活的专家数成正比，实现了参数量与计算量的解耦。

**离线 Scaling Law 实验**（万亿级抖音生产数据集）表明：
- RankMixer 的 AUC 随参数量增长持续提升，展现出良好的 Scaling 特性
- 相比传统 DLRM 架构（如 DCN + DeepFM 等组合），RankMixer 的 Scaling 斜率更陡
- 相同 FLOPs 预算下，RankMixer 始终优于传统架构

---

## 工业部署与业务效果

RankMixer 已在字节跳动多个核心业务场景全量上线，以下是最关键的部署数据：

### 效率提升

| 指标 | Baseline（旧架构） | RankMixer |
|------|-------------------|-----------|
| MFU | 4.5% | **45%** |
| 推理 SM Activity | ~30% | **~80%** |
| Dense 参数量 | ~16M | **~1B**（提升约 70 倍） |
| 推理延迟增幅 | — | 基本持平（经工程优化后） |

MFU 提升近 10 倍的核心原因：
- Token Mixing 将大量零碎的异构算子统一为规整的大矩阵乘法
- 消除 CPU 时代的碎片化算子（DCNv2、LHUC 等），单一架构覆盖全部特征交互
- 高 SM Activity 意味着 GPU 的流式多处理器被充分利用

### 业务效果（线上 A/B 测试）

| 场景 | 指标 | 提升 |
|------|------|------|
| 抖音主 Feed | 用户活跃天数（LT30） | **+0.3%** |
| 抖音主 Feed | 总使用时长 | **+1.08%** |
| 广告场景 | 多项核心指标 | 显著提升 |
| 电商广告 | 多项核心指标 | 显著提升 |

值得注意的是，低活跃用户的提升更为明显，说明更大的模型容量更好地捕获了长尾用户的偏好信号。

### 部署规模

目前 RankMixer-1B 已覆盖抖音首页 Feed 精排、电商广告精排等数十个内部业务场景，并已推广至字节跳动内部多个业务线。

---

## 与传统排序模型的对比

### 架构对比

| 模型 | 特征交互方式 | 并行性 | GPU MFU | 可扩展性 |
|------|-------------|--------|---------|---------|
| Wide & Deep | 手工交叉 + DNN | 低 | 低 | 差 |
| DCN / DCNv2 | 显式交叉网络 | 中 | 低（内存受限算子） | 中 |
| DHEN | 异构模块堆叠 | 中 | 中 | 中 |
| Wukong | 多子结构 bagging | 中 | 中 | 中 |
| AutoInt | Self-Attention | 高 | 中（注意力矩阵内存受限） | 中 |
| HiFormer | 改进 Attention | 高 | 中 | 中 |
| **RankMixer** | **Token Mixing + Per-Token FFN** | **高** | **高（~45%）** | **优（1B+）** |

### 关键差异分析

1. **vs Transformer**：RankMixer 保留了 Transformer 的高并行性，但用无参数 Token Mixing 替代 Self-Attention，消除了 $O(T^2)$ 的注意力矩阵计算和内存开销，更适合推荐场景中大量异构特征的交互。
2. **vs 传统 DLRM**：传统 DLRM 依赖多种异构模块的组合，每种模块的算子不同、GPU 利用率各异，整体 MFU 极低。RankMixer 用统一架构替代所有碎片化模块，MFU 提升 10 倍。
3. **vs MLP-Mixer**：RankMixer 的 Token Mixing 借鉴了 MLP-Mixer 的思想，但针对推荐场景做了深度适配——语义分组 Token 化、多头机制、Per-Token FFN 等设计都是推荐特有的。

---

## 技术洞见

### 1. 为什么"无参数"的 Token Mixing 反而更好？

在 NLP 中，Self-Attention 学习 token 之间的语义关系是有效的，因为自然语言的 token 都在同一语义空间中。但推荐系统的特征 Token 来自完全不同的语义域（用户属性 vs 物品标签 vs 行为序列），在这些异构空间之间学习注意力权重是困难的、不稳定的。

Token Mixing 通过简单的 reshape 操作将不同 Token 的信息放在一起，再用线性变换混合，本质上是在说"我不去建模异构空间之间的相似度，而是直接混合它们的信息"。这种简洁的设计反而更适合推荐场景。

### 2. Per-Token FFN 的本质：特征子空间的参数隔离

推荐数据的核心特性是特征子空间的高度异构性。传统共享 FFN 的问题在于：假设模型有 100 个特征 Token，它们共享同一个 FFN，那么高频特征（如热门视频）的梯度会主导参数更新，低频特征（如小众视频）的信号被淹没。

Per-Token FFN 通过参数隔离，让每个特征子空间拥有独立的表示变换能力。这类似于"因材施教"——不同语义空间用不同的参数来学习，避免了强特征对弱特征的压制。

### 3. 软硬件协同设计（Co-design）的范式意义

RankMixer 最深层的启示不在于某个具体模块，而在于它的**设计理念**：模型架构应该与底层硬件的计算特性对齐。

- GPU 擅长大规模矩阵乘法 → 将所有操作统一为矩阵运算
- GPU 不擅长碎片化的小算子 → 消除异构模块
- 推理延迟有硬约束 → 用 MoE 实现参数量与计算量的解耦

这种软硬件协同的思路，可能是未来推荐系统架构演进的核心方向。

---

## 后续工作：TokenMixer-Large

字节跳动团队在 RankMixer 的基础上进一步提出了 **TokenMixer-Large**（[arXiv:2602.06563](https://arxiv.org/abs/2602.06563)），解决了 RankMixer 在更深层配置中的几个关键瓶颈：

- **Mixing & Reverting 操作**：解决 Token Mixing 后原始语义信息丢失的问题
- **层间残差连接 + 辅助损失**：解决深层模型梯度消失问题
- **Sparse Per-Token MoE**：升级 ReLU-MoE 为全稀疏训练+稀疏推理范式
- 成功将模型扩展至 **7B（线上）/ 15B（离线）** 参数

TokenMixer-Large 已在字节跳动多个场景部署：电商订单量 +1.66%、广告 ADSS +2.0%、直播营收 +1.4%。

---

## 总结

RankMixer 是推荐系统领域在"大模型"方向的重要实践，其核心贡献包括：

1. **统一架构替代碎片化设计**：用 Token Mixing + Per-Token FFN 统一替代了传统排序模型中多种异构的手工特征交叉模块，MFU 从 4.5% 提升至 45%
2. **参数量与推理成本的解耦**：通过高 MFU + MoE + 工程优化，在不增加推理成本的前提下将参数量扩展了两个数量级
3. **软硬件协同设计理念**：证明了推荐模型应该与 GPU 的计算特性对齐设计，而非沿袭 CPU 时代的架构惯性
4. **大规模验证**：在抖音主 Feed 全量上线，用户活跃天数 +0.3%，使用时长 +1.08%，并已推广至数十个业务场景

RankMixer 验证了推荐系统中 Scaling Law 的可行性，其软硬件协同的设计范式对未来推荐系统架构的演进具有深远的指导意义。

---

**参考文献**

- [1] Zhu J, Fan Z, Zhu X, et al. RankMixer: Scaling Up Ranking Models in Industrial Recommenders. arXiv:2507.15551, 2025.
- [2] TokenMixer-Large: Scaling Up Large Ranking Models in Industrial Recommenders. arXiv:2602.06563, 2026.
- [3] Tolstikhin I, Houlsby N, et al. MLP-Mixer: An all-MLP Architecture for Vision. NeurIPS 2021.
- [4] Wang R, et al. DCN V2: Improved Deep & Cross Network. WWW 2021.
- [5] Li K, et al. Wukong: Towards a Scaling Law for Large-Scale Recommendation. arXiv:2311.11351, 2023.
- [6] Liu Z, et al. AutoInt: Automatic Feature Interaction Learning via Self-Attentive Neural Networks. CIKM 2019.
