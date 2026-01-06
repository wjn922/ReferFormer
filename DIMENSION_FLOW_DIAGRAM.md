# ReferFormer 维度流转详细示意图

## 完整的前向传播维度流程

```
假设输入:
- Batch size (B) = 1
- Num frames (T) = 5
- Input resolution: H=360, W=640
- Num queries per frame (Q) = 5
- Hidden dim (C) = 256
- Text length (L) = 15 (动态，padding后)
```

---

## Part 1: 视觉特征提取与处理

### 1.1 Backbone 特征提取
```
Input Video: [B, T, 3, H, W] = [1, 5, 3, 360, 640]
    ↓ (reshape for backbone)
[B*T, 3, H, W] = [5, 3, 360, 640]
    ↓ Backbone (ResNet50/Video-Swin)
Multi-scale features (res2-res5):
  res2: [5, 256,  90, 160]   (stride 4)
  res3: [5, 512,  45, 80]    (stride 8)   ← 使用
  res4: [5, 1024, 23, 40]    (stride 16)  ← 使用
  res5: [5, 2048, 12, 20]    (stride 32)  ← 使用
```

### 1.2 Input Projection (统一到 hidden_dim)
```
取最后3个尺度 (res3, res4, res5):

Level 0 (res3):
  [5, 512, 45, 80]
    ↓ Conv2d(512 → 256)
  [5, 256, 45, 80]  ← src_proj_0

Level 1 (res4):
  [5, 1024, 23, 40]
    ↓ Conv2d(1024 → 256)
  [5, 256, 23, 40]  ← src_proj_1

Level 2 (res5):
  [5, 2048, 12, 20]
    ↓ Conv2d(2048 → 256)
  [5, 256, 12, 20]  ← src_proj_2

(可选) Level 3 (额外下采样):
  [5, 256, 12, 20]
    ↓ Conv2d(stride=2)
  [5, 256, 6, 10]   ← src_proj_3
```

---

## Part 2: 文本特征提取

### 2.1 Text Encoding
```
Captions: ["a person wearing red shirt"] (list of B=1 strings)
    ↓ RoBERTa Tokenizer
Tokenized: [B, L] = [1, 15]  (padding到最长)
    ↓ RoBERTa Encoder
Word features:     [B, L, 768] = [1, 15, 768]
Sentence features: [B, 768] = [1, 768]
    ↓ FeatureResizer (768 → 256)
text_word_features:     [B, L, 256] = [1, 15, 256]
text_sentence_features: [B, 256] = [1, 256]
    ↓ Rearrange for attention
text_word_features: [L, B, 256] = [15, 1, 256]
text_word_masks:    [B, L] = [1, 15]  (True=padding)
```

---

## Part 3: Vision-Language Early Fusion

### 3.1 融合机制详解

**对于每个视觉尺度** (以 Level 0 为例):

```
输入视觉特征: src_proj_0 = [B*T, C, H, W] = [5, 256, 45, 80]
输入文本特征: text_word_features = [L, B, C] = [15, 1, 256]

Step 1: Rearrange 视觉特征为序列格式
  [5, 256, 45, 80]
    ↓ rearrange('(b t) c h w -> (t h w) b c', b=1, t=5)
  [T*H*W, B, C] = [18000, 1, 256]
  其中: T*H*W = 5 × 45 × 80 = 18,000

Step 2: Multi-Head Cross-Attention (8 heads)
  Query:  视觉特征 [18000, 1, 256]  (每个时空位置)
  Key:    文本特征 [15, 1, 256]     (每个 token)
  Value:  文本特征 [15, 1, 256]

  计算过程:
    1. Linear projections: Q, K, V
       Q: [18000, 1, 256] → [18000, 1, 8, 32] → [8, 18000, 1, 32]
       K: [15, 1, 256]    → [15, 1, 8, 32]    → [8, 15, 1, 32]
       V: [15, 1, 256]    → [15, 1, 8, 32]    → [8, 15, 1, 32]

    2. Attention scores:
       attn = softmax(Q @ K^T / sqrt(32))
       [8, 18000, 1, 32] @ [8, 1, 32, 15]^T
       = [8, 18000, 1, 15]  (attention weights)

    3. Weighted sum:
       out = attn @ V
       [8, 18000, 1, 15] @ [8, 15, 1, 32]
       = [8, 18000, 1, 32]
       ↓ reshape
       [18000, 1, 256]  ← attention output

Step 3: Element-wise Multiplication Fusion
  vision_fused = vision_original ⊗ attention_output
  [18000, 1, 256] ⊗ [18000, 1, 256]
  = [18000, 1, 256]

Step 4: Rearrange 回空间格式
  [18000, 1, 256]
    ↓ rearrange('(t h w) b c -> (b t) c h w', t=5, h=45, w=80)
  [5, 256, 45, 80]  ← 融合后的视觉特征
```

**重要观察**:
- **输出维度 = 输入视觉特征维度**: `[B*T, C, H, W]`
- **不是** `[N, L]` 的形式！
- 每个空间位置 attend 到所有 L 个文本 token，得到 256-dim 的加权表示
- 通过 element-wise multiplication 而非 concat 融合

### 3.2 所有尺度融合后

```
Level 0: [5, 256, 45, 80]  (8x)
Level 1: [5, 256, 23, 40]  (16x)
Level 2: [5, 256, 12, 20]  (32x)
(Level 3: [5, 256, 6, 10])  (64x, 可选)
```

---

## Part 4: Transformer Encoder 输入准备

### 4.1 展平与拼接

```
对每个尺度:

Level 0:
  [5, 256, 45, 80]
    ↓ flatten(2).transpose(1,2)
  [5, 3600, 256]  where 3600 = 45×80

Level 1:
  [5, 256, 23, 40]
    ↓ flatten(2).transpose(1,2)
  [5, 920, 256]   where 920 = 23×40

Level 2:
  [5, 256, 12, 20]
    ↓ flatten(2).transpose(1,2)
  [5, 240, 256]   where 240 = 12×20

拼接在序列维度:
  src_flatten = cat([Level0, Level1, Level2], dim=1)
  [5, 4760, 256]  where 4760 = 3600+920+240

同时记录:
  spatial_shapes = [(45,80), (23,40), (12,20)]
  level_start_index = [0, 3600, 4520]
```

### 4.2 添加 Level Embeddings

```
level_embed: [num_levels, 256] = [3, 256]  (learnable)

For each level:
  lvl_pos_embed = pos_embed + level_embed[lvl]

这样 encoder 可以区分不同尺度的特征
```

---

## Part 5: Deformable Transformer Encoder

### 5.1 Multi-Scale Deformable Attention

```
Input:
  src: [B*T, Σ(H*W), C] = [5, 4760, 256]
  spatial_shapes: [(45,80), (23,40), (12,20)]
  level_start_index: [0, 3600, 4520]

Deformable Attention 机制:
  对每个 query 位置 (共 4760 个):
    - 在每个尺度上采样 K=4 个点
    - 共 L=3 个尺度, N_heads=8
    - 总采样点: 3 × 8 × 4 = 96 个点

  采样偏移学习:
    offsets = Linear(query_feat)  # [4760, 256] → [4760, 96×2]
    # 96×2: 每个采样点的 (dx, dy)

  注意力权重学习:
    attn_weights = softmax(Linear(query_feat))  # [4760, 96]

  输出:
    output[i] = Σ(attn_weights[i,j] × sampled_features[i,j])
    [5, 4760, 256]

Encoder output:
  memory: [B*T, Σ(H*W), C] = [5, 4760, 256]
```

### 5.2 转换回 FPN 格式

```
从 memory 中按 spatial_shapes 切分:

memory[0:3600]:
  [5, 3600, 256]
    ↓ reshape
  [5, 45, 80, 256]
    ↓ permute
  [5, 256, 45, 80]  ← memory_level_0 (8x)

memory[3600:4520]:
  [5, 920, 256] → [5, 256, 23, 40]  ← memory_level_1 (16x)

memory[4520:4760]:
  [5, 240, 256] → [5, 256, 12, 20]  ← memory_level_2 (32x)
```

---

## Part 6: Transformer Decoder

### 6.1 Query 初始化

```
Content Query (tgt):
  text_sentence_features: [B, 256] = [1, 256]
    ↓ repeat for each frame & each query
  [B, T, Q, 256] = [1, 5, 5, 256]
    ↓ rearrange('b t q c -> (b t) q c')
  [B*T, Q, 256] = [5, 5, 256]

Position Query (query_embed):
  learnable: [Q, 256] = [5, 256]
    ↓ expand for each frame
  [B*T, Q, 256] = [5, 5, 256]
```

### 6.2 Decoder 层处理

**每层 Decoder 包含**:
1. Self-Attention (queries 之间)
2. Cross-Attention (queries attend to multi-scale memory)
3. FFN

```
Layer l (l=0,1,2,3):

  Input: tgt = [B*T, Q, 256] = [5, 5, 256]

  Step 1: Self-Attention
    Query = Key = tgt + query_embed
    Value = tgt
    ↓ MultiheadAttention (within each frame's Q queries)
    [5, 5, 256]  (same shape)

  Step 2: Cross-Attention (Deformable)
    Query: tgt [5, 5, 256]
    Key/Value: memory [5, 4760, 256]

    对每个 query:
      - 学习 reference_point: [5, 5, 2]  (归一化坐标)
      - 在每个尺度上采样 K=4 个点
      - 计算 attention 并聚合

    Output: [5, 5, 256]

  Step 3: FFN
    [5, 5, 256] → [5, 5, 2048] → [5, 5, 256]

  Output: hs[l] = [5, 5, 256]
```

### 6.3 Decoder 输出

```
所有层的输出堆叠:
  hs: [L, B*T, Q, 256] = [4, 5, 5, 256]

每层的预测:
  - Class logits: [L, B*T, Q, num_classes]
  - Box coords:   [L, B*T, Q, 4]
  - Reference points: [L, B*T, Q, 2]
```

---

## Part 7: Cross-Modal FPN Decoder (生成 Mask Features)

### 7.1 Top-down 融合

```
输入:
  - Backbone features (res2): [5, 256, 90, 160]  (4x)
  - Encoder memory (8x-32x): [5, 256, 45, 80], [5, 256, 23, 40], [5, 256, 12, 20]
  - Text features: [L, B, 256] = [15, 1, 256]

从粗到细处理:

Level 32x (memory_level_2):
  [5, 256, 12, 20]
    ↓ Lateral Conv
  [5, 256, 12, 20]
    ↓ VisionLanguageBlock (cross-attn with text)
  [5, 256, 12, 20]
    ↓ Output Conv
  y_32x = [5, 256, 12, 20]

Level 16x (memory_level_1):
  [5, 256, 23, 40]
    ↓ Lateral Conv
  [5, 256, 23, 40]
    ↓ VisionLanguageBlock
  [5, 256, 23, 40]
    ↓ + upsample(y_32x)
  [5, 256, 23, 40] + [5, 256, 23, 40]
    ↓ Output Conv
  y_16x = [5, 256, 23, 40]

Level 8x (memory_level_0):
  Similar process...
  y_8x = [5, 256, 45, 80]

Level 4x (backbone res2):
  [5, 256, 90, 160]
    ↓ Similar process with y_8x
  y_4x = [5, 256, 90, 160]

最终 mask features:
  [5, 256, 90, 160]
    ↓ Conv 256→mask_dim
  [5, mask_dim, 90, 160] = [5, 256, 90, 160]
```

### 7.2 VisionLanguageBlock 细节

```
视觉特征: [5, 256, H, W]
  ↓ rearrange('(b t) c h w -> (t h w) b c', b=1, t=5)
[T*H*W, B, 256]

Self-Attention (空间下采样 sr_ratio):
  如果 H, W 太大 (如32x层), 先下采样减少计算
  [T*H*W, B, 256] → [T*H'*W', B, 256] → Self-Attn → 上采样回 [T*H*W, B, 256]

Cross-Attention with Text:
  Query:  视觉 [T*H*W, B, 256]
  Key/Value: 文本 [L, B, 256]
  Output: [T*H*W, B, 256]

  ↓ rearrange back
[B*T, 256, H, W]
```

---

## Part 8: Dynamic Mask Generation

### 8.1 生成动态卷积参数

```
Decoder output (最后一层):
  hs[-1]: [B*T, Q, 256] = [5, 5, 256]

Controller MLP:
  [5, 5, 256]
    ↓ MLP (3 layers)
  [5, 5, num_params] = [5, 5, 132353]

解析参数 (per instance):
  Layer 0:
    weights_0: [258 * 256] = 66,048
    bias_0:    [256]
  Layer 1:
    weights_1: [256 * 256] = 65,536
    bias_1:    [256]
  Layer 2:
    weights_2: [256 * 1] = 256
    bias_2:    [1]

  Total: 132,353 parameters per query
```

### 8.2 准备 Mask Features 与相对坐标

```
Mask features: [B, T, mask_dim, H, W] = [1, 5, 256, 90, 160]
Reference points: [B, T*Q, 2] = [1, 25, 2]

计算相对坐标:
  pixel_locations: [90*160, 2] = [14400, 2]  (每个像素的绝对坐标)
  reference_points: [1, 5, 5, 2]  (每个query的中心点)

  For each query:
    relative_coords = pixel_locations - reference_point
    [90, 160, 2]  (每个像素相对于该query的偏移)

拼接:
  mask_features: [1, 5, 5, 256, 90, 160]
  relative_coords: [1, 5, 5, 2, 90, 160]
    ↓ concat
  [1, 5, 5, 258, 90, 160]
    ↓ reshape
  [1, 25*258, 90, 160]
```

### 8.3 执行 Dynamic Convolution

```
Input: [1, 25*258, 90, 160]
  其中: 25 = B*T*Q (每个 instance)
        258 = mask_dim + 2 (特征 + 相对坐标)

Reshape weights for grouped conv:
  weights_0: [25, 258, 256] → reshape → [25*256, 258, 1, 1]
  bias_0:    [25, 256] → reshape → [25*256]

Layer 0:
  F.conv2d(input, weights_0, bias_0, groups=25)
  [1, 25*258, 90, 160] → [1, 25*256, 90, 160]
  ↓ ReLU

Layer 1:
  weights_1: [25*256, 256, 1, 1]
  F.conv2d(..., groups=25)
  [1, 25*256, 90, 160] → [1, 25*256, 90, 160]
  ↓ ReLU

Layer 2:
  weights_2: [25*1, 256, 1, 1]
  F.conv2d(..., groups=25)
  [1, 25*256, 90, 160] → [1, 25*1, 90, 160]

Reshape output:
  [1, 25, 90, 160]
    ↓ reshape
  [B, T*Q, H, W] = [1, 25, 90, 160]

上采样 4x (stride 4 → stride 1):
  [1, 25, 90, 160]
    ↓ bilinear upsample
  [1, 25, 360, 640]
    ↓ reshape
  [B, T, Q, H, W] = [1, 5, 5, 360, 640]
```

---

## Part 9: 最终输出

```
输出字典:
  pred_logits: [B, T, Q, num_classes] = [1, 5, 5, 1]  (binary)
  pred_boxes:  [B, T, Q, 4] = [1, 5, 5, 4]  (cxcywh, normalized)
  pred_masks:  [B, T, Q, H, W] = [1, 5, 5, 360, 640]

如果 aux_loss=True:
  aux_outputs: [
    {pred_logits: [1,5,5,1], pred_boxes: [1,5,5,4], pred_masks: [1,5,5,360,640]},  # layer 0
    {...},  # layer 1
    {...},  # layer 2
  ]
```

---

## 维度流转总结图

```
Vision Branch:
  [B,T,3,H,W] → Backbone → Multi-scale features
  → Input Proj → [B*T, C, H_l, W_l] for each level
  → Vision-Language Fusion → [B*T, C, H_l, W_l]
  → Flatten & Concat → [B*T, Σ(H*W), C]
  → Encoder → [B*T, Σ(H*W), C]

Text Branch:
  Captions → RoBERTa → [B, L, 768]
  → Resizer → [B, L, C]
  → Rearrange → [L, B, C]
  ↓
  (融合到 Vision Branch)

Query Branch:
  text_sentence_features [B, C]
  → Repeat → [B*T, Q, C]
  + query_embed (learnable) [Q, C]
  ↓
Decoder:
  [B*T, Q, C] × 4 layers
  → hs: [L, B*T, Q, C]

Mask Branch:
  Encoder memory → FPN Decoder → mask_features [B*T, mask_dim, H/4, W/4]
  + hs → Controller → dynamic_params
  + relative_coords
  → Dynamic Conv
  → Masks [B, T, Q, H, W]

Classification & Box:
  hs → class_embed → [B, T, Q, num_classes]
  hs → bbox_embed → [B, T, Q, 4]
```

---

## 关键维度记忆口诀

1. **B, T, Q, C**: Batch, Time, Queries, Channels
2. **视觉特征**: 空间展平后按尺度拼接
3. **文本特征**: 序列格式 [L, B, C]
4. **融合**: Cross-attention后保持视觉维度
5. **Decoder**: 批处理所有帧 [B*T, Q, C]
6. **Mask**: 动态卷积针对每个instance独立生成

---

## 常见维度变换操作

```python
# 1. 展平空间维度
[B, C, H, W] → flatten(2).transpose(1, 2) → [B, H*W, C]

# 2. Batch-Time 合并/分离
[B, T, ...] → rearrange('b t ... -> (b t) ...') → [B*T, ...]
[B*T, ...] → rearrange('(b t) ... -> b t ...', b=B) → [B, T, ...]

# 3. 空间到序列
[B, C, H, W] → rearrange('b c h w -> (h w) b c') → [H*W, B, C]
[H*W, B, C] → rearrange('(h w) b c -> b c h w', h=H) → [B, C, H, W]

# 4. Query 扩展
[Q, C] → unsqueeze(0).expand(B, -1, -1) → [B, Q, C]
[B, C] → repeat('b c -> b t q c', t=T, q=Q) → [B, T, Q, C]
```
