# 📊 M-GPT 数据处理详细流程解析

## 🎯 训练目标与损失函数详解

### 1. 损失函数公式解析

```
loss = Σ_m loss_fct(g_i,t^(m), true_label)
```

**含义**：
- `g_i,t^(m)`: 第 `m` 阶（层）对位置 `i` 的物品 `t` 的预测得分
- `m ∈ [1, item_level]`: 遍历所有阶（默认 item_level=3）
- `loss_fct`: CrossEntropyLoss，多分类交叉熵损失

**代码实现**（第296-299行）：
```python
for i in range(self.item_level):  # m = 1, 2, 3
    logits = torch.matmul(multi_output[i], test_item_emb.transpose(0, 1))
    # logits: [B, mask_len, item_num] - 对所有物品的预测得分
    # pos_items: [B, mask_len] - 真实标签（被掩码的物品ID）
    
    loss = loss + torch.sum(
        loss_fct(logits.view(-1, item_num), pos_items.view(-1)) * targets
    ) / torch.sum(targets)
```

**损失计算步骤**：
1. **多阶预测**：对每个被掩码位置，计算 1阶、2阶、3阶的预测
2. **概率分布**：`logits` 是 `[B, mask_len, item_num]`，表示每个位置对所有物品的得分
3. **交叉熵**：真实标签 `pos_items` vs 预测分布 `logits`
4. **加权求和**：`targets` 用于忽略 padding（值为0的位置）

---

## 📈 完整数据流详细解析

### 阶段 0️⃣：输入数据格式

**原始序列**（来自 RecBole 的 SequentialDataset）：
```
item_seq:    [i1, i2, i3, i4, i5, 0, 0, ...]  # 长度 200，不足用 0 padding
type_seq:    [c,  c,  a,  c,  b,  0, 0, ...]  # c=click, a=cart, b=buy
last_buy:    i5  # 最后一个真实物品（购买物品）
```

**形状**：
- `item_seq`: `[B, 200]` - B 是 batch size
- `type_seq`: `[B, 200]` 
- `last_buy`: `[B]`

---

### 阶段 1️⃣：添加最后购买物品（第142-146行）

**目的**：确保最后位置是真实的购买物品（用于预测）

```python
# 计算有效序列长度
n_objs = (torch.count_nonzero(item_seq, dim=1) + 1).tolist()
# 例如：[5, 3, 7, ...] - 每个样本的有效长度

# 在序列末尾添加 last_buy
for batch_id in range(batch_size):
    n_obj = n_objs[batch_id]  # 例如：5
    item_seq[batch_id][n_obj - 1] = last_buy[batch_id]  # 位置4（第5个）
    type_seq[batch_id][n_obj - 1] = self.buy_type      # 设为购买行为
```

**结果**：
```
item_seq:    [i1, i2, i3, i4, i5, 0, 0, ...]  # i5 已在最后（或替换原i5）
type_seq:    [c,  c,  a,  c,  b,  0, 0, ...]  # 最后是 buy 行为
有效长度:    5
```

---

### 阶段 2️⃣：随机掩码策略（第157-173行）

**掩码规则**（重要！）：
1. **最后一个位置必掩码**（第162-167行）
2. **其他位置随机掩码**，概率 = `mask_ratio = 0.2`（第168-173行）
3. **掩码位置的行为类型设为 0**

**代码流程**：
```python
for instance_idx, instance in enumerate(sequence_instances):
    masked_sequence = instance.copy()  # 复制原始序列
    pos_item = []      # 存储被掩码的真实物品
    index_ids = []     # 存储被掩码的位置索引
    
    for index_id, item in enumerate(instance):
        # 规则1: 最后一个位置必定掩码
        if index_id == n_objs[instance_idx] - 1:
            pos_item.append(item)              # 保存真实物品
            masked_sequence[index_id] = self.mask_token  # 替换为 [MASK]
            type_instances[instance_idx][index_id] = 0    # 行为设为0
            index_ids.append(index_id)         # 记录位置
            break  # 最后一个处理完就退出
        
        # 规则2: 其他位置随机掩码
        prob = random.random()
        if prob < self.mask_ratio:  # mask_ratio = 0.2
            pos_item.append(item)
            masked_sequence[index_id] = self.mask_token
            type_instances[instance_idx][index_id] = 0
            index_ids.append(index_id)
```

**示例**（假设随机选择位置2被掩码）：
```
原始序列:
item_seq:    [i1, i2, i3, i4, i5]
type_seq:    [c,  c,  a,  c,  b]
有效长度:    5

掩码过程:
位置0 (i1):  随机数 0.85 > 0.2 → 不掩码
位置1 (i2):  随机数 0.15 < 0.2 → 掩码！ ✓
位置2 (i3):  随机数 0.92 > 0.2 → 不掩码
位置3 (i4):  随机数 0.78 > 0.2 → 不掩码
位置4 (i5):  最后位置 → 必定掩码！ ✓

掩码后:
masked_seq:  [i1, [M], i3, i4, [M]]  # [M] = mask_token (通常是 n_items+1)
type_seq:    [c,  0,   a,  c,  0]    # 掩码位置行为设为0
pos_items:   [i2, i5]                # 真实标签
masked_index:[1,  4]                 # 被掩码的位置
```

**输出形状**：
- `masked_item_sequence`: `[B, max_len+1]` - 掩码后的序列
- `pos_items`: `[B, mask_item_length]` - 真实物品（填充到固定长度）
- `masked_index`: `[B, mask_item_length]` - 掩码位置索引
- `type_instances`: `[B, max_len+1]` - 掩码后的行为序列

---

### 阶段 3️⃣：图卷积（步骤 1-3）

**输入**：
```
masked_item_seq:  [i1, [M], i3, i4, [M]]  # [B, N+1]
type_seq:         [c,  0,   a,  c,  0]    # [B, N+1]
```

**处理**：
```python
# 1. 物品嵌入
item_emb = self.item_embedding(masked_item_seq)  # [B, N+1, H]

# 2. 行为嵌入
type_emb = self.type_embedding(type_seq)        # [B, N+1, H]

# 3. 构建邻接矩阵（交互级依赖）
# E[i,j] = item_emb[i] · item_emb[j]
# B[i,j] = type_emb[i] · type_emb[j]
# A = E ⊙ B

# 4. 图卷积（多阶）
H = self.MLGCN_layer(item_emb, type_emb, adj_matrix)
# 输出: H^(1), H^(2), H^(3) - 1阶、2阶、3阶表示
```

**输出**：
```
H^(1):  [B, N+1, H]  # 1阶图卷积（直接邻居）
H^(2):  [B, N+1, H]  # 2阶图卷积（2跳邻居）
H^(3):  [B, N+1, H]  # 3阶图卷积（3跳邻居）
```

---

### 阶段 4️⃣：多面 Transformer（步骤 4-9）

**输入**：`H^(l)` - 图卷积的输出

**步骤5-6：全局模式**：
```python
# 添加位置编码
H_with_pos = H + position_embedding

# 线性自注意力（全局）
H_Lin = LinSA(H_with_pos)  # [B, N+1, H]
```

**步骤7：多粒度模式**：
```python
# 多粒度多头自注意力
S_t1 = MGMHSA(H_with_pos, scale=4)   # 短期（最近4个）
S_t2 = MGMHSA(H_with_pos, scale=20)  # 中期（最近20个）
```

**步骤8-9：融合与FFN**：
```python
# 融合
H_fused = Concat([H_Lin, S_t1, S_t2]) @ W_d  # [B, N+1, H]

# 前馈网络
H_out = LayerNorm(FFN(H_fused) + H_fused)  # [B, N+1, H]
```

**输出**（每个阶）：
```
seq_output[0]:  [B, N+1, H]  # 1阶的最终表示
seq_output[1]:  [B, N+1, H]  # 2阶的最终表示
seq_output[2]:  [B, N+1, H]  # 3阶的最终表示
```

---

### 阶段 5️⃣：MaxPooling 预测（步骤 10-17）

#### 步骤1：构建 Multi-hot 映射（第280行）

**目的**：从完整序列表示中提取被掩码位置的表示

```python
pred_index_map = self.multi_hot_embed(masked_index, masked_item_seq.size(-1))
# masked_index: [B, mask_len] = [[1, 4], [2, 5], ...]
# 输出: [B, mask_len, max_len+1]
```

**Multi-hot 示例**：
```python
masked_index = [[1, 4], [0, 2]]  # 2个样本，每个有2个掩码位置
max_len = 5

pred_index_map = [
    # 样本1
    [[0, 1, 0, 0, 0],   # 位置1的 one-hot
     [0, 0, 0, 0, 1]],  # 位置4的 one-hot
    # 样本2
    [[1, 0, 0, 0, 0],   # 位置0的 one-hot
     [0, 0, 1, 0, 0]]   # 位置2的 one-hot
]
```

#### 步骤2：提取被掩码位置的表示（第288行）

```python
for j in range(self.item_level):  # j = 0, 1, 2 (对应1,2,3阶)
    # 矩阵乘法：从完整序列中提取掩码位置的表示
    output_j = torch.bmm(pred_index_map, seq_output[j])
    # pred_index_map: [B, mask_len, max_len+1]
    # seq_output[j]:  [B, max_len+1, H]
    # 结果: [B, mask_len, H]
    multi_output.append(output_j)
```

**结果**：
```
multi_output[0]:  [B, mask_len, H]  # 1阶的掩码位置表示
multi_output[1]:  [B, mask_len, H]  # 2阶的掩码位置表示
multi_output[2]:  [B, mask_len, H]  # 3阶的掩码位置表示
```

#### 步骤3：计算预测得分（第297行）

```python
for i in range(self.item_level):
    logits = torch.matmul(multi_output[i], test_item_emb.transpose(0, 1))
    # multi_output[i]: [B, mask_len, H]
    # test_item_emb:   [item_num, H]
    # 结果: [B, mask_len, item_num]
```

**含义**：
- 对每个被掩码位置，计算它与**所有物品**的相似度
- `logits[b, m, v]` = 样本 `b` 的掩码位置 `m` 对物品 `v` 的预测得分

**示例**：
```
logits[0, 0, :] = [0.1, 0.9, 0.3, 0.2, ...]  # 样本1的掩码位置1对所有物品的得分
logits[0, 1, :] = [0.2, 0.1, 0.8, 0.1, ...]  # 样本1的掩码位置4对所有物品的得分

真实标签:
pos_items[0] = [i2, i5]  # 位置1的真实物品是i2，位置4的真实物品是i5
```

#### 步骤4：计算损失（第298-299行）

```python
loss_fct = nn.CrossEntropyLoss(reduction='none')
targets = (masked_index > 0).float().view(-1)  # 忽略padding位置

for i in range(self.item_level):
    # 每个阶的损失
    loss_i = loss_fct(
        logits.view(-1, item_num),  # [B*mask_len, item_num]
        pos_items.view(-1)            # [B*mask_len] - 真实标签
    ) * targets
    
    loss += torch.sum(loss_i) / torch.sum(targets)
```

**损失计算示例**：
```
假设有1个样本，2个掩码位置：
logits[0]:  [B=1, mask_len=2, item_num=1000]
pos_items: [i2=2, i5=5]

对位置1（真实物品i2）：
- logits[0,0,:] = [0.1, 0.9, 0.3, ...]  # 对i2的得分最高
- CrossEntropy(i2) = -log(softmax(0.9)) = 0.11  # 损失较小

对位置4（真实物品i5）：
- logits[0,1,:] = [0.2, 0.1, 0.8, ...]  # 对i5的得分
- CrossEntropy(i5) = -log(softmax(0.8)) = 0.22  # 损失

总损失 = (0.11 + 0.22) / 2 = 0.165
```

**多阶损失**：
```python
loss = loss_1阶 + loss_2阶 + loss_3阶  # 三个阶的损失相加
```

---

## 🎯 关键理解点

### 1. **为什么要掩码？**
- **自监督学习**：通过预测被掩码的物品，模型学习序列模式
- **类似 BERT**：通过掩码语言模型学习语言理解

### 2. **为什么最后位置必掩码？**
- **关键任务**：预测下一个购买物品（Next-Item Prediction）
- **实际应用**：在推荐系统中，预测用户下一步会买什么

### 3. **为什么多阶？**
- **1阶**：直接相邻物品的依赖（i3 依赖于 i2）
- **2阶**：2跳依赖（i4 依赖于 i2，通过 i3）
- **3阶**：更深层的依赖关系
- **MaxPooling**：选择最佳阶的预测

### 4. **mask_item_length 的作用？**
```python
self.mask_item_length = int(self.mask_ratio * self.max_seq_length)
# mask_ratio = 0.2, max_seq_length = 200
# mask_item_length = 40
```
- **固定长度**：确保 batch 中所有样本的 `pos_items` 长度一致
- **Padding**：如果掩码位置少于40个，用0填充

---

## 📝 完整示例

**输入**：
```
item_seq:    [i1, i2, i3, i4, i5, 0, 0, ...]
type_seq:    [c,  c,  a,  c,  b,  0, 0, ...]
last_buy:    i5
```

**掩码后**：
```
masked_seq:  [i1, [M], i3, i4, [M], 0, 0, ...]
type_seq:    [c,  0,   a,  c,  0,   0, 0, ...]
pos_items:   [i2, i5, 0,  0, ...]  # 填充到40
masked_index:[1,  4,  0,  0, ...]  # 填充到40
```

**模型输出**：
```
logits:      [B, 2, item_num]  # 2个掩码位置 × 所有物品
# 对位置1: 预测 i2（真实标签）
# 对位置4: 预测 i5（真实标签）
```

**损失**：
```
loss = CrossEntropy(pred_i2, true_i2) + CrossEntropy(pred_i5, true_i5)
```

---

## ✅ 总结

1. **掩码策略**：20%随机 + 最后位置必掩码
2. **多阶预测**：1阶、2阶、3阶图卷积 → 多粒度 Transformer → 各自预测
3. **损失计算**：每个阶分别计算 CrossEntropy，然后求和
4. **训练目标**：最小化所有阶在所有掩码位置的预测误差

这个流程使得 M-GPT 能够：
- ✅ 学习序列模式（通过掩码预测）
- ✅ 捕获多粒度偏好（多尺度 Transformer）
- ✅ 利用多阶依赖（图卷积）
- ✅ 自适应选择最佳预测（MaxPooling）

