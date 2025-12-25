# Megatron-LM 数据流详细指南

> 深入解析 Megatron-LM 的预训练数据流，包括不同数据格式的处理、数据加载器、Shuffle 机制、分布式数据加载和断点续训等。

---

## 目录

1. [数据格式对比](#1-数据格式对比)
2. [数据预处理流程](#2-数据预处理流程)
3. [数据加载架构](#3-数据加载架构)
4. [数据加载器类型](#4-数据加载器类型)
5. [Shuffle 机制](#5-shuffle-机制)
6. [分布式数据加载](#6-分布式数据加载)
7. [断点续训机制](#7-断点续训机制)
8. [性能优化建议](#8-性能优化建议)

---

## 1. 数据格式对比

### 1.1 支持的数据格式

Megatron-LM 支持两种主要的数据格式：

#### **格式 1：二进制索引数据集（IndexedDataset）**

**文件格式**：
- `.bin` 文件：存储 tokenized 后的二进制数据
- `.idx` 文件：存储索引元数据（序列长度、偏移量、文档索引等）

**预处理要求**：
- 必须使用 `preprocess_data.py` 预处理
- 从 JSONL → 二进制格式

**优点**：
- ✅ **高效随机访问**：通过索引文件快速定位数据
- ✅ **内存映射支持**：可以使用 mmap 减少内存占用
- ✅ **高吞吐量**：二进制格式 I/O 效率高
- ✅ **支持大规模数据**：可以处理 TB 级数据
- ✅ **缓存友好**：索引可以缓存，加速数据加载

**缺点**：
- ❌ 需要预处理步骤
- ❌ 不支持在线修改数据
- ❌ 需要额外的存储空间（索引文件）

#### **格式 2：在线 Tokenization（Iterable Dataset）**

**数据格式**：
- JSONL 文件（通过 HuggingFace datasets 库）
- Parquet 文件（通过 HuggingFace datasets 库）
- 其他可迭代数据源

**预处理要求**：
- 无需预处理
- 训练时在线 tokenization

**优点**：
- ✅ **灵活性高**：可以动态修改数据
- ✅ **无需预处理**：直接使用原始数据
- ✅ **支持多种格式**：JSONL、Parquet、CSV 等

**缺点**：
- ❌ **吞吐量较低**：在线 tokenization 有计算开销
- ❌ **随机访问困难**：需要顺序读取
- ❌ **内存占用较高**：需要加载整个数据集或使用流式处理

### 1.2 数据格式选择建议

| 场景 | 推荐格式 | 原因 |
|------|---------|------|
| **大规模预训练** | 二进制索引数据集 | 高吞吐量、高效随机访问 |
| **小规模实验** | 在线 Tokenization | 灵活性高、无需预处理 |
| **SFT/指令微调** | JSONL（在线） | 数据格式复杂、需要动态处理 |
| **多模态数据** | 二进制索引数据集 | 支持图像等二进制数据 |
| **快速迭代** | 在线 Tokenization | 无需重新预处理 |

### 1.3 数据格式处理流程对比

#### JSONL → 二进制索引数据集

```
原始 JSONL 文件
    ↓
preprocess_data.py
    ├─> 读取 JSONL（支持 gzip）
    ├─> [可选] 句子分割（NLTK）
    ├─> Tokenization（多进程并行）
    ├─> 构建 IndexedDataset
    │   ├─> 写入 .bin 文件（token IDs）
    │   └─> 写入 .idx 文件（元数据）
    └─> 输出：dataset_text_document.bin/idx
```

**性能特点**：
- **预处理时间**：一次性成本，可并行处理
- **训练时 I/O**：高效（内存映射、随机访问）
- **吞吐量**：**高**（通常 > 1000 samples/s）

#### JSONL/Parquet → 在线 Tokenization

```
原始 JSONL/Parquet 文件
    ↓
训练时加载
    ├─> HuggingFace datasets.load_dataset()
    ├─> 在线 Tokenization（每个样本）
    └─> 返回 tokenized 数据
```

**性能特点**：
- **预处理时间**：无（训练时处理）
- **训练时计算**：每个 batch 都需要 tokenization
- **吞吐量**：**中等**（通常 200-500 samples/s，取决于 tokenizer）

### 1.4 吞吐量对比（理论值）

| 数据格式 | 预处理时间 | 训练吞吐量 | I/O 开销 | Tokenization 开销 |
|---------|----------|-----------|---------|------------------|
| **二进制索引** | 高（一次性） | **很高** (1000+ samples/s) | 低（mmap） | 无（已预处理） |
| **JSONL 在线** | 无 | **中等** (200-500 samples/s) | 中（文件读取） | 高（每个样本） |
| **Parquet 在线** | 无 | **中等** (300-600 samples/s) | 低（列式存储） | 高（每个样本） |

**注意**：实际吞吐量取决于：
- 硬件配置（CPU、存储 I/O）
- 数据大小（序列长度、文件大小）
- Tokenizer 复杂度
- 数据加载器配置（num_workers、pin_memory 等）

---

## 2. 数据预处理流程

### 2.1 JSONL 预处理流程

#### 阶段 1：文件分片（可选）

```python
# 如果 partitions > 1
if args.partitions > 1:
    # 将大文件分片成多个小文件
    for idx in range(args.partitions):
        partitioned_file = open(f"{input_file}_{idx}.jsonl", 'w')
        # 轮询或顺序分配行到不同分片
        for line in input_file:
            partitioned_file.write(line)
```

**分片策略**：
- **轮询分配**（默认）：`index = (index + 1) % partitions`
- **顺序分配**（`--keep-sequential-samples`）：保持样本顺序

#### 阶段 2：句子分割（可选）

```python
if args.split_sentences:
    # 使用 NLTK 分割句子
    encoder = Encoder(args)
    pool = multiprocessing.Pool(workers, initializer=encoder.initializer)
    
    for json_line in input_file:
        # 分割句子
        sentences = encoder.splitter.tokenize(text)
        # 写入分割后的文件
        output_file.write(json.dumps({"text": sentences}))
```

**句子分割的作用**：
- 将长文档分割成句子
- 便于构建固定长度的训练样本
- 支持跨文档的样本构建

#### 阶段 3：Tokenization

```python
# 多进程并行 tokenization
pool = multiprocessing.Pool(workers, initializer=encoder.initializer)
encoded_docs = pool.imap(encoder.encode, input_file, 32)

for doc_ids, sentence_lens, _ in encoded_docs:
    # 添加到数据集构建器
    builder.add_document(doc_ids, sentence_lens)
```

**Tokenization 流程**：
```python
def encode(self, json_line):
    data = json.loads(json_line)
    doc_ids = []
    for sentence in sentences:
        sentence_ids = tokenizer.tokenize(sentence)
        doc_ids.extend(sentence_ids)
    
    # 可选：添加 EOD token
    if args.append_eod:
        doc_ids.append(tokenizer.eod)
    
    return doc_ids, sentence_lens
```

#### 阶段 4：构建二进制数据集

```python
# 为每个 json_key 创建构建器
for key in args.json_keys:
    builder = IndexedDatasetBuilder(
        f"{output_prefix}_{key}_document.bin",
        dtype=DType.optimal_dtype(tokenizer.vocab_size),
    )
    
    # 添加文档
    for doc_ids, sentence_lens in encoded_docs:
        builder.add_document(doc_ids, sentence_lens)
    
    # 完成构建
    builder.finalize(f"{output_prefix}_{key}_document.idx")
```

### 2.2 二进制数据集结构

#### `.bin` 文件结构

```
[Document 0]
  [Sequence 0]: token_0, token_1, ..., token_n
  [Sequence 1]: token_0, token_1, ..., token_m
  [EOD]
[Document 1]
  [Sequence 0]: token_0, token_1, ..., token_k
  [EOD]
...
```

**存储格式**：
- 使用 NumPy 数组存储 token IDs
- 数据类型根据词汇表大小自动选择（uint16 或 uint32）
- 连续存储，通过索引文件定位

#### `.idx` 文件结构

```
[Header]
  - index_version: int32
  - dtype_code: int32
  - num_sequences: int64
  - num_documents: int64

[Sequence Metadata]
  - sequence_lengths: int32[num_sequences]      # 每个序列的长度
  - sequence_pointers: int64[num_sequences]    # 每个序列在 .bin 中的字节偏移
  - document_indices: int32[num_documents]      # 每个文档的序列范围 [start, end)
  - sequence_modes: int32[num_sequences]       # 序列模式（多模态用）
```

**索引访问示例**：
```python
# 获取序列 100
sequence_length = index.sequence_lengths[100]
sequence_offset = index.sequence_pointers[100]
data = bin_file[sequence_offset:sequence_offset + sequence_length * dtype_size]
```

### 2.3 在线 Tokenization 流程（SFT 示例）

```python
class SFTLowLevelDataset:
    def __init__(self, dataset_path: str):
        from datasets import load_dataset
        # 支持 JSONL 和 Parquet
        self.dataset = load_dataset(
            "json",  # 或 "parquet"
            data_files=dataset_path,
            split="all"
        )
    
    def __getitem__(self, idx: int):
        # 返回原始消息列表
        return self.dataset[idx]["messages"]
```

**训练时 Tokenization**：
```python
class SFTDataset(MegatronDataset):
    def __getitem__(self, idx: int):
        # 获取原始对话
        conversation = self.dataset[idx]
        
        # 在线 tokenization
        tokens, target = tokenizer.tokenize_conversation(
            conversation,
            return_target=True
        )
        
        # Padding 和格式化
        # ...
        return {
            'tokens': tokens,
            'labels': target,
            # ...
        }
```

---

## 3. 数据加载架构

### 3.1 三层数据架构

```
┌─────────────────────────────────────────────────────────────┐
│                    BlendedDataset                            │
│              (顶层：混合多个数据集)                            │
└───────────────────────────┬─────────────────────────────────┘
                            │
        ┌───────────────────┼───────────────────┐
        │                   │                     │
        ▼                   ▼                     ▼
┌──────────────┐    ┌──────────────┐    ┌──────────────┐
│ GPTDataset   │    │ GPTDataset   │    │ GPTDataset   │
│ (数据集 1)    │    │ (数据集 2)    │    │ (数据集 3)    │
└──────┬───────┘    └──────┬───────┘    └──────┬───────┘
       │                   │                     │
       └───────────────────┼─────────────────────┘
                           │
                           ▼
                  ┌─────────────────┐
                  │ IndexedDataset   │
                  │ (底层：二进制数据) │
                  └─────────────────┘
```

### 3.2 数据加载流程

```
1. 构建数据集
   build_train_valid_test_datasets()
       ↓
   BlendedMegatronDatasetBuilder.build()
       ├─> 构建底层 IndexedDataset
       ├─> 构建中层 MegatronDataset（GPTDataset）
       └─> 构建顶层 BlendedDataset
       ↓

2. 构建数据加载器
   build_pretraining_data_loader(dataset, consumed_samples)
       ├─> 选择采样器（根据 dataloader_type）
       └─> 创建 PyTorch DataLoader
       ↓

3. 训练时迭代
   for batch in train_dataloader:
       ├─> 采样器生成样本索引
       ├─> Dataset.__getitem__(idx)
       │   ├─> 通过 shuffle_index 映射
       │   ├─> 从 IndexedDataset 读取数据
       │   ├─> 构建序列（可能跨文档）
       │   └─> 生成 masks 和 position_ids
       └─> 返回 batch
```

### 3.3 GPTDataset 内部索引结构

GPTDataset 维护三个关键索引：

#### 1. Document Index（文档索引）

```python
# 形状：[num_epochs * num_documents]
# 内容：文档 ID 的随机排列
document_index = [5, 2, 8, 1, 3, 7, 4, 6, ...]  # 已 shuffle
```

**构建逻辑**：
```python
def _build_document_index(documents, num_epochs, random_state):
    # 为每个 epoch 复制文档索引
    document_index = numpy.mgrid[0:num_epochs, 0:len(documents)][1]
    document_index[:] = documents
    document_index = document_index.reshape(-1)
    
    # 全局 shuffle
    random_state.shuffle(document_index)
    return document_index
```

#### 2. Sample Index（样本索引）

```python
# 形状：[num_samples + 1, 2]
# 内容：[document_idx, offset] 对，标记每个样本的起始位置
sample_index = [
    [0, 0],      # 样本 0：文档 0，偏移 0
    [0, 512],    # 样本 1：文档 0，偏移 512
    [1, 0],      # 样本 2：文档 1，偏移 0
    [1, 1024],   # 样本 3：文档 1，偏移 1024
    ...
]
```

**构建逻辑**（C++ 实现）：
```python
# 使用 C++ 辅助函数构建
from megatron.core.datasets import helpers

sample_index = helpers.build_sample_idx(
    sequence_lengths,      # 每个序列的长度
    document_index,        # 文档索引
    sequence_length,       # 目标序列长度
    num_epochs,           # epoch 数量
    num_tokens_per_epoch, # 每个 epoch 的 token 数
    drop_last_partial_sequence,  # 是否丢弃最后一个不完整序列
    add_extra_token_to_sequence, # 是否添加额外 token
)
```

#### 3. Shuffle Index（打乱索引）

```python
# 形状：[num_samples]
# 内容：样本索引的随机排列
shuffle_index = [42, 15, 8, 33, 1, 27, ...]  # 已 shuffle
```

**构建逻辑**：
```python
def _build_shuffle_index(num_samples, total_size, random_state):
    # 构建 [0, num_samples) 的随机排列
    shuffle_idx = numpy.arange(0, num_samples)
    random_state.shuffle(shuffle_idx)
    
    # 如果 total_size > num_samples，单独 shuffle 剩余部分
    if total_size > num_samples:
        shuffle_idx_last = numpy.arange(num_samples, total_size)
        random_state.shuffle(shuffle_idx_last)
        return numpy.concatenate([shuffle_idx, shuffle_idx_last])
    
    return shuffle_idx
```

### 3.4 数据访问流程

```python
def __getitem__(self, idx: int):
    # 1. 通过 shuffle_index 映射
    shuffled_idx = self.shuffle_index[idx]
    
    # 2. 获取样本的文档和偏移信息
    doc_idx_beg, offset_beg = self.sample_index[shuffled_idx]
    doc_idx_end, offset_end = self.sample_index[shuffled_idx + 1]
    
    # 3. 从 IndexedDataset 读取数据
    sample_parts = []
    if doc_idx_beg == doc_idx_end:
        # 样本在单个文档内
        data = self.dataset.get(
            self.document_index[doc_idx_beg],
            offset=offset_beg,
            length=offset_end - offset_beg
        )
        sample_parts.append(data)
    else:
        # 样本跨多个文档
        for doc_idx in range(doc_idx_beg, doc_idx_end + 1):
            offset = 0 if doc_idx > doc_idx_beg else offset_beg
            length = None if doc_idx < doc_idx_end else offset_end
            data = self.dataset.get(
                self.document_index[doc_idx],
                offset=offset,
                length=length
            )
            sample_parts.append(data)
    
    # 4. 拼接和 padding
    text = numpy.concatenate(sample_parts)
    if len(text) < sequence_length:
        text = numpy.pad(text, (0, sequence_length - len(text)), constant_values=pad_token_id)
    
    # 5. 生成 masks 和 position_ids
    attention_mask, loss_mask, position_ids = _get_ltor_masks_and_position_ids(...)
    
    return {
        'tokens': tokens,
        'labels': labels,
        'attention_mask': attention_mask,
        'loss_mask': loss_mask,
        'position_ids': position_ids,
    }
```

---

## 4. 数据加载器类型

### 4.1 数据加载器类型概览

Megatron 支持三种数据加载器类型（通过 `--dataloader-type` 指定）：

| 类型 | 采样器 | Shuffle | 适用场景 |
|------|--------|---------|---------|
| **`single`** | `MegatronPretrainingSampler` | 数据集内部 shuffle | 标准预训练 |
| **`cyclic`** | `MegatronPretrainingRandomSampler` | 每个 epoch 随机 | 需要更强随机性 |
| **`external`** | 用户提供 | 用户控制 | 自定义数据加载 |

### 4.2 `single` 模式 - 顺序采样

#### 特点

- **顺序访问**：按数据集构建的顺序访问样本
- **数据集级 Shuffle**：在构建数据集时通过 `shuffle_index` 打乱
- **支持断点续训**：通过 `consumed_samples` 跳过已处理数据
- **高效**：无需运行时 shuffle，开销小

#### 采样逻辑

```python
class MegatronPretrainingSampler:
    def __iter__(self):
        batch = []
        # 从 consumed_samples 开始
        for idx in range(self.consumed_samples, self.total_samples):
            batch.append(idx)
            
            # 当收集到全局批次大小时
            if len(batch) == self.micro_batch_times_data_parallel_size:
                # 计算当前 DP rank 的样本范围
                start_idx = self.data_parallel_rank * self.micro_batch_size
                end_idx = start_idx + self.micro_batch_size
                yield batch[start_idx:end_idx]
                batch = []
```

#### 数据流示例

```
数据集（已通过 shuffle_index 打乱）：
[42, 15, 8, 33, 1, 27, 19, 5, ...]

DP size = 4, micro_batch_size = 2
全局批次大小 = 4 * 2 = 8

批次 0：
  全局索引：[42, 15, 8, 33, 1, 27, 19, 5]
  DP rank 0: [42, 15]
  DP rank 1: [8, 33]
  DP rank 2: [1, 27]
  DP rank 3: [19, 5]

批次 1：
  全局索引：[12, 38, 24, 9, 31, 16, 7, 44]
  DP rank 0: [12, 38]
  ...
```

### 4.3 `cyclic` 模式 - 随机采样

#### 特点

- **每个 Epoch 随机**：每个 epoch 重新打乱数据
- **支持数据分片**：可以按 DP rank 分片数据
- **更强随机性**：适合需要数据随机化的场景

#### 采样逻辑

```python
class MegatronPretrainingRandomSampler:
    def __iter__(self):
        # 计算当前 epoch
        active_total_samples = self.total_samples - self.last_batch_size
        self.epoch = self.consumed_samples // active_total_samples
        current_epoch_samples = self.consumed_samples % active_total_samples
        
        if self.data_sharding:
            # 数据分片模式：每个 DP rank 处理一个数据桶
            bucket_size = (self.total_samples // self.micro_batch_times_data_parallel_size) * self.micro_batch_size
            start_idx = self.data_parallel_rank * bucket_size
            
            # 在桶内随机打乱
            g = torch.Generator()
            g.manual_seed(self.epoch)
            random_idx = torch.randperm(bucket_size, generator=g).tolist()
            idx_range = [start_idx + x for x in random_idx[bucket_offset:]]
        else:
            # 全局随机模式：全局打乱后按 DP rank 交错采样
            full_bucket_size = (self.total_samples // self.micro_batch_size) * self.micro_batch_size
            g = torch.Generator()
            g.manual_seed(self.epoch)
            idx_range_total = torch.randperm(full_bucket_size, generator=g).tolist()
            idx_range = idx_range_total[self.data_parallel_rank :: self.data_parallel_size]
        
        # 生成批次
        for idx in idx_range:
            batch.append(idx)
            if len(batch) == self.micro_batch_size:
                yield batch
                batch = []
```

#### 数据分片模式对比

**`data_sharding=True`（桶模式）**：
```
数据集：[0, 1, 2, ..., 9999]
DP size = 4, bucket_size = 2500

DP rank 0: 处理 [0, 2500)     # 桶 0
DP rank 1: 处理 [2500, 5000)   # 桶 1
DP rank 2: 处理 [5000, 7500)   # 桶 2
DP rank 3: 处理 [7500, 10000)  # 桶 3

每个 rank 在自己的桶内随机打乱
```

**`data_sharding=False`（全局随机）**：
```
数据集：[0, 1, 2, ..., 9999]
全局随机打乱：[42, 15, 8, 33, 1, 27, ...]

DP rank 0: [42, 8, 1, ...]      # 索引 0, 4, 8, ...
DP rank 1: [15, 33, 27, ...]    # 索引 1, 5, 9, ...
DP rank 2: [8, 1, ...]          # 索引 2, 6, 10, ...
DP rank 3: [33, 27, ...]        # 索引 3, 7, 11, ...
```

### 4.4 `external` 模式 - 外部数据加载器

#### 特点

- **完全自定义**：用户提供自己的数据加载器
- **灵活性最高**：可以处理任何数据格式
- **需要手动处理分布式**：用户需要确保数据并行分片

#### 使用示例

```python
class CustomDataLoader:
    def __init__(self, dataset_path, ...):
        # 自定义初始化
        pass
    
    def __iter__(self):
        # 自定义迭代逻辑
        for batch in self.custom_iteration():
            yield batch

# 在训练脚本中
args.dataloader_type = 'external'
train_dataloader = CustomDataLoader(...)
```

---

## 5. Shuffle 机制

### 5.1 多层 Shuffle 架构

Megatron 的数据 shuffle 发生在多个层次：

```
┌─────────────────────────────────────────────────────────────┐
│                    Shuffle 层次                              │
└─────────────────────────────────────────────────────────────┘

层次 1: 数据集构建时（GPTDataset）
    ├─> Document Index Shuffle
    │   └─> 文档顺序的随机排列
    │
    └─> Shuffle Index
        └─> 样本索引的随机排列

层次 2: 数据加载器（Sampler）
    ├─> single 模式：使用数据集 shuffle_index
    │
    └─> cyclic 模式：每个 epoch 重新随机打乱

层次 3: 数据并行分片
    └─> 每个 DP rank 处理不同的数据子集
```

### 5.2 数据集级 Shuffle（GPTDataset）

#### Document Index Shuffle

**目的**：打乱文档顺序，避免相邻样本来自相同文档

**实现**：
```python
def _build_document_index(documents, num_epochs, random_state):
    # 为每个 epoch 复制文档
    document_index = numpy.mgrid[0:num_epochs, 0:len(documents)][1]
    document_index[:] = documents
    document_index = document_index.reshape(-1)
    
    # 全局 shuffle
    random_state.shuffle(document_index)
    return document_index
```

**示例**：
```
原始文档：[0, 1, 2, 3, 4]
num_epochs = 2

Shuffle 前：
[0, 1, 2, 3, 4, 0, 1, 2, 3, 4]  # epoch 0 + epoch 1

Shuffle 后：
[3, 1, 4, 0, 2, 2, 4, 1, 3, 0]  # 随机排列
```

#### Shuffle Index

**目的**：打乱样本访问顺序

**实现**：
```python
def _build_shuffle_index(num_samples, total_size, random_state):
    # 构建 [0, num_samples) 的随机排列
    shuffle_idx = numpy.arange(0, num_samples)
    random_state.shuffle(shuffle_idx)
    return shuffle_idx
```

**访问流程**：
```python
def __getitem__(self, idx: int):
    # 通过 shuffle_index 映射到实际样本索引
    actual_idx = self.shuffle_index[idx]
    # 使用 actual_idx 访问 sample_index
    doc_idx, offset = self.sample_index[actual_idx]
    # ...
```

### 5.3 采样器级 Shuffle（cyclic 模式）

#### Epoch-based Shuffle

**特点**：
- 每个 epoch 使用不同的随机种子
- 确保每个 epoch 的数据顺序不同

**实现**：
```python
def __iter__(self):
    # 计算当前 epoch
    self.epoch = self.consumed_samples // active_total_samples
    
    # 使用 epoch 作为随机种子
    g = torch.Generator()
    g.manual_seed(self.epoch)
    
    # 每个 epoch 重新打乱
    idx_range = torch.randperm(full_bucket_size, generator=g).tolist()
```

### 5.4 Shuffle 缓存机制

#### 索引缓存

GPTDataset 可以将 shuffle 索引缓存到磁盘：

```python
# 缓存路径
path_to_shuffle_index = f"{cache_dir}/{hash}-shuffle_index.npy"

# 保存
numpy.save(path_to_shuffle_index, shuffle_index, allow_pickle=True)

# 加载（延迟内存映射）
shuffle_index = numpy.load(
    path_to_shuffle_index,
    allow_pickle=True,
    mmap_mode='r'  # 只读内存映射
)
```

**优点**：
- 避免每次重新构建索引
- 使用内存映射减少内存占用
- 加速数据集初始化

---

## 6. 分布式数据加载

### 6.1 数据并行分片原理

#### 核心概念

在分布式训练中，每个数据并行 rank 需要处理**不同的数据子集**，避免数据重复。

#### 分片计算

```python
# 全局批次大小
global_batch_size = micro_batch_size * data_parallel_size

# 每个 DP rank 的样本范围
def get_start_end_idx(self):
    start_idx = self.data_parallel_rank * self.micro_batch_size
    end_idx = start_idx + self.micro_batch_size
    return start_idx, end_idx
```

#### 分片示例

```
全局批次：[0, 1, 2, 3, 4, 5, 6, 7]
DP size = 4, micro_batch_size = 2

DP rank 0: [0, 1]  # start_idx=0, end_idx=2
DP rank 1: [2, 3]  # start_idx=2, end_idx=4
DP rank 2: [4, 5]  # start_idx=4, end_idx=6
DP rank 3: [6, 7]  # start_idx=6, end_idx=8
```

### 6.2 分布式数据加载流程

```
┌─────────────────────────────────────────────────────────────┐
│              分布式数据加载流程                                │
└─────────────────────────────────────────────────────────────┘

1. 数据集构建（所有 rank 同步）
   ├─> Rank 0 构建数据集（或所有 rank 并行构建）
   └─> 广播数据集元数据

2. 采样器初始化（每个 rank 独立）
   ├─> 获取当前 DP rank
   ├─> 获取 consumed_samples
   └─> 创建采样器实例

3. 数据加载（每个 rank 独立）
   ├─> 采样器生成当前 rank 的样本索引
   ├─> Dataset.__getitem__(idx) 获取数据
   └─> 返回当前 rank 的 batch

4. 梯度同步（训练时）
   └─> All-Reduce 跨 DP rank 同步梯度
```

### 6.3 数据并行同步点

#### 同步点 1：数据集构建

```python
# 在 BlendedMegatronDatasetBuilder 中
if synchronize_ranks:
    # Rank 0 先构建
    if torch.distributed.get_rank() == 0:
        dataset = build_dataset(...)
    torch.distributed.barrier()
    
    # 其他 rank 构建（可能使用缓存）
    if torch.distributed.get_rank() != 0:
        dataset = build_dataset(...)
    torch.distributed.barrier()
```

#### 同步点 2：数据加载器初始化

```python
# 所有 rank 同时初始化
train_dataloader = build_pretraining_data_loader(
    train_ds,
    args.consumed_train_samples  # 所有 rank 使用相同的 consumed_samples
)
```

#### 同步点 3：训练循环

```python
# 每个 iteration 后同步
for iteration in range(args.train_iters):
    # 前向和后向传播（每个 rank 处理不同数据）
    loss = forward_backward_func(...)
    
    # 梯度同步（All-Reduce）
    optimizer.step()
    
    # 更新 consumed_samples（所有 rank 同步更新）
    args.consumed_train_samples += global_batch_size
```

### 6.4 数据并行一致性保证

#### 关键保证

1. **相同的 consumed_samples**：所有 DP rank 使用相同的 `consumed_samples` 值
2. **不同的数据子集**：每个 rank 通过 `data_parallel_rank` 获取不同的样本
3. **同步的批次边界**：所有 rank 的批次边界对齐

#### 验证机制

```python
# 在训练循环中验证
if args.check_data_consistency:
    # 收集所有 rank 的 consumed_samples
    consumed_samples_tensor = torch.tensor(
        [args.consumed_train_samples],
        dtype=torch.long,
        device='cuda'
    )
    torch.distributed.all_reduce(
        consumed_samples_tensor,
        op=torch.distributed.ReduceOp.MAX
    )
    assert consumed_samples_tensor[0] == args.consumed_train_samples
```

---

## 7. 断点续训机制

### 7.1 断点续训概述

Megatron 通过跟踪**已消费的样本数**（`consumed_samples`）实现断点续训。

### 7.2 consumed_samples 跟踪

#### 保存到检查点

```python
# 在 save_checkpoint 中
state_dict = {
    'args': args,  # 包含 consumed_train_samples
    'iteration': iteration,
    # ...
}

# args.consumed_train_samples 会被保存
```

#### 从检查点恢复

```python
# 在 load_checkpoint 中
if 'args' in state_dict:
    checkpoint_args = state_dict['args']
    args.consumed_train_samples = getattr(
        checkpoint_args,
        'consumed_train_samples',
        0
    )
    args.consumed_valid_samples = getattr(
        checkpoint_args,
        'consumed_valid_samples',
        0
    )
```

### 7.3 断点续训流程

```
训练开始
    ↓
加载检查点
    ├─> 恢复 consumed_train_samples
    └─> 恢复 iteration
    ↓
构建数据加载器
    ├─> 使用 consumed_train_samples 初始化采样器
    └─> 采样器从 consumed_samples 开始采样
    ↓
训练循环
    ├─> 处理数据（从 consumed_samples 开始）
    ├─> 更新 consumed_train_samples
    └─> 定期保存检查点
```

### 7.4 consumed_samples 更新

#### 更新时机

```python
# 在训练循环中
for iteration in range(args.train_iters):
    # 前向和后向传播
    forward_backward_func(...)
    
    # 更新 consumed_samples
    batch_size = (
        mpu.get_data_parallel_world_size() *
        args.micro_batch_size *
        get_num_microbatches()
    )
    args.consumed_train_samples += batch_size
```

#### 更新逻辑

**关键点**：
- `consumed_train_samples` 是**全局样本数**（所有 DP rank 共享）
- 每次 iteration 增加 `global_batch_size`
- 所有 DP rank 同步更新

**示例**：
```
初始状态：
  consumed_train_samples = 1000
  global_batch_size = 32

Iteration 0:
  处理样本：[1000, 1001, ..., 1031]
  consumed_train_samples = 1000 + 32 = 1032

Iteration 1:
  处理样本：[1032, 1033, ..., 1063]
  consumed_train_samples = 1032 + 32 = 1064
```

### 7.5 采样器中的 consumed_samples 使用

#### MegatronPretrainingSampler

```python
def __iter__(self):
    batch = []
    # 从 consumed_samples 开始采样
    for idx in range(self.consumed_samples, self.total_samples):
        batch.append(idx)
        if len(batch) == self.micro_batch_times_data_parallel_size:
            yield batch[start_idx:end_idx]
            batch = []
```

#### MegatronPretrainingRandomSampler

```python
def __iter__(self):
    # 计算当前 epoch 和 epoch 内的偏移
    active_total_samples = self.total_samples - self.last_batch_size
    self.epoch = self.consumed_samples // active_total_samples
    current_epoch_samples = self.consumed_samples % active_total_samples
    
    # 从 current_epoch_samples 开始采样
    idx_range = idx_range_total[current_epoch_samples:]
    # ...
```

### 7.6 断点续训注意事项

#### 1. 数据集一致性

**要求**：恢复训练时使用的数据集必须与保存检查点时相同

**检查**：
```python
# 在 load_checkpoint 中
check_checkpoint_args(checkpoint_args)
# 验证关键参数（num_layers, hidden_size 等）是否匹配
```

#### 2. 数据顺序一致性

**问题**：如果数据集构建时的随机种子不同，shuffle_index 会不同

**解决**：
- 使用相同的 `random_seed`
- 或使用缓存的索引文件（`path_to_cache`）

#### 3. 批次大小一致性

**要求**：恢复训练时的 `global_batch_size` 应该与保存时相同

**注意**：
- 如果改变了 DP size，需要相应调整 `micro_batch_size`
- 或使用 `rampup_batch_size` 支持动态批次大小

### 7.7 断点续训示例

#### 场景：训练中断后恢复

```python
# 第一次训练
python pretrain_gpt.py \
    --data-path /path/to/data \
    --save /path/to/checkpoints \
    --train-samples 1000000 \
    --save-interval 1000

# 训练到 iteration 500 时中断
# 检查点保存了：
#   - iteration = 500
#   - consumed_train_samples = 500 * 32 = 16000

# 恢复训练
python pretrain_gpt.py \
    --data-path /path/to/data \
    --load /path/to/checkpoints \
    --train-samples 1000000 \
    --save-interval 1000

# 自动恢复：
#   - iteration = 500
#   - consumed_train_samples = 16000
#   - 数据加载器从样本 16000 开始
```

---

## 8. 性能优化建议

### 8.1 数据格式选择

#### 大规模预训练（推荐：二进制索引数据集）

**原因**：
- 高吞吐量（1000+ samples/s）
- 低 I/O 开销（内存映射）
- 支持大规模数据

**配置**：
```bash
# 预处理
python tools/preprocess_data.py \
    --input data.jsonl \
    --output-prefix data \
    --workers 32 \
    --partitions 4

# 训练
--data-path data \
--mmap-bin-files True \
--fast-cache-load True
```

#### 小规模实验（推荐：在线 Tokenization）

**原因**：
- 无需预处理
- 灵活性高
- 快速迭代

**配置**：
```python
# 使用 SFTDataset 或自定义 Iterable Dataset
class CustomDataset(Iterable):
    def __iter__(self):
        for item in self.data_source:
            yield tokenize(item)
```

### 8.2 数据加载器配置

#### 工作进程数

```python
# 推荐配置
num_workers = min(8, CPU核心数 / DP_size)

# 示例
# 32 个 CPU 核心，DP size = 4
# num_workers = min(8, 32/4) = 8
```

#### 内存固定

```python
# 启用 pin_memory 加速 CPU → GPU 传输
pin_memory=True

# 启用 persistent_workers 避免重复初始化
persistent_workers=True if num_workers > 0 else False
```

### 8.3 缓存优化

#### 索引缓存

```bash
# 启用索引缓存
--path-to-cache /path/to/cache

# 快速缓存加载（跳过验证）
--fast-cache-load
```

#### 内存映射

```bash
# 启用 .bin 文件内存映射
--mmap-bin-files

# 延迟索引内存映射
--defer-npy-index-mmap
```

### 8.4 吞吐量优化技巧

#### 1. 预处理优化

- **多进程并行**：`workers * partitions = CPU核心数`
- **文件分片**：处理超大文件时使用分片
- **压缩存储**：使用 gzip 压缩 JSONL 文件

#### 2. 数据加载优化

- **批量读取**：使用合适的 `micro_batch_size`
- **预取**：PyTorch DataLoader 自动预取下一个 batch
- **异步 I/O**：使用多进程数据加载器

#### 3. 存储优化

- **使用 SSD**：提升 I/O 性能
- **对象存储**：支持 S3 等对象存储（需要配置）
- **本地缓存**：对于远程数据，使用本地缓存

### 8.5 性能基准测试

#### 测试数据加载吞吐量

```python
import time

def benchmark_dataloader(dataloader, num_batches=100):
    start_time = time.time()
    for i, batch in enumerate(dataloader):
        if i >= num_batches:
            break
    end_time = time.time()
    
    throughput = num_batches / (end_time - start_time)
    print(f"Throughput: {throughput:.2f} batches/s")
```

#### 典型性能指标

| 配置 | 吞吐量（samples/s） | 说明 |
|------|-------------------|------|
| 二进制索引 + mmap | 1000-2000 | 最优配置 |
| 二进制索引（无 mmap） | 500-1000 | 次优配置 |
| JSONL 在线 | 200-500 | 中等性能 |
| Parquet 在线 | 300-600 | 中等性能 |

**注意**：实际性能取决于硬件、数据大小、序列长度等因素。

---

## 总结

本文档详细介绍了 Megatron-LM 的预训练数据流：

1. **数据格式**：二进制索引数据集 vs 在线 Tokenization
2. **数据预处理**：JSONL → 二进制格式的完整流程
3. **数据加载架构**：三层架构（BlendedDataset → MegatronDataset → IndexedDataset）
4. **数据加载器**：single、cyclic、external 三种模式
5. **Shuffle 机制**：数据集级和采样器级的多层 shuffle
6. **分布式数据加载**：数据并行分片和同步机制
7. **断点续训**：通过 consumed_samples 实现精确恢复
8. **性能优化**：数据格式选择、配置优化、缓存策略

这些机制共同确保了 Megatron-LM 在大规模预训练中的高效数据加载和处理能力。

