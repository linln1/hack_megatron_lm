# Megatron-LM 工具集详细指南

> 深入解析 `tools/` 目录下的核心工具，包括数据预处理、数据集合并、Retro 工具、推理性能测试和内存报告等。

---

## 目录

1. [preprocess_data.py - 数据预处理工具](#1-preprocess_datapy---数据预处理工具)
2. [merge_datasets.py - 数据集合并工具](#2-merge_datasetspy---数据集合并工具)
3. [retro/ - Retro 检索增强工具](#3-retro---retro-检索增强工具)
4. [run_inference_performance_test.py - 推理性能测试](#4-run_inference_performance_testpy---推理性能测试)
5. [report_theoretical_memory.py - 理论内存报告](#5-report_theoretical_memorypy---理论内存报告)

---

## 1. preprocess_data.py - 数据预处理工具

### 1.1 功能概述

`preprocess_data.py` 是 Megatron-LM 的**核心数据预处理工具**，用于将原始 JSON/JSONL 格式的文本数据转换为 Megatron 训练所需的二进制索引数据集格式（`.bin` 和 `.idx` 文件）。

### 1.2 主要功能

1. **JSON/JSONL 文件读取**：支持普通文件和 gzip 压缩文件
2. **句子分割**：可选使用 NLTK 进行句子分割
3. **Tokenization**：使用 Megatron tokenizer 进行编码
4. **多进程并行处理**：支持多进程加速预处理
5. **文件分片**：支持将大文件分片处理
6. **二进制数据集构建**：生成 `.bin` 和 `.idx` 文件

### 1.3 处理流程

```
原始 JSON/JSONL 文件
    ↓
[可选] 文件分片 (partitions > 1)
    ↓
[可选] 句子分割 (--split-sentences)
    ↓
Tokenization (多进程并行)
    ↓
构建二进制数据集 (.bin + .idx)
    ↓
[可选] 合并分片结果
```

### 1.4 核心类和方法

#### `Encoder` 类

**功能**：负责文本编码和句子分割

**关键方法**：

1. **`initializer()`**：初始化 tokenizer 和句子分割器
```python
def initializer(self):
    # 初始化 tokenizer（支持 legacy 和新的 tokenizer 系统）
    if self.args.legacy_tokenizer:
        Encoder.tokenizer = build_tokenizer(self.args)
    else:
        Encoder.tokenizer = build_new_tokenizer(self.args)
    
    # 初始化句子分割器（如果启用）
    if self.args.split_sentences:
        splitter = nltk.load(...)
        Encoder.splitter = splitter
    else:
        Encoder.splitter = IdentitySplitter()
```

2. **`split(json_line)`**：句子分割
```python
def split(self, json_line):
    data = json.loads(json_line)
    output = {}
    for key in self.args.json_keys:
        text = data[key]
        # 使用 NLTK 分割句子
        tokens_list = [Encoder.splitter.tokenize(text[i:i+max_len]) 
                      for i in range(0, len(text), max_len)]
        output[key] = [tokens for partial in tokens_list for tokens in partial]
    return json.dumps(output), len(json_line)
```

3. **`encode(json_line)`**：Tokenization
```python
def encode(self, json_line):
    data = json.loads(json_line)
    ids = {}
    lens = {}
    for key in self.args.json_keys:
        text = data[key]
        sentences = [text] if isinstance(text, str) else text
        doc_ids = []
        sentence_lens = []
        for sentence in sentences:
            sentence_ids = Encoder.tokenizer.tokenize(sentence)
            if len(sentence_ids) > 0:
                doc_ids.extend(sentence_ids)
                sentence_lens.append(len(sentence_ids))
        
        # 可选：添加 EOD token
        if len(doc_ids) > 0 and self.args.append_eod:
            doc_ids.append(Encoder.tokenizer.eod)
            sentence_lens[-1] += 1
        
        ids[key] = doc_ids
        lens[key] = sentence_lens
    return ids, lens, len(json_line)
```

#### `Partition` 类

**功能**：管理文件分片和并行处理

**关键方法**：

1. **`split_sentences(file_name)`**：句子分割处理
```python
def split_sentences(self, file_name):
    input_file_name, output_file_name = file_name
    encoder = Encoder(self.args)
    # 使用多进程池并行处理
    pool = multiprocessing.Pool(self.workers, initializer=encoder.initializer)
    split_docs = pool.imap(encoder.split, fin, 32)
    
    # 写入分割后的文件
    for doc, bytes_processed in split_docs:
        fout.write(doc + "\n")
```

2. **`process_json_file(file_name)`**：处理 JSON 文件并生成二进制数据集
```python
def process_json_file(self, file_name):
    input_file_name, output_prefix = file_name
    encoder = Encoder(self.args)
    pool = multiprocessing.Pool(self.workers, initializer=encoder.initializer)
    encoded_docs = pool.imap(encoder.encode, fin, 32)
    
    # 为每个 json_key 创建数据集构建器
    builders = {}
    for key in self.args.json_keys:
        output_bin_file = f"{output_prefix}_{key}_{level}.bin"
        output_idx_file = f"{output_prefix}_{key}_{level}.idx"
        builders[key] = IndexedDatasetBuilder(
            output_bin_file,
            dtype=DType.optimal_dtype(tokenizer.vocab_size),
        )
    
    # 处理编码后的文档
    for doc, sentence_lens, bytes_processed in encoded_docs:
        for key in doc.keys():
            builders[key].add_document(doc[key], sentence_lens[key])
    
    # 完成构建
    for key in builders:
        builders[key].finalize(output_idx_files[key])
```

### 1.5 命令行参数

#### 输入数据参数
```bash
--input <path>              # 输入 JSON/JSONL 文件路径（支持 glob 模式）
--json-keys <keys>         # 从 JSON 中提取的键（默认：['text']）
```

#### Tokenization 参数
```bash
--split-sentences          # 是否分割句子（使用 NLTK）
--keep-newlines            # 分割句子时保留换行符
--append-eod               # 在文档末尾添加 EOD token
--lang <language>          # NLTK 句子分割语言（默认：'english'）
```

#### 输出参数
```bash
--output-prefix <path>     # 输出文件前缀（不含后缀）
```

#### 运行时参数
```bash
--workers <num>            # 工作进程数（建议：workers * partitions = CPU 核心数）
--partitions <num>         # 文件分片数（默认：1）
--log-interval <num>       # 进度日志间隔（默认：1000）
--keep-sequential-samples  # 保持样本顺序（当 partitions > 1 时）
```

#### Tokenizer 参数
```bash
--tokenizer-type <type>    # Tokenizer 类型（GPT2BPETokenizer, GPTSentencePieceTokenizer 等）
--vocab-file <path>        # 词汇表文件
--merge-file <path>        # BPE merge 文件（GPT2 需要）
--tokenizer-model <path>   # SentencePiece 模型文件
```

### 1.6 使用示例

#### 基本用法
```bash
python tools/preprocess_data.py \
    --input /path/to/data.jsonl \
    --output-prefix /path/to/output/data \
    --json-keys text \
    --tokenizer-type GPTSentencePieceTokenizer \
    --tokenizer-model /path/to/tokenizer.model \
    --vocab-file /path/to/vocab.txt \
    --workers 8 \
    --append-eod
```

#### 启用句子分割
```bash
python tools/preprocess_data.py \
    --input /path/to/data.jsonl \
    --output-prefix /path/to/output/data \
    --json-keys text \
    --split-sentences \
    --keep-newlines \
    --tokenizer-type GPTSentencePieceTokenizer \
    --tokenizer-model /path/to/tokenizer.model \
    --vocab-file /path/to/vocab.txt \
    --workers 8
```

#### 处理多个文件（使用分片）
```bash
python tools/preprocess_data.py \
    --input "/path/to/data_*.jsonl" \
    --output-prefix /path/to/output/data \
    --json-keys text \
    --tokenizer-type GPTSentencePieceTokenizer \
    --tokenizer-model /path/to/tokenizer.model \
    --vocab-file /path/to/vocab.txt \
    --workers 32 \
    --partitions 4 \
    --keep-sequential-samples
```

### 1.7 输出文件格式

**单个键（`--json-keys text`）**：
- `output_text_document.bin`：二进制数据文件
- `output_text_document.idx`：索引文件

**多个键（`--json-keys text title`）**：
- `output_text_document.bin` / `output_text_document.idx`
- `output_title_document.bin` / `output_title_document.idx`

**句子级别（`--split-sentences`）**：
- `output_text_sentence.bin` / `output_text_sentence.idx`

### 1.8 性能优化建议

1. **工作进程数**：`workers * partitions` 应该等于可用 CPU 核心数
2. **文件分片**：对于超大文件，使用 `--partitions` 可以并行处理
3. **内存管理**：处理大文件时，分片可以减少内存占用
4. **I/O 优化**：使用 SSD 存储可以显著提升处理速度

### 1.9 注意事项

1. **NLTK 依赖**：使用 `--split-sentences` 需要安装 NLTK 和下载 punkt tokenizer
2. **Tokenizer 兼容性**：确保 tokenizer 类型与训练时使用的 tokenizer 一致
3. **文件格式**：输入文件必须是有效的 JSON/JSONL 格式
4. **内存占用**：处理超大文件时，考虑使用分片避免内存溢出

---

## 2. merge_datasets.py - 数据集合并工具

### 2.1 功能概述

`merge_datasets.py` 用于**合并多个已预处理的二进制数据集**（`.bin` 和 `.idx` 文件）为一个统一的数据集。这对于合并不同来源的数据集或合并分片处理的结果非常有用。

### 2.2 主要功能

1. **自动发现数据集**：扫描目录中的所有 `.bin` 和 `.idx` 文件对
2. **数据集合并**：将多个数据集合并为一个
3. **多模态支持**：支持多模态数据集的合并
4. **索引合并**：正确合并索引文件，保持数据顺序

### 2.3 处理流程

```
输入目录
    ↓
扫描 .bin 和 .idx 文件对
    ↓
按前缀排序
    ↓
创建合并构建器
    ↓
逐个添加数据集索引
    ↓
生成合并后的 .bin 和 .idx 文件
```

### 2.4 核心实现

```python
def main():
    args = get_args()
    
    # 1. 扫描目录，发现所有数据集前缀
    prefixes = set()
    for basename in os.listdir(args.input):
        prefix, ext = os.path.splitext(basename)
        
        # 检查对应的 .bin 或 .idx 文件是否存在
        ext_pair = ".bin" if ext == ".idx" else ".idx"
        assert os.path.isfile(os.path.join(args.input, prefix) + ext_pair)
        
        prefixes.add(prefix)
    
    # 2. 创建合并构建器
    builder = None
    for prefix in sorted(prefixes):
        if builder is None:
            # 第一个数据集：读取 dtype 并创建构建器
            dataset = IndexedDataset(
                os.path.join(args.input, prefix), 
                multimodal=args.multimodal
            )
            builder = IndexedDatasetBuilder(
                get_bin_path(args.output_prefix),
                dtype=dataset.index.dtype,
                multimodal=args.multimodal
            )
            del dataset
        
        # 3. 添加数据集索引
        builder.add_index(os.path.join(args.input, prefix))
    
    # 4. 完成合并
    builder.finalize(get_idx_path(args.output_prefix))
```

### 2.5 命令行参数

```bash
--input <directory>        # 输入目录（包含要合并的 .bin 和 .idx 文件）
--output-prefix <path>     # 输出文件前缀（不含后缀）
--multimodal              # 是否是多模态数据集
```

### 2.6 使用示例

#### 基本用法
```bash
python tools/merge_datasets.py \
    --input /path/to/datasets/ \
    --output-prefix /path/to/merged_dataset
```

#### 多模态数据集
```bash
python tools/merge_datasets.py \
    --input /path/to/multimodal_datasets/ \
    --output-prefix /path/to/merged_multimodal_dataset \
    --multimodal
```

### 2.7 输入目录结构

```
/path/to/datasets/
├── dataset1_text_document.bin
├── dataset1_text_document.idx
├── dataset2_text_document.bin
├── dataset2_text_document.idx
├── dataset3_text_document.bin
└── dataset3_text_document.idx
```

**合并后**：
```
/path/to/
├── merged_dataset_text_document.bin
└── merged_dataset_text_document.idx
```

### 2.8 注意事项

1. **数据类型一致性**：所有要合并的数据集必须使用相同的数据类型（dtype）
2. **文件完整性**：每个 `.bin` 文件必须有对应的 `.idx` 文件
3. **前缀命名**：工具通过文件前缀识别数据集对，确保命名规范
4. **内存占用**：合并大数据集时可能需要较多内存

---

## 3. retro/ - Retro 检索增强工具

### 3.1 Retro 概述

**Retro (Retrieval-Enhanced Transformer)** 是一种检索增强的自回归语言模型，通过从大规模检索数据库中检索相关文档来增强模型的知识表示。

**核心特点**：
- **检索增强预训练**：在预训练阶段集成检索机制
- **大规模检索数据库**：支持从数万亿 token 中检索
- **知识更新灵活性**：通过更新检索数据库更新知识，无需重新训练模型
- **InstructRetro**：支持指令微调的 Retro 变体

### 3.2 Retro 工具目录结构

```
tools/retro/
├── README.md                    # Retro 使用指南
├── build_db.md                  # 检索数据库构建指南
├── preprocess_data.py           # Retro 数据预处理
├── config_utils.py              # 配置工具
├── cli/                         # 命令行接口
│   ├── cli.py                   # 主 CLI 实现
│   └── __main__.py              # CLI 入口
├── sft/                         # 监督微调相关
│   ├── sft_retro.py            # Retro SFT 脚本
│   ├── dataset_conv.py         # 数据集转换
│   └── open_inst.sh            # 指令数据集配置
└── text_generation/             # 文本生成相关
    ├── retro_generation.py      # Retro 生成脚本
    ├── retro_text_generation.py # Retro 文本生成
    ├── retro_api.py             # Retro API
    ├── evaluate.py              # 评估脚本
    └── metrics.py               # 评估指标
```

### 3.3 Retro 工作流程

#### 步骤 1：构建检索数据库

**目的**：从预训练语料中构建检索数据库（chunk database）

**关键组件**：
- **GPT Chunk Dataset**：将预训练语料分割成固定长度的 chunk
- **BERT Embeddings**：使用 BERT 编码器为每个 chunk 生成嵌入向量
- **Faiss Index**：使用 Faiss 构建向量索引

**处理流程**：
```
预训练语料 (GPT Dataset)
    ↓
分割成 Chunks (固定长度)
    ↓
BERT Tokenization & Embedding
    ↓
构建 Faiss 索引
    ↓
保存检索数据库
```

#### 步骤 2：查询检索邻居

**目的**：为预训练语料的每个样本查询检索邻居

**处理流程**：
```
预训练样本
    ↓
提取查询向量（BERT embedding）
    ↓
在 Faiss 索引中检索 Top-K 邻居
    ↓
保存检索邻居（用于预训练）
```

#### 步骤 3：预训练

使用包含检索邻居的数据集进行 Retro 模型预训练。

#### 步骤 4：指令微调（可选）

在预训练模型上进行指令微调，得到 InstructRetro。

### 3.4 核心工具详解

#### `preprocess_data.py` - Retro 数据预处理

**功能**：执行 Retro 数据预处理的各个阶段

**支持的任务**（通过 `--retro-tasks` 指定）：
1. **`build_db`**：构建检索数据库
2. **`train_index`**：训练 Faiss 索引
3. **`add_to_index`**：向索引添加向量
4. **`query`**：查询检索邻居

**关键函数**：

```python
def initialize_megatron_retro():
    """初始化 Megatron 并加载 Retro 配置"""
    # 解析 Retro 项目目录
    retro_project_dir = ...
    
    # 初始化 Megatron
    initialize_megatron(extra_args_provider=add_retro_args)
    
    # 加载 Retro 配置
    config = get_retro_preprocessing_config()
    
    # 保存配置
    save_config(config)
    
    return config
```

**使用示例**：
```bash
python tools/retro/preprocess_data.py \
    --retro-project-dir /path/to/retro_project \
    --retro-tasks build_db,train_index,add_to_index,query \
    --retro-gpt-chunk-length 64 \
    --retro-neighbors 2 \
    --retro-num-neighbors-query 2000
```

#### `cli/cli.py` - Retro 命令行接口

**功能**：提供交互式命令行接口，用于探索 Retro 数据集和检索数据库

**主要功能**：

1. **初始化 Retro 环境**
```python
@classmethod
def init(cls, project_dir: str):
    """初始化 Megatron、tokenizers 和数据集"""
    # 初始化 Megatron args
    args = parse_args(...)
    args.retro_project_dir = project_dir
    set_global_variables(args)
    
    # 加载 Retro 配置
    cls.config = load_retro_config(project_dir)
    
    # 加载检索数据库
    cls.db_dataset = get_db_dataset(project_dir, ...)
    
    # 加载预训练数据集
    cls.pt_datasets = build_train_valid_test_datasets(...)
```

2. **检索数据库操作**
```python
# 获取数据库信息
retro.get_db_num_chunks()              # 数据库中的 chunk 数量
retro.get_db_chunk_text(idx)           # 获取 chunk 文本
retro.get_db_chunk_gpt(idx)           # 获取 chunk GPT tokens
retro.get_db_chunk_bert(idx)          # 获取 chunk BERT tokens
```

3. **预训练语料操作**
```python
# 获取预训练样本
retro.get_pt_sample_text('train', idx)    # 获取训练样本文本
retro.get_pt_sample_gpt('train', idx)     # 获取训练样本 GPT tokens
retro.get_pt_neighbors_text('train', idx) # 获取检索邻居文本
```

**使用示例**：
```python
from tools.retro.cli import retro

# 初始化
retro.init('/path/to/retro_project')

# 探索检索数据库
num_chunks = retro.get_db_num_chunks()
chunk_text = retro.get_db_chunk_text(0)

# 探索预训练数据
sample_text = retro.get_pt_sample_text('train', 0)
neighbors = retro.get_pt_neighbors_text('train', 0)
```

#### `sft/sft_retro.py` - Retro 监督微调

**功能**：对预训练的 Retro 模型进行指令微调

**关键特性**：
- 支持混合指令数据集
- 保持检索机制
- 支持继续训练

**使用示例**：
```bash
bash tools/retro/sft/sft_retro_lm.sh \
    open_inst \              # 数据集混合配置
    843m \                   # 模型大小
    128 \                    # 批次大小
    5e-6 \                   # 学习率
    <path/to/pretrained/retro>  # 预训练检查点路径
```

#### `text_generation/retro_generation.py` - Retro 文本生成

**功能**：使用 Retro 模型进行文本生成

**关键特性**：
- 支持零样本评估
- 支持多种采样策略（greedy, top-k, top-p）
- 支持长文本生成

**使用示例**：
```bash
bash tools/retro/text_generation/retro_generate.sh \
    nq \                     # 任务名称（Natural Questions）
    843m \                   # 模型大小
    greedy \                 # 采样策略
    test \                   # 数据集分割
    0 \                      # 起始索引
    20000 \                  # 结束索引
    1000 \                   # 批次大小
    5 \                      # Top-K 检索数量
    pp1 \                    # 流水线并行大小
    <checkpoint_path> \      # 检查点路径
    2                        # 数据并行大小
```

### 3.5 Retro 配置

Retro 使用 JSON 配置文件管理各种参数：

```json
{
    "retro_gpt_chunk_length": 64,
    "retro_bert_chunk_length": 256,
    "retro_bert_max_chunk_length": 256,
    "retro_num_neighbors": 2,
    "retro_num_neighbors_query": 2000,
    "retro_bert_batch_size": 512,
    "retro_block_size": 100000,
    "retro_index_str": "OPQ64,IVF4194304_HNSW32,PQ64",
    "retro_index_nprobe": 4096,
    "retro_index_load": false,
    "retro_index_train_load": false
}
```

### 3.6 Retro 数据格式

**检索数据库格式**：
- Chunk 数据集：`<prefix>_text_document.bin/idx`
- BERT 嵌入：`<prefix>_bert_embeddings.h5`
- Faiss 索引：`<prefix>.index`

**预训练数据格式**：
- GPT 数据集：`<prefix>_text_document.bin/idx`
- 检索邻居：`<prefix>_neighbors_text_document.bin/idx`

### 3.7 Retro 使用建议

1. **数据库构建**：确保检索数据库覆盖预训练语料
2. **索引配置**：根据数据规模选择合适的 Faiss 索引类型
3. **内存管理**：大规模检索数据库需要足够的内存或使用磁盘索引
4. **检索质量**：调整 `retro_num_neighbors_query` 平衡检索质量和速度

---

## 4. run_inference_performance_test.py - 推理性能测试

### 4.1 功能概述

`run_inference_performance_test.py` 用于**测试和评估 Megatron 模型的推理性能**，包括吞吐量、延迟、内存使用等指标。

### 4.2 主要功能

1. **静态推理引擎测试**：测试标准批处理推理
2. **动态推理引擎测试**：测试动态批处理推理（PagedAttention）
3. **性能指标收集**：
   - **TPOT (Time Per Output Token)**：每个输出 token 的时间
   - **延迟 (Latency)**：端到端生成时间
   - **内存使用**：峰值 GPU 内存占用
   - **吞吐量**：tokens/秒
4. **CUDA Profiling**：支持 CUDA profiler 集成
5. **流式生成**：支持流式 token 生成

### 4.3 支持的推理引擎

#### 静态推理引擎 (`--engine-type static`)

**特点**：
- 固定批次大小
- 简单高效
- 适合批量推理场景

**使用场景**：
- 批量文本生成
- 固定批次大小的推理任务

#### 动态推理引擎 (`--engine-type dynamic`)

**特点**：
- 动态批次管理
- PagedAttention 内存管理
- 支持可变序列长度
- 更高的 GPU 利用率

**使用场景**：
- 在线推理服务
- 可变批次大小的场景
- 需要高效内存管理的场景

### 4.4 核心实现

#### 推理引擎获取

```python
def get_inference_engine(args, model):
    """获取推理引擎"""
    tokenizer = get_tokenizer()
    
    inference_wrapper_config = InferenceWrapperConfig(
        hidden_size=args.hidden_size,
        inference_batch_times_seqlen_threshold=args.inference_batch_times_seqlen_threshold,
        inference_max_requests=args.inference_max_batch_size,
        inference_max_seq_length=args.inference_max_seq_length,
        # ... 更多配置
    )
    
    if args.engine_type == "static":
        # 静态推理引擎
        inference_wrapped_model = GPTInferenceWrapper(model, inference_wrapper_config)
        text_generation_controller = TextGenerationController(
            inference_wrapped_model=inference_wrapped_model,
            tokenizer=tokenizer
        )
        return StaticInferenceEngine(text_generation_controller)
    
    elif args.engine_type == "dynamic":
        # 动态推理引擎
        context = DynamicInferenceContext(
            params_dtype=args.params_dtype,
            num_layers=args.num_layers,
            max_sequence_length=args.inference_max_seq_length,
            buffer_size_gb=args.inference_dynamic_batching_buffer_size_gb,
            # ... 更多配置
        )
        inference_wrapped_model = GPTInferenceWrapper(
            model, inference_wrapper_config, inference_context=context
        )
        text_generation_controller = TextGenerationController(...)
        return DynamicInferenceEngine(
            text_generation_controller,
            context,
            enable_cuda_graph=args.cuda_graph_impl == "local",
        )
```

#### 性能测试主循环

```python
@torch.inference_mode()
def main():
    # 初始化 Megatron
    initialize_megatron(extra_args_provider=add_inference_benchmarking_args)
    
    # 加载模型
    model = get_model(...)
    load_checkpoint(model, None, None)
    model.eval()
    
    # 获取推理引擎
    inference_engine = get_inference_engine(args, model)
    
    # 准备请求
    if args.num_input_tokens is not None:
        # 使用随机 token 生成请求
        requests = []
        for i in range(args.inference_max_batch_size):
            prompt_tokens = get_random_prompt_tokens(tokenizer, args.num_input_tokens)
            requests.append(InferenceRequest(...))
    else:
        # 使用提供的 prompts
        requests = [InferenceRequest(prompt=p) for p in args.prompts]
    
    # CUDA Graph 预热（如果启用）
    if args.cuda_graph_impl == "local":
        inference_engine.generate(prompts=["warmup"], ...)
    
    # 性能测试
    start_time = time.perf_counter()
    
    if args.engine_type == "static":
        results = inference_engine.generate(
            prompts=args.prompts,
            inference_requests=requests,
            sampling_params=sampling_params
        )
    elif args.engine_type == "dynamic":
        results = generate_dynamic(args, requests, inference_engine)
    
    end_time = time.perf_counter()
    latency = end_time - start_time
    
    # 收集性能指标
    memory_allocated = torch.cuda.max_memory_allocated()
    
    # 打印结果
    for result in results:
        print({
            'num_input_tokens': len(result.prompt_tokens),
            'num_output_tokens': len(result.generated_tokens),
            'tpot': result.tpot,  # Time Per Output Token
            'latency': latency,
            'memory_usage_GB': memory_allocated / (1024**3),
        })
```

### 4.5 命令行参数

#### 推理引擎参数
```bash
--engine-type {static,dynamic}    # 推理引擎类型
--inference-max-batch-size <num>  # 最大批次大小
--inference-max-seq-length <num>  # 最大序列长度
```

#### 动态批处理参数（仅 dynamic 引擎）
```bash
--inference-dynamic-batching-buffer-size-gb <size>      # 缓冲区大小（GB）
--inference-dynamic-batching-buffer-guaranteed-fraction <frac>  # 保证分配比例
--inference-dynamic-batching-block-size <size>          # Block 大小（tokens）
--inference-dynamic-batching-num-cuda-graphs <num>      # CUDA Graph 数量
```

#### 测试参数
```bash
--num-input-tokens <num>          # 输入 token 数量（生成随机 prompts）
--prompts <prompts>               # 测试 prompts（与 --num-input-tokens 二选一）
--num-tokens-to-generate <num>    # 生成 token 数量
--benchmark-profile               # 启用 CUDA profiler
--stream                          # 启用流式生成
```

#### 采样参数
```bash
--temperature <temp>              # 采样温度
--top-k <k>                      # Top-K 采样
--top-p <p>                      # Top-P (nucleus) 采样
--return-log-probs               # 返回对数概率
```

### 4.6 使用示例

#### 静态推理引擎测试
```bash
python tools/run_inference_performance_test.py \
    --load /path/to/checkpoint \
    --engine-type static \
    --inference-max-batch-size 8 \
    --inference-max-seq-length 2048 \
    --num-input-tokens 128 \
    --num-tokens-to-generate 512 \
    --temperature 0.0 \
    --top-k 1
```

#### 动态推理引擎测试
```bash
python tools/run_inference_performance_test.py \
    --load /path/to/checkpoint \
    --engine-type dynamic \
    --inference-max-batch-size 32 \
    --inference-max-seq-length 4096 \
    --inference-dynamic-batching-buffer-size-gb 20 \
    --num-input-tokens 128 \
    --num-tokens-to-generate 512 \
    --temperature 0.7 \
    --top-p 0.9
```

#### 使用自定义 Prompts
```bash
python tools/run_inference_performance_test.py \
    --load /path/to/checkpoint \
    --engine-type static \
    --prompts "What is machine learning?" "Explain transformer architecture." \
    --num-tokens-to-generate 256 \
    --temperature 0.7
```

#### 启用 CUDA Profiling
```bash
python tools/run_inference_performance_test.py \
    --load /path/to/checkpoint \
    --engine-type static \
    --num-input-tokens 128 \
    --num-tokens-to-generate 512 \
    --benchmark-profile
```

### 4.7 性能指标解读

#### TPOT (Time Per Output Token)
- **定义**：生成每个输出 token 的平均时间
- **计算**：`总生成时间 / 输出 token 数量`
- **单位**：毫秒 (ms)
- **优化目标**：降低 TPOT 以提高吞吐量

#### 延迟 (Latency)
- **定义**：从输入到完成生成的总时间
- **影响因素**：
  - 输入长度
  - 输出长度
  - 批次大小
  - 模型大小
  - 硬件配置

#### 内存使用
- **定义**：峰值 GPU 内存占用
- **影响因素**：
  - 模型大小
  - 批次大小
  - 序列长度
  - KV Cache 大小（动态引擎）

#### 吞吐量
- **定义**：每秒生成的 token 数量
- **计算**：`输出 token 数量 / 总时间`
- **单位**：tokens/秒

### 4.8 性能优化建议

1. **批次大小**：根据 GPU 内存和模型大小调整批次大小
2. **序列长度**：使用合适的最大序列长度，避免浪费内存
3. **CUDA Graphs**：启用 CUDA Graphs 可以减少内核启动开销
4. **动态批处理**：对于在线服务，使用动态批处理提高 GPU 利用率
5. **混合精度**：使用 FP16/BF16 减少内存占用和提升速度

---

## 5. report_theoretical_memory.py - 理论内存报告

### 5.1 功能概述

`report_theoretical_memory.py` 是一个**轻量级工具**，用于计算和报告模型训练的理论内存占用，**无需实例化模型或运行训练**。

### 5.2 主要功能

1. **理论内存计算**：基于模型配置计算理论内存占用
2. **详细报告**：提供权重、优化器状态、激活内存的详细分解
3. **无需 GPU**：可以在没有 GPU 的环境中运行
4. **快速评估**：快速评估不同配置的内存需求

### 5.3 实现原理

该工具实际上是 `theoretical_memory_usage.py` 模块的简单包装：

```python
from megatron.training import get_args
from megatron.training.initialize import initialize_megatron
from megatron.training.theoretical_memory_usage import report_theoretical_memory

if __name__ == "__main__":
    # 初始化 Megatron（跳过 MPU 初始化，允许无 CUDA）
    initialize_megatron(
        allow_no_cuda=True,
        skip_mpu_initialization=True
    )
    
    args = get_args()
    
    # 计算并报告理论内存
    report_theoretical_memory(args, verbose=True)
```

### 5.4 使用方法

#### 基本用法
```bash
python tools/report_theoretical_memory.py \
    --num-layers 24 \
    --hidden-size 1024 \
    --num-attention-heads 16 \
    --ffn-hidden-size 4096 \
    --seq-length 2048 \
    --micro-batch-size 4 \
    --pipeline-model-parallel-size 1 \
    --tensor-model-parallel-size 1 \
    --use-distributed-optimizer
```

#### 使用 YAML 配置
```bash
python tools/report_theoretical_memory.py \
    --yaml-cfg /path/to/config.yaml
```

#### 详细输出
工具会自动启用 `verbose=True`，输出详细的内存分解信息。

### 5.5 输出示例

```
Number of parameters in transformer block in billions:  0.35
Number of parameters in embedding layers in billions:  0.13
Total number of parameters in billions:  0.48
Number of parameters in most loaded shard in billions:  0.4800
Activation memory footprint per transformer layer:  45.2 MB
Memory penalty from interleaved schedule:  1.00
Number of in-flight microbatches:  1
Theoretical memory footprints: weight and optimizer=8640.00 MB, activation=1084.80 MB, total=9724.80 MB
```

### 5.6 使用场景

1. **配置规划**：在训练前评估不同配置的内存需求
2. **资源分配**：确定需要多少 GPU 和内存
3. **优化决策**：评估不同优化策略（序列并行、激活重计算等）的内存影响
4. **容量规划**：规划训练基础设施

### 5.7 注意事项

1. **理论值**：报告的是理论内存占用，实际值可能因实现细节而有所不同
2. **激活重计算**：如果启用激活重计算，实际激活内存会减少
3. **内存碎片**：实际内存占用可能因内存碎片而略高于理论值
4. **其他开销**：不包括 PyTorch、CUDA 等框架的内存开销

---

## 总结

本文档详细介绍了 Megatron-LM 工具集中的核心工具：

1. **preprocess_data.py**：将原始文本数据转换为训练所需的二进制格式
2. **merge_datasets.py**：合并多个预处理的数据集
3. **retro/**：Retro 检索增强模型的完整工具链
4. **run_inference_performance_test.py**：推理性能测试和评估
5. **report_theoretical_memory.py**：理论内存占用计算

这些工具共同构成了 Megatron-LM 完整的数据处理和评估生态系统，为大规模语言模型的训练和部署提供了强大的支持。

