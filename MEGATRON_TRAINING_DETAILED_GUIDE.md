# Megatron Training 模块详细指南

> 深入解析 `megatron/training/` 目录下的核心代码，包括 YAML 配置、参数处理、初始化、训练循环、检查点、工具函数等。

---

## 目录

1. [YAML 配置文件写法](#1-yaml-配置文件写法)
2. [argument_utils.py - 参数工具](#2-argument_utilspy---参数工具)
3. [initialize.py - 初始化核心](#3-initializepy---初始化核心)
4. [training.py - 训练主循环](#4-trainingpy---训练主循环)
5. [checkpointing.py - 检查点管理](#5-checkpointingpy---检查点管理)
6. [utils.py - 工具函数](#6-utilspy---工具函数)
7. [global_vars.py - 全局变量管理](#7-global_varspy---全局变量管理)
8. [async_utils.py - 异步工具](#8-async_utilspy---异步工具)
9. [dist_signal_handler.py - 信号处理](#9-dist_signal_handlerpy---信号处理)
10. [theoretical_memory_usage.py - 内存计算](#10-theoretical_memory_usagepy---内存计算)
11. [data_samplers.py - 数据采样器](#11-data_samplerspy---数据采样器)

---

## 1. YAML 配置文件写法

### 1.1 YAML 配置概述

Megatron-LM 支持使用 YAML 文件替代命令行参数进行配置。YAML 配置通过 `yaml_arguments.py` 模块处理。

### 1.2 YAML 文件结构

YAML 配置文件采用**嵌套命名空间**结构，主要分为以下几个部分：

```yaml
# 语言模型配置
language_model:
  num_layers: 24
  hidden_size: 1024
  num_attention_heads: 16
  ffn_hidden_size: null  # null 表示使用默认值
  activation_func: swiglu
  # ... 更多配置

# 模型并行配置
model_parallel:
  tensor_model_parallel_size: 1
  pipeline_model_parallel_size: 1
  sequence_parallel: True
  fp16: False
  bf16: True
  # ... 更多配置

# 训练配置（顶层）
use_legacy_models: False
micro_batch_size: 2
global_batch_size: 128
train_iters: null
eval_iters: 32
# ... 更多配置
```

### 1.3 环境变量支持

YAML 文件支持使用环境变量，使用 `${VAR_NAME}` 语法：

```yaml
data_path: ${DATA_PATH}/train_data
save: ${CHECKPOINT_DIR}/checkpoints
```

**注意**：环境变量必须在运行前设置，否则会抛出异常。

### 1.4 YAML 配置处理流程

```python
# yaml_arguments.py 中的处理流程

1. load_yaml(yaml_path)
   └─> 读取 YAML 文件
   └─> 转换为嵌套 SimpleNamespace 对象
   └─> 添加 yaml_cfg 属性指向配置文件路径

2. validate_yaml(args, defaults)
   └─> 验证并行配置（TP/PP/DP/CP/EP）
   └─> 计算数据并行大小
   └─> 验证批次大小设置
   └─> 验证混合精度设置
   └─> 验证激活重计算配置
   └─> 验证 MoE 配置
   └─> 合并嵌套命名空间到顶层

3. core_transformer_config_from_yaml(args)
   └─> 从 YAML 配置构建 TransformerConfig
   └─> 处理激活函数映射（swiglu -> F.silu）
   └─> 处理初始化方法映射
```

### 1.5 关键配置验证规则

#### 并行配置验证
```python
# 验证 TP/PP/CP 的整除关系
assert world_size % (tp_size * pp_size * cp_size) == 0

# 计算数据并行大小
data_parallel_size = world_size // (tp_size * pp_size * cp_size)
```

#### 批次大小验证
```python
# 如果未指定 global_batch_size，自动计算
if global_batch_size is None:
    global_batch_size = micro_batch_size * data_parallel_size
```

#### 混合精度验证
```python
# FP16 和 BF16 互斥
if fp16:
    assert not bf16
    params_dtype = torch.half
if bf16:
    assert not fp16
    params_dtype = torch.bfloat16
    # BF16 需要 FP32 梯度累积
    accumulate_allreduce_grads_in_fp32 = True
```

### 1.6 YAML 配置示例

完整示例见 `examples/gpt3/gpt_config.yaml`：

```yaml
language_model:
  num_layers: 24
  hidden_size: 1024
  num_attention_heads: 16
  activation_func: swiglu
  normalization: "LayerNorm"

model_parallel:
  tensor_model_parallel_size: 1
  pipeline_model_parallel_size: 1
  sequence_parallel: True
  bf16: True

micro_batch_size: 2
global_batch_size: 128
seq_length: 4096
train_iters: null
train_samples: 268554688
```

---

## 2. argument_utils.py - 参数工具

### 2.1 核心功能

`argument_utils.py` 提供了 `ArgumentGroupFactory` 类，用于**自动从 dataclass 生成 argparse 参数**。

### 2.2 ArgumentGroupFactory 类

#### 设计目的
- 减少重复代码：不需要手动为每个配置字段编写 argparse 代码
- 类型推断：自动从 dataclass 类型注解推断 argparse 参数类型
- 文档自动提取：从字段级 docstring 提取帮助文本

#### 核心方法

**1. `_extract_type(config_type)`**
```python
# 支持的类型：
# - 基本类型：int, float, str, bool
# - Optional[T]：自动处理 None
# - list[T]：自动添加 nargs="+"
# - enum.Enum：自动生成 choices
# - typing.Literal：自动生成 choices
```

**2. `_build_argparse_kwargs_from_field(attribute)`**
```python
# 构建 argparse 参数的 kwargs：
# - arg_names: 从字段名生成（--field-name）
# - type: 从类型注解推断
# - default: 从 dataclass default 获取
# - help: 从字段 docstring 提取
# - action: bool 类型自动使用 store_true/store_false
```

**3. `build_group(parser, title)`**
```python
# 为 ArgumentParser 添加参数组
# 遍历 dataclass 的所有字段
# 为每个字段生成对应的命令行参数
```

### 2.3 元数据覆盖机制

可以通过字段的 `metadata` 覆盖自动推断的参数：

```python
@dataclass
class Config:
    your_attribute: int | str | None = field(
        default=None,
        metadata={
            "argparse_meta": {
                "arg_names": ["--your-arg-name1", "--your-arg-name2"],
                "type": str,
                "nargs": "+",
                "default": "foo",
            }
        },
    )
```

### 2.4 使用示例

```python
from dataclasses import dataclass, field
from argparse import ArgumentParser
from megatron.training.argument_utils import ArgumentGroupFactory

@dataclass
class ModelConfig:
    """模型配置"""
    num_layers: int = 24
    """Transformer 层数"""
    
    hidden_size: int = 1024
    """隐藏层大小"""
    
    use_fp16: bool = False
    """是否使用 FP16 混合精度"""

# 使用
parser = ArgumentParser()
factory = ArgumentGroupFactory(ModelConfig)
factory.build_group(parser, title="Model Configuration")

# 自动生成的参数：
# --num-layers (type: int, default: 24)
# --hidden-size (type: int, default: 1024)
# --no-use-fp16 / --disable-use-fp16 (action: store_false, default: False)
```

### 2.5 字段文档字符串提取

`_get_field_docstrings()` 方法通过**AST 解析**提取字段级文档字符串：

```python
# 支持的格式：
@dataclass
class Config:
    field_name: int = 24
    """这是字段的文档字符串"""
```

---

## 3. initialize.py - 初始化核心

### 3.1 初始化流程概览

```
initialize_megatron()
├─> parse_args() / validate_yaml()
├─> set_global_variables()
│   ├─> 设置全局 args
│   ├─> 初始化 tokenizer
│   ├─> 初始化 TensorBoard/WandB/OneLogger
│   ├─> 初始化 ADLR autoresume
│   └─> 初始化 timers
├─> setup_logging()
├─> initialize_rerun_state_machine()
└─> finish_mpu_init()
    ├─> _initialize_distributed()
    ├─> _set_random_seed()
    └─> _compile_dependencies()
```

### 3.2 核心函数详解

#### `initialize_megatron()`

**功能**：Megatron 训练的入口初始化函数

**关键参数**：
- `extra_args_provider`: 额外的参数提供函数
- `args_defaults`: 参数默认值字典
- `allow_no_cuda`: 是否允许无 CUDA 环境（仅用于 CPU 数据处理）
- `skip_mpu_initialization`: 是否跳过 MPU 初始化（用于外部 DDP 管理）
- `lazy_mpu_init`: 延迟初始化 MPU（返回初始化函数供外部调用）

**初始化步骤**：

1. **参数解析与验证**
```python
if parsed_args is None:
    args = parse_args(extra_args_provider, ignore_unknown_args)
else:
    args = parsed_args

if args.yaml_cfg is not None:
    args = validate_yaml(args, args_defaults)
else:
    validate_args(args, args_defaults)
```

2. **全局变量设置**
```python
set_global_variables(args)
# 设置 args, tokenizer, tensorboard, timers 等
```

3. **分布式初始化**
```python
def finish_mpu_init():
    _initialize_distributed(...)  # torch.distributed 初始化
    _set_random_seed(...)          # 随机种子设置
    _compile_dependencies()         # 编译 C++ 依赖
```

#### `_initialize_distributed()`

**功能**：初始化 PyTorch 分布式和 Megatron 模型并行

**关键步骤**：

1. **检查是否已初始化**
```python
if torch.distributed.is_initialized():
    args.rank = torch.distributed.get_rank()
    args.world_size = torch.distributed.get_world_size()
else:
    # 初始化 torch.distributed
    torch.distributed.init_process_group(
        backend=args.distributed_backend,
        world_size=args.world_size,
        rank=args.rank,
        timeout=timedelta(minutes=args.distributed_timeout_minutes),
    )
```

2. **初始化模型并行**
```python
mpu.initialize_model_parallel(
    tensor_model_parallel_size=args.tensor_model_parallel_size,
    pipeline_model_parallel_size=args.pipeline_model_parallel_size,
    virtual_pipeline_model_parallel_size=args.virtual_pipeline_model_parallel_size,
    context_parallel_size=args.context_parallel_size,
    expert_model_parallel_size=args.expert_model_parallel_size,
    # ... 更多配置
)
```

#### `_set_random_seed()`

**功能**：设置随机种子以确保可复现性

**种子计算逻辑**：
```python
# 基础种子
seed = seed_ + (100 * pipeline_model_parallel_rank)

# 数据并行随机初始化（可选）
if data_parallel_random_init:
    seed = seed + (10 * data_parallel_rank)

# 设置各种 RNG
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
tensor_parallel.model_parallel_cuda_manual_seed(seed, ...)
```

**为什么不同 PP 阶段使用不同种子？**
- 确保不同流水线阶段有不同的随机行为
- 避免所有阶段产生相同的随机数序列

#### `_compile_dependencies()`

**功能**：编译 C++ 辅助函数和融合内核

**编译内容**：

1. **数据集 C++ 辅助函数**
```python
from megatron.core.datasets.utils import compile_helpers
compile_helpers()  # 编译索引构建器等
```

2. **融合内核**
```python
from megatron.legacy import fused_kernels
fused_kernels.load(args)  # 加载融合的 softmax、layer norm 等
```

**编译约束检查**：
```python
# 检查是否满足融合内核的约束条件
custom_kernel_constraint = (
    seq_len > 16 and seq_len <= 16384 and 
    seq_len % 4 == 0 and attn_batch_size % 4 == 0
)
```

#### `_initialize_tp_communicators()`

**功能**：初始化张量并行通信器（用于 TP 通信/GEMM 重叠优化）

**关键步骤**：
```python
# 使用 Transformer Engine 的 UserBuffer
te_module.base.initialize_ub(
    shape=input_shape,
    tp_size=args.tensor_model_parallel_size,
    quantization_modes=[...],
    ub_cfgs=ub_cfgs,  # 从 YAML 配置文件读取
    bootstrap_backend=args.tp_comm_bootstrap_backend,
)
```

### 3.3 初始化顺序的重要性

1. **参数解析必须在最前**：所有后续初始化都依赖 args
2. **全局变量设置在分布式初始化前**：tokenizer、timers 等不需要分布式
3. **分布式初始化在模型并行前**：MPU 依赖 torch.distributed
4. **随机种子在模型创建前**：确保模型初始化可复现
5. **依赖编译在模型创建前**：融合内核需要在模型中使用

---

## 4. training.py - 训练主循环

### 4.1 训练流程架构

```
pretrain()
├─> initialize_megatron()
├─> build_model()
├─> build_optimizer()
├─> build_data_loader()
├─> load_checkpoint()
└─> train()
    └─> 训练循环
        ├─> forward_step()
        ├─> backward_step()
        ├─> optimizer.step()
        └─> save_checkpoint()
```

### 4.2 核心函数：`pretrain()`

**功能**：预训练的主入口函数

**关键步骤**：

1. **初始化**
```python
initialize_megatron(extra_args_provider=extra_args_provider)
args = get_args()
```

2. **构建模型**
```python
model = model_provider_fn()
# 包装模型（DDP/FSDP）
if args.use_megatron_fsdp:
    model = wrap_with_megatron_fsdp(model, ...)
else:
    model = wrap_with_ddp(model, ...)
```

3. **构建优化器**
```python
optimizer = get_megatron_optimizer(model, ...)
opt_param_scheduler = OptimizerParamScheduler(optimizer, ...)
```

4. **构建数据加载器**
```python
train_data_iterator = build_pretraining_data_loader(
    train_dataset, args.consumed_train_samples
)
```

5. **加载检查点**
```python
iteration = 0
if checkpoint_exists(args.load):
    iteration = load_checkpoint(model, optimizer, opt_param_scheduler, ...)
```

6. **开始训练**
```python
train(
    forward_step_func=forward_step_func,
    model=model,
    optimizer=optimizer,
    opt_param_scheduler=opt_param_scheduler,
    train_data_iterator=train_data_iterator,
    ...
)
```

### 4.3 核心函数：`train()`

**功能**：训练主循环

**训练循环结构**：

```python
while iteration < args.train_iters:
    # 1. 更新学习率
    opt_param_scheduler.step()
    
    # 2. 前向和后向传播
    losses_reduced = forward_backward_func(
        forward_step_func=forward_step_func,
        data_iterator=train_data_iterator,
        model=model,
        num_microbatches=get_num_microbatches(),
        ...
    )
    
    # 3. 梯度裁剪
    if args.clip_grad > 0:
        grad_norm = clip_grad_norm(model, args.clip_grad)
    
    # 4. 优化器更新
    optimizer.step()
    
    # 5. 更新 consumed_samples
    args.consumed_train_samples += (
        get_current_global_batch_size() * get_num_microbatches()
    )
    
    # 6. 保存检查点
    if iteration % args.save_interval == 0:
        save_checkpoint(iteration, model, optimizer, ...)
    
    # 7. 验证
    if iteration % args.eval_interval == 0:
        evaluate_and_print_results(...)
    
    iteration += 1
```

### 4.4 前向传播函数：`forward_step()`

**功能**：单个微批次的前向传播

**典型实现**：
```python
def forward_step(data_iterator, model):
    # 1. 获取批次数据
    batch = next(data_iterator)
    
    # 2. 前向传播
    output_tensor = model(
        batch['tokens'],
        batch['position_ids'],
        batch['attention_mask'],
    )
    
    # 3. 计算损失
    loss = loss_func(output_tensor, batch['labels'])
    
    # 4. 返回损失字典
    return loss, {'lm_loss': loss}
```

### 4.5 流水线并行调度

Megatron 使用 `get_forward_backward_func()` 获取流水线调度函数：

```python
from megatron.core.pipeline_parallel import get_forward_backward_func

forward_backward_func = get_forward_backward_func(
    args.virtual_pipeline_model_parallel_size,
    args.pipeline_model_parallel_size,
)
```

**支持的调度策略**：
- **1F1B (One Forward One Backward)**：标准流水线调度
- **Interleaved 1F1B**：交错调度（需要 virtual pipeline）

### 4.6 FLOPs 计算

`num_floating_point_operations()` 函数计算理论 FLOPs：

**计算公式**：
```python
# Transformer 层 FLOPs
# - 注意力：12 * batch * seq * hidden^2 * (1 + seq/hidden/2)
# - MLP：12 * batch * seq * hidden * ffn_hidden
# - 总 FLOPs = (注意力 + MLP) * num_layers

# 系数 12 的来源：
# - 3x：前向(1) + 后向 wgrad(1) + 后向 dgrad(1)
# - 2x：每层有两个 GEMM（如 MLP 的 up_proj 和 down_proj）
# - 2x：GEMM 的浮点运算数（m*n*k 需要 2mnk 次运算）
```

---

## 5. checkpointing.py - 检查点管理

### 5.1 检查点类型

Megatron 支持多种检查点格式：

```python
class CheckpointType(Enum):
    LEGACY = auto()      # 传统格式（每个 rank 一个文件）
    LOCAL = auto()       # 本地非持久化检查点
    GLOBAL = auto()      # 全局分布式检查点
    TORCH_DCP = auto()   # PyTorch DCP 格式
    FSDP_DTENSOR = auto() # FSDP DTensor 格式
```

### 5.2 保存检查点：`save_checkpoint()`

**功能**：保存模型、优化器、RNG 状态等

**保存流程**：

1. **准备状态字典**
```python
state_dict = generate_state_dict(
    args, model, optimizer, opt_param_scheduler,
    rng_state, iteration=iteration,
    optim_sd_kwargs=dict(metadata=sharded_sd_metadata),
    model_sd_kwargs=dict(metadata=sharded_sd_metadata),
)
```

2. **选择保存策略**
```python
if ckpt_type == CheckpointType.GLOBAL:
    if ckpt_format == "torch_dist":
        # 使用 Megatron Core 分布式检查点
        dist_checkpointing.save(
            state_dict, checkpoint_name, save_strategy,
            async_sharded_save=args.async_save,
        )
    elif ckpt_format == "torch_dcp":
        # 使用 PyTorch DCP
        torch.distributed.checkpoint.save(
            state_dict=state_dict,
            storage_writer=fs_storage_writer,
        )
elif ckpt_type == CheckpointType.LEGACY:
    # 传统格式：每个 rank 保存自己的文件
    torch.save(state_dict, checkpoint_name)
```

3. **更新跟踪文件**
```python
# 更新 latest_checkpointed_iteration.txt
with open(tracker_filename, 'w') as f:
    f.write(str(iteration))
```

### 5.3 加载检查点：`load_checkpoint()`

**功能**：从检查点恢复训练状态

**加载流程**：

1. **检测检查点格式**
```python
ckpt_format = _get_checkpoint_format(checkpoint_name, args)
# 检查是否存在 metadata.json（分布式检查点）
# 或 mp_rank_* 目录（传统格式）
```

2. **加载状态字典**
```python
if ckpt_format == "torch_dist":
    state_dict = dist_checkpointing.load(
        sharded_state_dict, checkpoint_name, load_strategy,
        strict=args.dist_ckpt_strictness,
    )
elif ckpt_format == "torch":
    state_dict = torch.load(checkpoint_name, map_location='cpu')
```

3. **恢复模型和优化器**
```python
# 恢复模型
model[0].load_state_dict(state_dict['model'], strict=strict)

# 恢复优化器
if not args.finetune:
    optimizer.load_state_dict(state_dict['optimizer'])

# 恢复 RNG 状态
if 'rng_state' in state_dict:
    random.setstate(rng_state['random_rng_state'])
    torch.set_rng_state(rng_state['torch_rng_state'])
    # ...
```

### 5.4 RNG 状态管理

**功能**：确保训练可复现性

**RNG 状态收集**：
```python
rng_state = {
    'random_rng_state': random.getstate(),
    'np_rng_state': np.random.get_state(),
    'torch_rng_state': torch.get_rng_state(),
    'cuda_rng_state': torch.cuda.get_rng_state(),
    'rng_tracker_states': tensor_parallel.get_cuda_rng_tracker().get_states(),
}

# 如果使用数据并行随机初始化，需要收集所有 DP rank 的状态
if args.data_parallel_random_init:
    rng_state_list = [None] * dp_size
    torch.distributed.all_gather_object(
        rng_state_list, rng_state, group=dp_group
    )
```

**RNG 状态分片**（分布式检查点）：
```python
# 将 RNG 状态分片到不同的 PP/TP/DP rank
rng_state_list = ShardedObject(
    'rng_state',
    rng_state_list,
    (pp_size, tp_size, dp_size),
    (pp_rank, tp_rank, dp_rank),
    replica_id=0,
)
```

### 5.5 检查点验证

**功能**：确保检查点与当前配置兼容

```python
def check_checkpoint_args(checkpoint_args):
    # 检查关键参数是否匹配
    _compare('num_layers')
    _compare('hidden_size')
    _compare('num_attention_heads')
    _compare('tensor_model_parallel_size')
    _compare('pipeline_model_parallel_size')
    # ...
```

### 5.6 异步检查点保存

**功能**：在后台异步保存检查点，不阻塞训练

**使用方式**：
```python
if args.async_save:
    async_save_request = dist_checkpointing.save(
        state_dict, checkpoint_name, save_strategy,
        async_sharded_save=True,
    )
    schedule_async_save(async_save_request)
    # 训练可以继续，检查点在后台保存
```

**等待异步保存完成**：
```python
maybe_finalize_async_save(blocking=True)  # 阻塞等待
maybe_finalize_async_save(blocking=False) # 仅完成已完成的请求
```

---

## 6. utils.py - 工具函数

### 6.1 参数 L2 范数计算

**`calc_params_l2_norm(model)`**

**功能**：计算模型参数的 L2 范数（用于梯度裁剪和监控）

**关键逻辑**：

1. **分离不同类型的参数**
```python
params_data = []        # 普通参数
moe_params_data = []    # MoE 专家参数
sharded_params_data = [] # 分片的优化器参数（分布式优化器）
```

2. **处理混合精度**
```python
if args.bf16:
    # 使用 main_param（FP32 主参数）
    params_data.append(param.main_param)
else:
    params_data.append(param.data)
```

3. **计算范数**
```python
# 使用融合的多张量 L2 范数计算
norm, _ = multi_tensor_applier(
    multi_tensor_l2norm, dummy_overflow_buf, [params_data], False
)
norm_2 = norm * norm
```

4. **跨并行组归约**
```python
# 普通参数：跨模型并行组归约
torch.distributed.all_reduce(
    norm_2, op=torch.distributed.ReduceOp.SUM, 
    group=mpu.get_model_parallel_group()
)

# MoE 参数：跨专家+张量+流水线并行组归约
torch.distributed.all_reduce(
    moe_norm_2, op=torch.distributed.ReduceOp.SUM,
    group=mpu.get_expert_tensor_model_pipeline_parallel_group()
)
```

### 6.2 批次数据获取

**`get_batch_on_this_tp_rank(data_iterator)`**

**功能**：在张量并行 rank 0 获取数据，然后广播到其他 TP rank

**关键逻辑**：
```python
if mpu.get_tensor_model_parallel_rank() == 0:
    # TP rank 0 从数据迭代器获取数据
    data = next(data_iterator)
    batch = {
        'tokens': data["tokens"].cuda(non_blocking=True),
        'labels': data["labels"].cuda(non_blocking=True),
        # ...
    }
    
    # 根据流水线阶段决定广播哪些数据
    if mpu.is_pipeline_first_stage():
        _broadcast(batch['tokens'])
        _broadcast(batch['position_ids'])
    elif mpu.is_pipeline_last_stage():
        _broadcast(batch['labels'])
        _broadcast(batch['loss_mask'])
else:
    # 其他 TP rank 创建空张量并接收广播
    tokens = torch.empty(..., device='cuda')
    _broadcast(tokens)  # 接收数据
    # ...
```

### 6.3 损失归约

**`average_losses_across_data_parallel_group(losses)`**

**功能**：跨数据并行组平均损失

```python
averaged_losses = torch.cat([loss.clone().detach().view(1) for loss in losses])
torch.distributed.all_reduce(
    averaged_losses, 
    group=mpu.get_data_parallel_group()
)
averaged_losses = averaged_losses / mpu.get_data_parallel_group().size()
```

### 6.4 掩码和位置 ID 生成

**`get_ltor_masks_and_position_ids(...)`**

**功能**：为从左到右的语言模型生成注意力掩码和位置 ID

**生成内容**：

1. **注意力掩码**（下三角矩阵）
```python
attention_mask = torch.tril(
    torch.ones((batch_size, seq_length, seq_length), device=device)
)
```

2. **损失掩码**
```python
loss_mask = torch.ones(data.size(), dtype=torch.float, device=device)
if eod_mask_loss:
    loss_mask[data == eod_token] = 0.0
if pad_mask_loss:
    loss_mask[data == pad_token] = 0.0
```

3. **位置 ID**
```python
position_ids = torch.arange(seq_length, dtype=torch.long, device=device)
position_ids = position_ids.unsqueeze(0).expand_as(data)

# 如果 reset_position_ids，在每个 EOD token 后重置位置
if reset_position_ids:
    for b in range(micro_batch_size):
        eod_indices = (data[b] == eod_token).nonzero()
        for i in eod_indices:
            position_ids[b, (i+1):] -= (i+1 - prev_index)
```

### 6.5 打印函数

**`print_rank_0(message)`**：仅在 rank 0 打印
**`print_rank_last(message)`**：仅在最后一个 rank 打印
**`warn_rank_0(message)`**：仅在 rank 0 警告

### 6.6 数据集混合配置

**`get_blend_and_blend_per_split(args)`**

**功能**：从参数中提取数据集混合配置

**支持的配置方式**：

1. **全局混合**：`data_path` 或 `data_args_path`
2. **按分割混合**：`train_data_path`, `valid_data_path`, `test_data_path` 或 `per_split_data_args_path`

```python
# 全局混合示例
data_path: ["dataset1", "dataset2", "0.5", "dataset3"]
# 结果：blend = [("dataset1", None), ("dataset2", 0.5), ("dataset3", None)]

# 按分割混合示例
train_data_path: ["train1", "train2"]
valid_data_path: ["valid1"]
# 结果：blend_per_split = [
#   [("train1", None), ("train2", None)],
#   [("valid1", None)],
#   []
# ]
```

---

## 7. global_vars.py - 全局变量管理

### 7.1 全局变量列表

```python
_GLOBAL_ARGS = None              # 训练参数
_GLOBAL_TOKENIZER = None         # Tokenizer
_GLOBAL_TENSORBOARD_WRITER = None # TensorBoard 写入器
_GLOBAL_WANDB_WRITER = None      # WandB 写入器
_GLOBAL_ONE_LOGGER = None        # OneLogger（NVIDIA 内部日志系统）
_GLOBAL_ADLR_AUTORESUME = None   # ADLR 自动恢复
_GLOBAL_TIMERS = None            # 性能计时器
_GLOBAL_ENERGY_MONITOR = None    # 能耗监控器
_GLOBAL_SIGNAL_HANDLER = None    # 信号处理器
```

### 7.2 初始化：`set_global_variables()`

**功能**：初始化所有全局变量

**初始化顺序**：

1. **设置 args**
```python
set_args(args)
```

2. **初始化微批次计算器**
```python
init_num_microbatches_calculator(
    args.rank,
    args.rampup_batch_size,
    args.global_batch_size,
    args.micro_batch_size,
    args.data_parallel_size,
)
```

3. **构建 Tokenizer**
```python
if build_tokenizer:
    _GLOBAL_TOKENIZER = build_tokenizer(args)
```

4. **初始化日志系统**
```python
_set_tensorboard_writer(args)   # TensorBoard
_set_wandb_writer(args)         # WandB
_set_one_logger(args)           # OneLogger
```

5. **初始化其他组件**
```python
_set_adlr_autoresume(args)      # ADLR autoresume
_set_timers(args)                # 计时器
_set_energy_monitor(args)        # 能耗监控
_set_signal_handler(args)        # 信号处理（如果启用）
```

### 7.3 TensorBoard 初始化

**`_set_tensorboard_writer(args)`**

**关键逻辑**：
```python
# 仅在最后一个 rank 初始化（避免多进程写入冲突）
if args.rank == (args.world_size - 1):
    from torch.utils.tensorboard import SummaryWriter
    _GLOBAL_TENSORBOARD_WRITER = SummaryWriter(
        log_dir=args.tensorboard_dir,
        max_queue=args.tensorboard_queue_size,
    )
```

### 7.4 WandB 初始化

**`_set_wandb_writer(args)`**

**关键逻辑**：
```python
if args.wandb_project and args.rank == (args.world_size - 1):
    import wandb
    wandb.init(
        dir=save_dir,
        name=args.wandb_exp_name,
        project=args.wandb_project,
        config=vars(args),  # 记录所有参数
    )
    _GLOBAL_WANDB_WRITER = wandb
```

### 7.5 ADLR Autoresume

**功能**：自动检测和恢复训练（NVIDIA 内部系统）

```python
if args.adlr_autoresume:
    from userlib.auto_resume import AutoResume
    _GLOBAL_ADLR_AUTORESUME = AutoResume
```

### 7.6 清理：`unset_global_variables()`

**功能**：清理所有全局变量（用于多轮训练）

```python
def unset_global_variables():
    global _GLOBAL_ARGS, _GLOBAL_TOKENIZER, ...
    _GLOBAL_ARGS = None
    _GLOBAL_TOKENIZER = None
    # ... 清理所有全局变量
    unset_num_microbatches_calculator()
```

---

## 8. async_utils.py - 异步工具

### 8.1 异步检查点队列

**`AsyncCallsQueue`**：管理异步检查点保存请求的队列

**核心功能**：
- 调度异步保存请求
- 跟踪未完成的请求
- 在需要时等待请求完成

### 8.2 关键函数

#### `schedule_async_save(async_request)`

**功能**：调度异步保存请求

```python
_async_calls_queue.schedule_async_request(async_request)
```

#### `maybe_finalize_async_save(blocking, terminate)`

**功能**：完成异步保存请求

**参数**：
- `blocking=True`：阻塞等待所有请求完成
- `blocking=False`：仅完成已完成的请求
- `terminate=True`：完成后关闭异步队列

```python
if blocking and not is_empty_async_queue():
    print_rank_0('Unfinalized async checkpoint saves. Finalizing them synchronously now.')

_async_calls_queue.maybe_finalize_async_calls(blocking, no_dist=False)

if terminate:
    _async_calls_queue.close()
```

#### `is_empty_async_queue()`

**功能**：检查异步队列是否为空

```python
return _async_calls_queue.get_num_unfinalized_calls() == 0
```

### 8.3 持久化异步工作器

**`init_persistent_async_worker()`**

**功能**：初始化持久化异步工作器（用于多进程环境）

```python
_async_calls_queue = AsyncCallsQueue(persistent=True)
```

**使用场景**：
- 多进程数据加载器
- 需要跨进程共享异步队列

---

## 9. dist_signal_handler.py - 信号处理

### 9.1 功能概述

`DistributedSignalHandler` 类用于在分布式训练中**统一处理信号**（如 SIGTERM），确保所有进程都能收到信号并优雅退出。

### 9.2 处理的信号

**默认信号**：`signal.SIGTERM`

**可配置信号**：通过 `args.exit_signal` 指定

### 9.3 实现原理

#### 信号接收
```python
class DistributedSignalHandler:
    def __init__(self, sig: signal.Signals = signal.SIGTERM):
        self.sig = sig
        self._signal_received = False
    
    def __enter__(self):
        # 注册信号处理器
        def handler(signum, frame):
            self._signal_received = True
        
        signal.signal(self.sig, handler)
        return self
```

#### 跨进程同步
```python
def signals_received(self):
    # 收集所有进程的信号接收状态
    all_received = all_gather_item(
        self._signal_received, dtype=torch.int32
    )
    return all_received
```

**`all_gather_item()`**：将标量值收集到所有进程

```python
def all_gather_item(item, dtype, group=None):
    tensor = torch.tensor([item], device=device, dtype=dtype)
    output_tensors = [
        torch.zeros(1, dtype=tensor.dtype, device=tensor.device)
        for _ in range(group_size)
    ]
    torch.distributed.all_gather(output_tensors, tensor, group)
    return [elem.item() for elem in output_tensors]
```

### 9.4 使用方式

#### 在训练循环中检查信号
```python
signal_handler = get_signal_handler()
if signal_handler.signals_received():
    # 保存检查点并退出
    save_checkpoint(...)
    sys.exit(0)
```

#### 在数据加载器工作进程中
```python
def worker_init_fn(_):
    DistributedSignalHandler(args.exit_signal).__enter__()
```

### 9.5 为什么需要分布式信号处理？

**问题**：在分布式训练中，信号可能只被主进程接收，其他进程无法感知。

**解决方案**：
1. 每个进程注册信号处理器
2. 当信号到达时，标记 `_signal_received = True`
3. 通过 `all_gather` 同步所有进程的信号状态
4. 所有进程统一退出

---

## 10. theoretical_memory_usage.py - 内存计算

### 10.1 功能概述

计算训练时的**理论内存占用**，包括：
- 权重和优化器状态内存
- 激活内存
- 总内存

### 10.2 权重和优化器内存：`compute_weight_and_optimizer_memory()`

#### 参数计算

**1. Transformer 层参数（密集层）**
```python
num_parameters_in_transformer_layer_dense = (
    2 * hidden_size * (
        # MLP（考虑 SwiGLU）
        ffn_hidden_size * gated_linear_multiplier +
        # LayerNorm（2 个）
        2
    ) +
    # 自注意力参数
    self_attn_term
)
```

**2. Transformer 层参数（MoE 层）**
```python
num_parameters_in_transformer_layer_moe = (
    2 * hidden_size * (
        # MoE MLP（多个专家）
        moe_ffn_hidden_size * num_experts * gated_linear_multiplier +
        # 共享专家 MLP
        shared_expert_ffn_hidden_size * gated_linear_multiplier +
        # LayerNorm
        2
    ) +
    self_attn_term
)
```

**3. 嵌入层参数**
```python
embedding_size = hidden_size * padded_vocab_size
if untie_embeddings_and_output_weights:
    num_parameters_in_embedding_layers = 2 * embedding_size
else:
    num_parameters_in_embedding_layers = embedding_size
```

#### 自注意力参数计算

**标准注意力（MHA/GQA）**：
```python
self_attn_term = (
    2 * hidden_size * hidden_size * (
        # Q 投影
        1 +
        # K/V 投影（GQA 时减少）
        (num_query_groups / num_attention_heads)
    ) * query_projection_to_hidden_size_ratio
)
```

**多潜在注意力（MLA）**：
```python
# Q LoRA + RoPE + Q Norm
q_term = q_lora_rank * (
    hidden_size + 
    num_attention_heads * (qk_head_dim + qk_pos_emb_head_dim) + 
    1
)

# KV LoRA + RoPE + KV Norm
kv_term = kv_lora_rank * (
    hidden_size + 
    num_attention_heads * (qk_head_dim + v_head_dim) + 
    1
)

# O 投影
o_term = (num_attention_heads * v_head_dim) * hidden_size

self_attn_term = q_term + kv_term + o_term + ...
```

#### 参数分片计算

**最重载的模型分片**（第一个流水线阶段）：
```python
num_parameters_on_most_loaded_model_shard = (
    (num_parameters_in_transformer_block / pipeline_model_parallel_size) +
    num_parameters_in_mtp_block +
    embedding_size
) / tensor_model_parallel_size
```

**其他模型分片**（其他流水线阶段）：
```python
num_parameters_on_other_model_shards = (
    num_parameters_in_transformer_block / 
    (pipeline_model_parallel_size * tensor_model_parallel_size)
)
```

#### 优化器状态内存

**每个参数的内存占用**：
```python
if not use_distributed_optimizer:
    # 标准优化器：参数(2) + 梯度(2) + 优化器状态(14) = 18 字节/参数
    num_bytes_per_parameter = 18
else:
    # 分布式优化器：参数(2) + 梯度(2) + 优化器状态分片(12/DP) = 6 + 12/DP 字节/参数
    num_bytes_per_parameter = 6 + (12 / data_parallel_size)
```

**优化器状态组成**（Adam）：
- 参数：2 字节（FP16/BF16）
- 梯度：2 字节（FP16/BF16）
- 优化器状态：14 字节（FP32）
  - `exp_avg`（动量）：4 字节
  - `exp_avg_sq`（二阶动量）：4 字节
  - 其他状态：6 字节

**总权重和优化器内存**：
```python
weight_and_optimizer_memory = (
    num_parameters_on_most_loaded_model_shard * num_bytes_per_parameter
)
```

### 10.3 激活内存：`compute_activation_memory()`

#### 基础激活内存（使用序列并行）

**每个 Transformer 层的激活内存**：
```python
activation_memory_per_layer = (
    seq_length * micro_batch_size * hidden_size
) * (
    18 + (4 * ffn_hidden_size / hidden_size)
)
```

**系数说明**：
- `18`：注意力激活（Q、K、V、注意力分数、输出等）
- `4 * ffn_hidden_size / hidden_size`：MLP 激活（考虑 SwiGLU 的 3/2 倍）

**总激活内存**：
```python
activation_memory = activation_memory_per_layer * num_layers
```

#### 嵌入层激活

```python
# 输入到嵌入层（pp_size 个微批次在流水线中）
activation_memory += (
    8 * seq_length * micro_batch_size * pipeline_model_parallel_size
)

# 嵌入层 dropout（pp_size 个微批次在流水线中）
activation_memory += (
    seq_length * micro_batch_size * hidden_size * pipeline_model_parallel_size
)
```

#### 流水线并行内存因子

**交错调度（Interleaved 1F1B）**：
```python
if virtual_pipeline_model_parallel_size is not None:
    interleaved_schedule_memory_penalty = 1 + (
        (pipeline_model_parallel_size - 1) /
        (pipeline_model_parallel_size * virtual_pipeline_model_parallel_size)
    )
    activation_memory *= interleaved_schedule_memory_penalty
```

**非交错调度**：
```python
if virtual_pipeline_model_parallel_size is None:
    # 考虑实际在流水线中的微批次数量
    activation_memory *= min(1, num_microbatches / pipeline_model_parallel_size)
```

#### 输出层激活（仅当 PP=1 时）

```python
if pipeline_model_parallel_size == 1:
    # 最终 LayerNorm 输出
    activation_memory += (
        seq_length * micro_batch_size * hidden_size * 4
    )
    # Logits（词汇表大小）
    activation_memory += (
        seq_length * micro_batch_size * padded_vocab_size * 4
    )
```

#### 激活内存分片

```python
# 激活内存按张量并行和序列并行分片
activation_memory = activation_memory / tensor_model_parallel_size
```

### 10.4 激活内存（不使用序列并行）：`compute_activation_memory_without_sp()`

**每个层的精确激活内存**：
```python
per_layer_memory = (
    seq_length * micro_batch_size * hidden_size * 
    (10 + (24 / tensor_model_parallel_size))
)
```

**系数说明**：
- `10`：基础激活（注意力输出、MLP 输入等）
- `24 / tensor_model_parallel_size`：需要跨 TP 通信的激活

**总激活内存**：
```python
total_activation_memory = per_layer_memory * num_layers
# 加上嵌入层、流水线因子、输出层等（同上述）
# 最后添加 5% 开销
total_activation_memory *= 1.05
```

### 10.5 内存报告：`report_theoretical_memory()`

**功能**：计算并打印理论内存占用

**计算流程**：
```python
def report_theoretical_memory(args, num_microbatches=None, verbose=False):
    # 1. 计算权重和优化器内存
    weight_and_optimizer_memory = (
        compute_weight_and_optimizer_memory(args, verbose=verbose) / 
        NUM_BYTES_IN_MEGABYTE
    )
    
    # 2. 选择激活内存计算方法
    if args.sequence_parallel and args.recompute_granularity == 'selective':
        activation_memory = (
            compute_activation_memory(args, num_microbatches, verbose) /
            NUM_BYTES_IN_MEGABYTE
        )
    else:
        activation_memory = (
            compute_activation_memory_without_sp(args, num_microbatches, verbose) /
            NUM_BYTES_IN_MEGABYTE
        )
    
    # 3. 计算总内存
    total_memory = weight_and_optimizer_memory + activation_memory
    
    # 4. 打印报告
    print(
        f"Theoretical memory footprints: "
        f"weight and optimizer={weight_and_optimizer_memory:.2f} MB, "
        f"activation={activation_memory:.2f} MB, "
        f"total={total_memory:.2f} MB"
    )
    
    return weight_and_optimizer_memory, activation_memory, total_memory
```

### 10.6 内存计算的关键假设

1. **激活重计算**：如果启用，激活内存会减少（但会增加计算开销）
2. **序列并行**：激活按序列维度分片，减少内存占用
3. **流水线并行**：激活内存与流水线中的微批次数量相关
4. **分布式优化器**：优化器状态分片，减少内存占用
5. **混合精度**：参数和梯度使用 FP16/BF16，但优化器状态使用 FP32

### 10.7 内存优化建议

1. **启用序列并行**：当 TP > 1 时，可以显著减少激活内存
2. **使用分布式优化器**：减少优化器状态内存（特别是大模型）
3. **启用激活重计算**：牺牲计算换取内存
4. **调整流水线并行大小**：平衡内存和通信开销
5. **使用梯度累积**：减少微批次大小，降低激活内存

---

## 11. data_samplers.py - 数据采样器

### 11.1 功能概述

`data_samplers.py` 提供了用于 Megatron 预训练的数据采样器，支持：
- **顺序采样**：`MegatronPretrainingSampler`
- **随机采样**：`MegatronPretrainingRandomSampler`
- **随机种子数据集包装器**：`RandomSeedDataset`

### 11.2 数据加载器构建：`build_pretraining_data_loader()`

**功能**：根据配置构建预训练数据加载器

**支持的加载器类型**：

1. **`single`**：使用 `MegatronPretrainingSampler`（顺序采样）
2. **`cyclic`**：使用 `MegatronPretrainingRandomSampler`（随机采样）
3. **`external`**：外部数据加载器（用户提供）

**关键逻辑**：
```python
def build_pretraining_data_loader(dataset, consumed_samples):
    args = get_args()
    
    # 验证集：使用完整数据集
    if split == Split.valid and args.full_validation:
        batch_sampler = MegatronPretrainingSampler(
            total_samples=len(dataset),
            consumed_samples=0,  # 从开始采样
            micro_batch_size=args.micro_batch_size,
            data_parallel_rank=mpu.get_data_parallel_rank(),
            data_parallel_size=mpu.get_data_parallel_world_size(),
        )
    
    # 训练集：根据 dataloader_type 选择采样器
    elif args.dataloader_type == 'single':
        batch_sampler = MegatronPretrainingSampler(...)
    elif args.dataloader_type == 'cyclic':
        batch_sampler = MegatronPretrainingRandomSampler(...)
    elif args.dataloader_type == 'external':
        return dataset  # 直接返回外部数据加载器
    
    # 构建 PyTorch DataLoader
    return torch.utils.data.DataLoader(
        dataset,
        batch_sampler=batch_sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=True if args.num_workers > 0 else False,
        worker_init_fn=maybe_worker_init_fn,  # 信号处理
    )
```

### 11.3 MegatronPretrainingSampler - 顺序采样器

#### 设计目的

- **数据并行分片**：每个数据并行 rank 处理不同的数据子集
- **支持断点续训**：通过 `consumed_samples` 跳过已处理的数据
- **批次对齐**：确保所有 DP rank 的批次大小一致

#### 初始化参数

```python
def __init__(
    self,
    total_samples: int,           # 数据集总样本数
    consumed_samples: int,         # 已消费的样本数（用于断点续训）
    micro_batch_size: int,         # 微批次大小
    data_parallel_rank: int,       # 当前 DP rank
    data_parallel_size: int,       # DP 总大小
    drop_last: bool = True,        # 是否丢弃最后一个不完整批次
):
```

#### 关键属性

```python
self.micro_batch_times_data_parallel_size = (
    micro_batch_size * data_parallel_size
)
# 全局批次大小 = 微批次大小 × DP 大小
```

#### 采样逻辑

```python
def __iter__(self):
    batch = []
    # 从 consumed_samples 开始采样
    for idx in range(self.consumed_samples, self.total_samples):
        batch.append(idx)
        
        # 当收集到全局批次大小的样本时
        if len(batch) == self.micro_batch_times_data_parallel_size:
            # 计算当前 DP rank 的样本范围
            start_idx, end_idx = self.get_start_end_idx()
            # 返回当前 rank 的样本
            yield batch[start_idx:end_idx]
            batch = []
    
    # 处理最后一个不完整批次
    if len(batch) > 0 and not self.drop_last:
        start_idx, end_idx = self.get_start_end_idx()
        yield batch[start_idx:end_idx]
```

#### 批次分片计算

```python
def get_start_end_idx(self):
    """计算当前 DP rank 在批次中的样本范围"""
    start_idx = self.data_parallel_rank * self.micro_batch_size
    end_idx = start_idx + self.micro_batch_size
    return start_idx, end_idx
```

**示例**：
- DP size = 4, micro_batch_size = 2
- 全局批次 = [0, 1, 2, 3, 4, 5, 6, 7]
- DP rank 0: [0, 1]
- DP rank 1: [2, 3]
- DP rank 2: [4, 5]
- DP rank 3: [6, 7]

### 11.4 MegatronPretrainingRandomSampler - 随机采样器

#### 设计目的

- **随机采样**：每个 epoch 随机打乱数据
- **数据分片**：支持两种分片模式
  - `data_sharding=True`：将数据分成桶（bucket），每个 DP rank 处理一个桶
  - `data_sharding=False`：全局随机打乱，然后按 DP rank 交错采样

#### 初始化参数

```python
def __init__(
    self,
    dataset,                      # 数据集对象
    total_samples: int,
    consumed_samples: int,
    micro_batch_size: int,
    data_parallel_rank: int,
    data_parallel_size: int,
    data_sharding: bool,          # 是否使用数据分片
):
```

#### 随机采样逻辑（数据分片模式）

```python
def __iter__(self):
    # 计算当前 epoch
    active_total_samples = self.total_samples - self.last_batch_size
    self.epoch = self.consumed_samples // active_total_samples
    current_epoch_samples = self.consumed_samples % active_total_samples
    
    if self.data_sharding:
        # 计算桶大小
        bucket_size = (
            self.total_samples // self.micro_batch_times_data_parallel_size
        ) * self.micro_batch_size
        
        # 计算当前 DP rank 的桶起始位置
        bucket_offset = current_epoch_samples // self.data_parallel_size
        start_idx = self.data_parallel_rank * bucket_size
        
        # 在桶内随机打乱
        g = torch.Generator()
        g.manual_seed(self.epoch)  # 每个 epoch 使用不同的随机种子
        random_idx = torch.randperm(bucket_size, generator=g).tolist()
        
        # 选择当前 DP rank 的样本
        idx_range = [
            start_idx + x 
            for x in random_idx[bucket_offset:]
        ]
    else:
        # 全局随机打乱
        full_bucket_size = (
            self.total_samples // self.micro_batch_size
        ) * self.micro_batch_size
        full_bucket_offset = current_epoch_samples
        
        g = torch.Generator()
        g.manual_seed(self.epoch)
        idx_range_total = torch.randperm(full_bucket_size, generator=g).tolist()
        idx_range_active = idx_range_total[full_bucket_offset:]
        
        # 按 DP rank 交错采样
        idx_range = idx_range_active[
            self.data_parallel_rank :: self.data_parallel_size
        ]
    
    # 生成批次
    batch = []
    for idx in idx_range:
        batch.append(idx)
        if len(batch) == self.micro_batch_size:
            self.consumed_samples += self.micro_batch_times_data_parallel_size
            yield batch
            batch = []
```

#### 数据分片模式对比

**`data_sharding=True`（桶模式）**：
- 优点：每个 DP rank 处理连续的数据块，缓存友好
- 缺点：数据分布可能不均匀（如果数据集本身不均匀）

**`data_sharding=False`（全局随机）**：
- 优点：数据分布更均匀
- 缺点：需要全局随机打乱，开销较大

### 11.5 RandomSeedDataset - 随机种子数据集包装器

#### 设计目的

- **确定性随机化**：为每个样本设置独立的随机种子
- **支持数据增强**：确保相同样本在不同 epoch 有不同的增强结果
- **可复现性**：通过种子控制随机行为

#### 实现原理

```python
class RandomSeedDataset(Dataset):
    def __init__(self, dataset, seed):
        self.base_seed = seed
        self.curr_seed = seed
        self.dataset = dataset
    
    def set_epoch(self, epoch):
        """设置当前 epoch，改变随机种子偏移"""
        self.curr_seed = self.base_seed + epoch
    
    def __getitem__(self, idx):
        """为每个样本设置独立的随机种子"""
        seed = idx + self.curr_seed
        torch.manual_seed(seed)
        random.seed(seed)
        np.random.seed(seed)
        return self.dataset[idx]
```

#### 使用场景

- **数据增强**：图像分类、目标检测等任务
- **随机掩码**：BERT 风格的掩码语言模型
- **随机采样**：需要为每个样本生成随机数的场景

### 11.6 采样器选择建议

1. **顺序采样（`single`）**：
   - 适用于：大规模预训练、需要确定性训练
   - 优点：简单、高效、支持断点续训
   - 缺点：数据顺序固定，可能影响训练效果

2. **随机采样（`cyclic`）**：
   - 适用于：需要数据随机化的场景
   - 优点：每个 epoch 数据顺序不同，可能提升训练效果
   - 缺点：需要额外的随机打乱开销

3. **数据分片模式**：
   - `data_sharding=True`：适合大规模数据集，减少通信开销
   - `data_sharding=False`：适合小规模数据集，数据分布更均匀

### 11.7 与数据并行的配合

**关键点**：采样器确保每个数据并行 rank 处理不同的数据子集，避免数据重复。

**批次构建流程**：
```
1. 采样器生成全局批次索引 [0, 1, 2, ..., global_batch_size-1]
2. 每个 DP rank 获取自己的子集：
   - DP rank 0: [0, 1, ..., micro_batch_size-1]
   - DP rank 1: [micro_batch_size, ..., 2*micro_batch_size-1]
   - ...
3. 每个 rank 从数据集中加载对应的样本
4. 前向传播时，每个 rank 处理不同的数据
5. 梯度同步时，所有 rank 的梯度被平均
```

---

## 总结

本文档详细介绍了 `megatron/training/` 目录下的核心模块：

1. **YAML 配置**：支持使用 YAML 文件替代命令行参数，提供更灵活的配置方式
2. **参数工具**：自动从 dataclass 生成 argparse 参数，减少重复代码
3. **初始化**：完整的分布式训练初始化流程，包括参数解析、全局变量设置、分布式初始化等
4. **训练循环**：核心训练逻辑，包括前向传播、后向传播、优化器更新、检查点保存等
5. **检查点管理**：支持多种检查点格式，包括传统格式、分布式检查点、异步保存等
6. **工具函数**：参数范数计算、批次数据获取、损失归约等实用函数
7. **全局变量管理**：集中管理训练过程中的全局状态
8. **异步工具**：支持异步检查点保存，提升训练效率
9. **信号处理**：分布式环境下的信号处理，确保优雅退出
10. **内存计算**：理论内存占用计算，帮助优化训练配置
11. **数据采样器**：支持顺序和随机采样，适配不同的训练需求

这些模块共同构成了 Megatron-LM 完整的训练框架，为大规模语言模型训练提供了坚实的基础。