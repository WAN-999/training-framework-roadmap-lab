
# Module

### parameter、buffer、module
模型内主要管理和保存的就是这三个对象。

parameter: 本质是一个特殊的Tensor，requires_grad=True 默认开启
buffer: 不需要梯度、不被优化器更新，但需要跟着模型走的张量，不参与反向，不参与优化，会保存到state_dict()
module: 本质是nn.module，在调用module.parameter、model.train等函数时会递归调用

### hook机制

module支持以下hook，执行时机如下（每次都会执行所有的hook）：

  |hook | 触发时机 |
  |:---|:---|
  |_global_parameter_registration_hooks|注册 Parameter 时|
  |_global_module_registration_hooks|注册子 Module 时|
  |_forward_hooks|forward() 执行后| 
  |_forward_pre_hooks|forward() 执行前| 
  |_backward_hooks|反向传播时| 

#### 举个例子
```python
  import torch
  import torch.nn as nn

  # ① 定义 hook 函数
  def shard_hook(module, name, param):
      rank = torch.distributed.get_rank()   # 当前 GPU 编号（0~3）
      world_size = torch.distributed.get_world_size()  # 总 GPU 数 = 4

      # 按第 0 维切分，每个 GPU 只拿自己的那一段 
      chunk_size = param.shape[0] // world_size
      start = rank * chunk_size 
      end   = start + chunk_size

      sharded = param.data[start:end].contiguous()
      return nn.Parameter(sharded)   # 返回新 param，替换原来的

  # ② 注册全局 hook（在模型创建之前注册）
  handle = nn._global_parameter_registration_hooks["shard"] = shard_hook

  # ③ 正常创建模型，hook 自动介入
  model = nn.Linear(1024, 1024)
  # 此时 model.weight.shape = [256, 1024]（只有 1/4）

  # ④ 用完后可以移除
  del nn._global_parameter_registration_hooks["shard"]
```

### 参数的自动注册

通过重写``__setattr__``在初始化时自动注册

``` python
# nn.Parameter会被默认注册为参数
self.weight = nn.Parameter(torch.randn(10, 10))

# 不会被注册为参数
self.weight = torch.randn(10, 10)
```

```
parameter、buffer、module这三个类型会被自动注册

分为四个dict：
_parameters
_buffers
_modules
_non_persistent_buffers_set

其中buffer分为 persistent buffer 和 non persistent buffer
```

### forward

``` python
# 继承的module必须自己实现forwar，否则报异常
forward: Callable[..., Any] = _forward_unimplemented

```




