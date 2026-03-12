# SDF Pretrain Module

视觉感知模块的 SDF 鍟预训练框架，## 模块概述

本模块提供端到端的 SDF (Signed Distance Field) 齦从高程图输入，使用 `mpx.terrain.heightmap` 中的 `create_sensor_matrix` 进行批量 ray casting，生成局部高程图。

## 核心特性
- **在线数据生成**: 无需预先存储数据，- **批量处理**: 支持批量生成多个样本
- **可微分**: 完全支持 JAX 自动微分

## 安装依赖

```bash
pip install jax jaxlib flax optax
```

## 快速开始

```python
from mpx.sdf_pretrain import SDFOnlineGenerator

# 创建生成器
xml_path = "mpx/data/r2-1024/mjcf/scene_terrain_test.xml"
generator = SDFOnlineGenerator(xml_path)

# 生成测试批次
key = jax.random.PRNGKey(42)
batch = generator.generate_batch(key, batch_size=4)
heightmap = batch['heightmap']  # (4, 21, 3)
queries_local = batch['queries_local']  # (4, 64, 3)
queries_global = batch['queries_global']  # (4, 64, 3)
sdf_gt = batch['sdf']  # (4,)

print(f"Heightmap shape: {heightmap.shape}")
print(f"Queries shape: {queries_local.shape}")
print(f"SDF shape: {sdf_gt.shape}")

# 统计
inside = (sdf_gt < 0).sum()
print(f"Inside: {inside}/{sdf_gt.size} ({100*inside/sdf_gt.size:.1f}%)")
```

