# MPX Terrain Module

程序化地形生成模块，基于 Wave Function Collapse (WFC) 算法生成复杂的楼梯、平台和粗糙地面。

## 概述

本模块从 `phase_guided_terrain_traversal` 项目迁移而来，适配 MPX 项目中的 R2-1024 双足机器人。支持生成以下地形类型：

- **平坦平台** (Tile 0, 1)
- **直线楼梯** (Tile 2-5) - 4个方向
- **上坡转弯楼梯** (Tile 6-9) - 4个方向
- **下坡转弯楼梯** (Tile 10-13) - 4个方向
- **粗糙地面** - 随机放置的方块

## 目录结构

```
mpx/terrain/
├── __init__.py           # 模块初始化
├── generator.py          # 地形生成器核心
├── get_indexes.py        # WFC 瓦片索引工具
├── heightmap.py          # 基于 mjx.ray 的高程图提取
├── test_terrain_gen.py   # 测试脚本
├── wfc/                  # WFC 算法实现
│   ├── __init__.py
│   └── wfc.py            # WFC 求解器
└── README.md             # 本文档
```

## 安装依赖

```bash
pip install alive-progress noise opencv-python
```

## 快速开始

### 1. 生成随机测试地形

```python
from mpx.terrain import random_test_env

# 生成 9x9 的测试地形，保存到 scene_terrain_test.xml
random_test_env(
    num_objects=150,    # 最大地形对象数
    size=9,             # WFC 地图大小 (9x9)
    step_height=0.08,   # 每级台阶高度 (米)
    width=0.35,         # 每级台阶宽度 (米)
    num_steps=3         # 台阶级数
)
```

### 2. 从 WFC 地图生成地形

```python
from mpx.terrain import generate_14, TerrainGenerator, addElement, create_centered_grid

# 生成 9x9 的 WFC 地图
wave = generate_14(size=9, test=False)

# 创建地形生成器
tg = TerrainGenerator(
    width=0.35,
    step_height=0.08,
    num_stairs=3,
    render=True  # 渲染到 XML
)

# 创建网格
grid = create_centered_grid(9, tg.length)

# 根据 WFC 地图添加地形元素
for i in range(grid.shape[0]):
    for j in range(grid.shape[1]):
        addElement(tg, wave[i, j], grid[i, j])

# 保存场景
tg.Save("my_terrain.xml")
```

### 3. 使用高程图感知地形

```python
from mujoco import mjx
import jax.numpy as jnp
from mpx.terrain import create_sensor_matrix

# 加载场景
model = mujoco.MjModel.from_xml_path("scene_terrain_test.xml")
mx = mjx.put_model(model)
dx = mjx.put_data(model, mujoco.MjData(model))

# 生成局部高程图
heightmap = create_sensor_matrix(
    mx, dx,
    center=[0, 0, 0.5],  # 机器人位置
    yaw=0.0,             # 机器人朝向
    dist_x=0.1,          # X 方向扫描间隔
    dist_y=0.1,          # Y 方向扫描间隔
    num_heightscans=11,  # 高度方向扫描数
    num_widthscans=11    # 宽度方向扫描数
)
```

## 测试

### 运行基本测试

```bash
cd /home/wzn/双足/mpx
conda activate mpx_env
python -m mpx.terrain.test_terrain_gen
```

### 可视化测试

```bash
python -m mpx.terrain.test_terrain_gen --visualize
```

### 自定义参数测试

```bash
python -m mpx.terrain.test_terrain_gen \
    --size 9 \
    --num-objects 150 \
    --step-height 0.12 \
    --width 0.4 \
    --num-steps 4
```

## 地形类型说明

### Tile 索引对应关系

| 索引 | 地形类型 | 说明 |
|------|---------|------|
| 0 | 空地/地面 | 不添加任何地形 |
| 1 | 高台 | 平坦的高台 |
| 2-5 | 直线楼梯 | 上楼梯 (0°, 90°, 180°, 270°) |
| 6-9 | 上坡转弯楼梯 | 转弯上楼梯 (0°, 90°, 180°, 270°) |
| 10-13 | 下坡转弯楼梯 | 转弯下楼梯 (0°, 90°, 180°, 270°) |

### WFC 连接规则

地形瓦片之间遵循特定的连接规则，确保：
- 楼梯底部连接低平台或地面
- 楼梯顶部连接高平台或继续延伸
- 转弯楼梯的方向一致性

## 与 MPC 集成

### 1. 修改配置文件

在 `config/config_r2.py` 中添加地形相关配置：

```python
# 地形配置
use_terrain = True
terrain_scene_path = "mpx/data/r2-1024/mjcf/scene_terrain_test.xml"

# 地形感知
use_terrain_estimation = True
heightmap_size = (11, 11)
heightmap_resolution = 0.1
```

### 2. 修改代价函数

在 `utils/objectives.py` 中添加地形约束：

```python
def r2_wb_obj_with_terrain(...):
    # 原有的姿态、速度等代价
    cost = r2_wb_obj(...)

    # 添加地形约束
    if use_terrain_estimation:
        # 从高程图获取足端期望高度
        foot_heights = get_foot_heights_from_heightmap(heightmap, foot_positions)

        # 添加足端高度跟踪代价
        cost += W_foot_height * (foot_z - foot_heights)**2

    return cost
```

## 性能优化

### 批量环境

使用 `BatchedMPCControllerWrapper` 可以同时处理多个地形环境：

```python
from mpx.terrain import create_random_matrix

# 为 100 个环境生成地形数据
terrain_data = create_random_matrix(
    num_envs=100,
    num_bodies=150,
    size=9,
    height_min=0.05,
    height_max=0.15
)

# terrain_data.shape = (100, 150, 10)
# 每个环境 150 个地形对象，每个对象 10 个参数 (pos[3], quat[4], size[3])
```

## 文件说明

### 核心文件

- **generator.py**: 地形生成器，包含 `TerrainGenerator` 类和各种地形元素生成函数
- **wfc/wfc.py**: Wave Function Collapse 算法实现
- **get_indexes.py**: WFC 瓦片索引辅助函数
- **heightmap.py**: 基于 `mjx.ray` 的局部高程图提取

### 场景文件

- **scene_terrain_boxes.xml**: 带占位符的地形场景模板
- **scene_terrain_test.xml**: 测试生成的地形场景（运行测试后生成）

## 已知问题

1. **小地图全为边界**: 当 `size=5` 且 `test=True` 时，WFC 可能生成全 0 的地图。建议使用 `size >= 7` 或 `test=False`。

2. **NumPy 版本冲突**: `opencv-python` 4.13+ 需要 numpy >= 2.0，但 JAX 可能需要特定版本。建议使用 `opencv-python==4.9.0.80`。

## 未来扩展

- [ ] 添加更多地形类型（斜坡、台阶组合等）
- [ ] 支持从高度图生成地形
- [ ] 集成到端到端训练流程
- [ ] 添加地形难度自适应调整

## 参考

- 原始项目: `phase_guided_terrain_traversal`
- WFC 算法: [Wave Function Collapse](https://github.com/mxgmn/WaveFunctionCollapse)
- MuJoCo 文档: [mjx.ray](https://mujoco.readthedocs.io/en/stable/api.html#mjx.ray)
