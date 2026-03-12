# MPX 地形生成系统文档

## 概述

MPX 地形生成系统用于创建各种崎岖地形的 MuJoCo 高度场（height field），支持程序化生成和可视化预览。

---

## 核心原理

### 高度场（Height Field）

MuJoCo 使用 **高度场** 来表示复杂地形：
- 将地形离散化为 **N × M** 的网格
- 每个网格点存储一个高度值
- 物理引擎在网格间进行线性插值，形成连续表面

### Perlin 噪声地形生成

我们使用多层正弦波叠加来模拟 Perlin 噪声效果：

```python
# 多个频率分量的波形叠加
height = Σ[amplitude[i] × sin(frequency[i] × X + phase) × cos(frequency[i] × Y + phase)]
```

**关键参数：**
- `frequencies`: 频率分量，控制地形的"粗糙度"
- `amplitudes`: 振幅分量，控制每个频率层的影响程度
- 高频 = 细节纹理，低频 = 大地形起伏

### 地形特征增强

除了基础噪声，我们还添加：

1. **高斯山峰**：随机位置的凸起
```python
gaussian = peak_height × exp(-((X-cx)² + (Y-cy)²) / (2×radius²))
```

2. **高斯深坑**：随机位置的凹陷
```python
gaussian = -pit_depth × exp(-((X-cx)² + (Y-cy)²) / (2×radius²))
```

---

## 代码实现

### 地形生成器

```python
def generate_perlin_terrain(size=(512, 512), scale=0.3, height_scale=0.8, seed=42):
    """
    生成类 Perlin 噪声地形

    Args:
        size: 地形分辨率 (nrow, ncol)
        scale: 噪声缩放因子，控制地形密度
        height_scale: 最大高度（米）
        seed: 随机种子

    Returns:
        height: 高度数组 (nrow, ncol)
    """
```

**参数说明：**

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `size` | (512, 512) | 分辨率，越高越平滑 |
| `scale` | 0.3 | 噪声频率缩放，越大越密集 |
| `height_scale` | 0.8 | 最大高度（米） |
| `seed` | 42 | 随机种子，可复现地形 |

### MuJoCo 高度场配置

```xml
<hfield name="rough_terrain"
        nrow="512"
        ncol="512"
        size="x_size y_size z_max bottom_offset"/>

<geom name="terrain"
      type="hfield"
      hfield="rough_terrain"
      material="ground_terrain"/>
```

**MuJoCo 3.4.0 格式：** `size="x_size y_size z_max bottom_offset"` （4个值）

| 值 | 说明 | 示例 |
|----|------|------|
| x_size | X方向物理尺寸（米） | 6 |
| y_size | Y方向物理尺寸（米） | 6 |
| z_max | 最大高度（米） | 1.0 |
| bottom_offset | 底部偏移（米） | 0.05 |

---

## 使用方法

### 方法一：动态预览（推荐）

运行动态地形预览工具：

```bash
python mpx/utils/preview_terrain_dynamic.py
```

**特性：**
- ✅ 实时生成地形
- ✅ 球体物理仿真演示
- ✅ 可调参数（分辨率、高度等）

### 方法二：生成 PNG 高度图

使用地形生成器：

```bash
# Perlin 噪声地形
python mpx/utils/terrain_generator.py --type perlin --output terrain.png --height 0.8

# 阶梯地形
python mpx/utils/terrain_generator.py --type steps --output steps.png --height 0.3

# 碎石地形
python mpx/utils/terrain_generator.py --type rubble --output rubble.png --height 0.15
```

### 方法三：在代码中直接使用

```python
import mujoco
import numpy as np
from mpx.utils.preview_terrain_dynamic import generate_perlin_terrain

# 1. 创建模型
model = mujoco.MjModel.from_xml_path("scene_terrain.xml")

# 2. 生成地形数据
height_data = generate_perlin_terrain(
    size=(512, 512),
    scale=0.4,
    height_scale=0.8,
    seed=42
)

# 3. 填充到 MuJoCo 高度场
hfield_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_HFIELD, "rough_terrain")
adr = model.hfield_adr[hfield_id]
model.hfield_data[adr:adr + 512*512] = height_data.flatten()
```

---

## 配置参数调优

### 地形分辨率

| 分辨率 | 网格大小 | 效果 | 性能 |
|--------|----------|------|------|
| 64×64 | ~9.4cm | 方块感明显 | 最快 |
| 128×128 | ~4.7cm | 较平滑 | 快 |
| 256×256 | ~2.3cm | 平滑 | 中等 |
| **512×512** | **~1.2cm** | **非常平滑** | **推荐** |
| 1024×1024 | ~0.6cm | 极致细节 | 慢 |

**推荐：512×512** （平衡质量和性能）

### 地形高度

| 高度 | 适用场景 |
|------|----------|
| 0.1-0.2m | 轻微起伏 |
| 0.3-0.5m | 中等崎岖 |
| **0.6-1.0m** | **高度崎岖** |
| 1.0m+ | 极端地形 |

### 频率分量

```python
# 当前配置（超崎岖）
frequencies = [1, 2, 4, 8, 16, 32]
amplitudes  = [1.0, 0.8, 0.5, 0.3, 0.15, 0.08]

# 更平滑（减少高频）
frequencies = [1, 2, 4, 8]
amplitudes  = [1.0, 0.7, 0.4, 0.2]

# 更粗糙（增加低频权重）
frequencies = [1, 2, 4]
amplitudes  = [1.5, 0.5, 0.2]
```

---

## 完整示例

### 场景文件示例

```xml
<!-- scene_rough_terrain.xml -->
<mujoco model="rough_terrain">
  <asset>
    <!-- 纯色地面材质 -->
    <material name="ground_terrain"
              rgba="0.5 0.45 0.35 1"
              reflectance="0.1"
              specular="0"
              shininess="0.1"/>

    <!-- 高度场配置 -->
    <hfield name="rough_terrain"
            nrow="512"
            ncol="512"
            size="6 6 1.0 0.05"/>
  </asset>

  <worldbody>
    <light pos="0 0 4" dir="0 0 -1" directional="true"/>
    <geom name="terrain"
          type="hfield"
          hfield="rough_terrain"
          material="ground_terrain"
          contype="2"
          conaffinity="1"/>
  </worldbody>
</mujoco>
```

### Python 生成代码

```python
import mujoco
import numpy as np

# 生成地形数据
def generate_terrain():
    np.random.seed(42)

    # 创建坐标网格
    x = np.linspace(0, 0.4 * 4 * np.pi, 512)
    y = np.linspace(0, 0.4 * 4 * np.pi, 512)
    X, Y = np.meshgrid(x, y)

    height = np.zeros_like(X)

    # 多层噪声叠加
    frequencies = [1, 2, 4, 8, 16, 32]
    amplitudes = [1.0, 0.8, 0.5, 0.3, 0.15, 0.08]

    for freq, amp in zip(frequencies, amplitudes):
        phase_x = np.random.rand() * 2 * np.pi
        phase_y = np.random.rand() * 2 * np.pi
        height += amp * np.sin(freq * X + phase_x) * np.cos(freq * Y + phase_y)
        height += amp * 0.5 * np.sin(freq * X * 1.3 + phase_x) * np.sin(freq * Y * 0.7 + phase_y)

    # 添加山峰
    for _ in range(15):
        cx = np.random.rand() * 512
        cy = np.random.rand() * 512
        radius = np.random.randint(3, 12)
        peak_height = np.random.uniform(0.5, 1.5) * 0.8
        gaussian = peak_height * np.exp(-((X - cx)**2 + (Y - cy)**2) / (2 * (radius/12)**2))
        height += gaussian

    # 添加深坑
    for _ in range(5):
        cx = np.random.rand() * 512
        cy = np.random.rand() * 512
        radius = np.random.randint(8, 20)
        pit_depth = np.random.uniform(0.3, 0.6) * 0.8
        gaussian = -pit_depth * np.exp(-((X - cx)**2 + (Y - cy)**2) / (2 * (radius/10)**2))
        height += gaussian

    # 归一化并缩放
    height = (height - height.min()) / (height.max() - height.min())
    height = height * 0.8

    return height

# 应用到 MuJoCo
height_data = generate_terrain()
model.hfield_data[adr:adr + 512*512] = height_data.flatten()
```

---

## 常见问题

### Q1: 地形看起来是方块状的？

**A:** 分辨率太低。增加到 256×256 或 512×512。

### Q2: MuJoCo 报错 "size does not have enough data"？

**A:** MuJoCo 3.4.0 需要的 hfield size 格式是 4 个值：
```xml
<hfield size="x_size y_size z_max bottom_offset"/>
```

### Q3: 如何让地形更平缓？

**A:** 减少 `height_scale` 或减少高频分量：
```python
height_scale = 0.3  # 更低的高度
frequencies = [1, 2, 4]  # 减少高频
```

### Q4: 如何创建特定地形（如楼梯、斜坡）？

**A:** 使用 `terrain_generator.py`：
```bash
python mpx/utils/terrain_generator.py --type steps --output stairs.png
```

---

## 文件结构

```
mpx/
├── utils/
│   ├── preview_terrain_dynamic.py    # 动态地形预览工具
│   └── terrain_generator.py          # PNG 地形生成器
├── data/r2-1024/mjcf/
│   ├── scene_perlin.xml             # Perlin 地形场景
│   └── terrain_*.png                # 预生成的地形图
└── docs/
    └── terrain_generation.md        # 本文档
```

---

## 与 MPC 集成

要在 MPX MPC 控制中使用崎岖地形：

1. **创建地形场景文件**
```bash
cp mpx/data/r2-1024/mjcf/scene.xml mpx/data/r2-1024/mjcf/scene_rough.xml
```

2. **修改场景，添加高度场**
```xml
<asset>
  <hfield name="rough_terrain" nrow="512" ncol="512" size="6 6 0.8 0.05"/>
</asset>
<worldbody>
  <geom name="terrain" type="hfield" hfield="rough_terrain" material="ground_terrain"/>
</worldbody>
```

3. **运行 MPC 示例**
```bash
python mpx/examples/mjx_r2.py  # 使用崎岖地形场景
```

---

## 参考资料

- [MuJoCo 官方文档 - Height Fields](https://mujoco.readthedocs.io/en/latest/XMLreference.html#hfield)
- [Perlin Noise Wikipedia](https://en.wikipedia.org/wiki/Perlin_noise)
- [MPX 项目 README](../README.md)

---

**最后更新：** 2026年2月
