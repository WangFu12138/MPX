# 搬运 phase_guided_terrain_traversal 地形生成与高程图模块到 MPX 项目

## 背景

目标是将 `phase_guided_terrain_traversal` 的地形生成和 `mjx.ray` 高程图提取模块搬运到 MPX 项目中，使得 R2-1024 双足机器人能在程序化生成的楼梯/粗糙地形上进行基于 SDF 惩罚的 MPC 控制。

第一阶段（地形生成模块）已搬运完成，能够成功可视化 WFC 生成的箱体地形。
第二阶段（高程图集成）旨在将 `mjx.ray` 获取的局部地形高程图，作为动态参数传入 MPC 求解器，引入新的接触惩罚使得求解器能感知地形起伏。

---

## Phase 1: 地形生成模块搬运（已完成）

| 源路径 | 功能 |
|--------|------|
| `terrain/generator.py` | 核心地形生成器（TerrainGenerator 类 + WFC 地图生成） |
| `terrain/getIndexes.py` | WFC 瓦片索引辅助函数 |
| `wfc/wfc/wfc.py` | Wave Function Collapse 求解器 |
| `robots/heightmap.py` | 基于 mjx.ray 的局部高程图提取 |

创建了 `mpx/terrain/` 目录，并适配了相关的导入路径、`robot.xml` 路径和场景文件。已可以通过 `test_terrain_gen.py` 和 `visualize_terrain.py` 正确可视化地形。

---

## Phase 2: 高程图模块与 MPC 集成 (Proposed Changes)

将局部高程图信息喂给 MPC 并实现地形感知避障，需要修改以下核心文件：

### 1. `mpx/terrain/heightmap.py`

现有基于 `mjx.ray` 的扫图模块已经迁入，我们需要在其中封装一个易用的高层抽象函数：
- **`get_local_heightmap(mjx_model, mjx_data, center_pos, yaw)`**: 调用 `create_sensor_matrix` 获取机器人正下方和前方一定范围的网格离散点（例如 $21 \times 21$ 的 `heightmap`），用来表征局部的地表起伏。

### 2. `mpx/utils/mpc_wrapper.py`

`MPCControllerWrapper` 负责在每次 MPC run 之前提取环境状态并调用 `reference_generator`：
- 在 `run()` 中：利用 JIT 编译好的射线检测函数，传入当前的 `mjx_model`、`mjx_data`（在 `mj_step`/`mj_kinematics` 后更新）和机器人的当前位置 `qpos[:2]` 获取当前局部 `heightmap`（shape: `[H, W]`）。
- 将获取到的 `heightmap` 作为一个新的变量或者直接拼接进 `input` 数组传入 `self._ref_gen()`。

### 3. `mpx/utils/mpc_utils.py`

`reference_generator` 负责生成参考轨迹和 `parameter` 张量：
- 修改函数的输入签名接收 `heightmap`。
- 当前框架中，`parameter` 张量被传给 `dynamics` 和 `cost`。
- 我们需要将 $H \times W$ 的 `heightmap` 展平（或者直接保持矩阵形态），拼接到现有的 `parameter` 数组中。如果 `parameter` 当前维度（仅用于碰撞检测）容纳不了，扩充 `parameter` 的维度，使得 `shape` 中能够存放下整个高程图矩阵。

### 4. `mpx/utils/objectives.py`

这里定义了 MPC 的二次损失函数，增加针对双足足步离地间隙（clearance）和避免穿模的防碰撞成本函数（SDF / Heightmap penalty）：
- 实现基于双线性插值（`jax.scipy.ndimage.map_coordinates`）的连续且可微的地形查询函数 `get_terrain_height_at(x, y, heightmap_params)`。
- 添加 **`terrain_penetration_cost`**: 计算四个足端点（`p_leg`）位置，用目标点的 $z_{leg}$ 减去其所在平面的插值高程 $z_{terrain}$。
- 如果 $z_{leg} < z_{terrain}$（穿模），则利用 `penalty` 施加极大的对数势垒/二次惩罚（类似于目前的接触锥函数 `friction_cone`）。
- 将这个惩罚项加入现有的代价计算（对于阶段代价 `stage_cost` 叠加）。由于这个惩罚依赖于 `p_leg`（属于状态向量 `x`），所以 `hessian_gn` 也能自动通过 `jax.jacobian` 计算得到关于状态的梯度。

### 5. `mpx/config/config_r2_stairs.py`

启用地形感知和成本权重：
- 初始化/更新目标权重 `W`：分配一个新的权重给 `terrain_penetration_cost`。
- 将带有 `heightmap` 惩罚逻辑的新 `cost` 和 `hessian_approx` 函数，绑定到这里的控制配置中。

---

## Verification Plan

1. **Heightmap 提取验证**: 在 `visualize_terrain.py` 中单独调用一次射线扫描，绘制提取的高点，确认 raycast 没有因为 mask 设置或位置偏移而测错高度。
2. **零位势垒测试**: 传入生成的 WFC 楼梯地形，单独针对 `objectives.py` 中的 `terrain_penetration_cost` 计算其在一个假定坐标上的损失值与梯度，检查是否在穿模时陡增。
3. **闭环 MPC 仿真**: 运行 `examples/mjx_r2_rough_terrain.py`。
    - **Expected**: R2 机器人在具有不同高度的 WFC 阶梯地形上移动时，MPC 规划出的 `p_leg`（腿部末端落点）在摆动和落脚阶段会自动避开台阶阻挡，落脚在合适的高程，不再发生脚掌埋入台阶内部的穿模现象。
