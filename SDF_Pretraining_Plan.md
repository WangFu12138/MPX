# SDF 视觉预训练：在线生成与 JAX/Flax 端到端工程蓝图 (详细文件字典)

## 0. 核心架构：放弃 RL，构建纯净的监督学习流

王甫12138，你的要求非常对。在动手写任何代码测试之前，我们必须**像一本字典一样**，把所有的文件、每个文件里有哪些函数、这些函数究竟负责什么功能，以及它们是如何从参考仓库 `phase_guided_terrain_traversal` 演变过来的，全部定义清楚。

下面是我们要构建的整个 JAX 在线监督预训练框架的文件拓扑和函数签名设计。

---

## 一、 文件层级全景图 (Project Directory Structure)

```text
mpx/mpx/
├── sdf_pretrain/                      # [NEW] 专门用于 SDF 预训练的根目录
│   ├── __init__.py
│   ├── data/
│   │   ├── dataset_generator.py       # [NEW] 在线数据集生成核心 (替代 RL env)
│   │   ├── transforms.py              # [NEW] 坐标与矩阵系转换工具
│   │   └── analytical_sdf.py          # [NEW] 基于规则的 SDF 真值提取
│   ├── models/
│   │   ├── networks.py                # [NEW] ResNet 与 MLP 网络架构
│   │   └── losses.py                  # [NEW] 面向 JAX 的自动求导与 Loss 计算
│   └── train.py                       # [NEW] 主训练循环入口 (替代 PPO train.py)
```

---

## 二、 核心功能模块详细拆解

### 模块 A. 在线数据生成与坐标转换 (Data & Transforms)

基于我们手头的 `mpx/data/r2-1024/mjcf/scene_terrain_test.xml`，以及未来从参考仓库白嫖的随机地形算法。

#### `sdf_pretrain/data/dataset_generator.py`
**对标参考仓库**：功能类比于 `mujoco_playground/locomotion` 里面的 `Joystick` Env。但是，我们**不继承** RL 的 `Env` 类，而是写一个独立的数据生成器。
*   **`class SDFOnlineGenerator:`**
    *   **核心功能**：初始化 MuJoCo 场景，并充当一个在线的数据流管道。
*   **`def __init__(self, xml_path: str):`**
    *   **功能**：调用 `mujoco.MjModel.from_xml_path`，实例化模型。预先解析出 `<worldbody>` 里面所有的基础几何体（如 `<geom type="box">`）的全局参数信息并缓存。
*   **`def sample_lidar_poses(self, key: jax.random.PRNGKey, batch_size: int) -> jnp.ndarray:`**
    *   **解答问题1**：如何随机放置传感器？
    *   **功能**：在机器人可能行走的高度平面内（例如 $z \in [0.4, 0.8]$），基于均匀分布或特定的先验分布，随机生成一组相机的 6D 位姿 $(x, y, z, roll, pitch, yaw)$。
*   **`def get_heightmaps(self, lidar_poses: jnp.ndarray) -> jnp.ndarray:`**
    *   **功能**：利用 `mjx.ray` 计算在当前随机位姿下的 2.5D 高程图 $X$。

#### `sdf_pretrain/data/transforms.py`
**解答问题2**：如何处理 MuJoCo 全局坐标与局部坐标的转换？
*   **`def get_lidar_transforms(lidar_poses: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:`**
    *   **功能**：根据 6D 位姿，计算出雷达到世界的变换矩阵 $T_{lidar\_to\_world}$ (4x4) 和世界到雷达的变换矩阵 $T_{world\_to\_lidar}$。
*   **`def sample_local_queries(key: jax.random.PRNGKey, num_points: int) -> jnp.ndarray:`**
    *   **功能**：在雷达的**局部视锥坐标系**内撒点。比如在雷达的可视包围盒 $[-1, 1] \times [-1, 1] \times [0.1, 2.0]$ 内生成均匀点云和边缘噪声点，生成查询点 $Q_{local}$。
*   **`def local_to_global_points(queries_local: jnp.ndarray, T_lidar_to_world: jnp.ndarray) -> jnp.ndarray:`**
    *   **功能**：这是一个矩阵乘法函数，将 $Q_{local}$ (拼接成齐次坐标) 乘以变换矩阵，输出全局查询点 $Q_{global}$，以便计算真值。

#### `sdf_pretrain/data/analytical_sdf.py`
**解答问题3**：如何通过实际计算得到 SDF 真值？
*   **`def parse_xml_to_boxes(xml_path: str) -> List[Dict]:`**
    *   **功能**：启动时调用一次。解析 XML，获取所有 `box` 的中心绝对坐标和长宽高参数。
*   **`def compute_sdf_box_jax(queries_global: jnp.ndarray, box_center: jnp.ndarray, box_halfsizes: jnp.ndarray) -> jnp.ndarray:`**
    *   **功能**：底层数学算子。给定一个装箱子的中心和尺寸，计算一批 $Q_{global}$ 点到这个盒子的精确解析法有符号距离。
*   **`def get_ground_truth_sdf(queries_global: jnp.ndarray, boxes_info: List[Dict]) -> jnp.ndarray:`**
    *   **功能**：顶层真值提取器。利用 `jax.vmap` 并行计算 $Q_{global}$ 到所有地形盒子的距离，然后执行 `jnp.min()`，最终输出完美的 $\text{SDF}_{gt}$ 标量值。

---

### 模块 B. JAX 纯净网络架构 (Networks & Losses)

#### `sdf_pretrain/models/networks.py`
**对标参考仓库**：替代 `brax.training.agents.ppo.networks.make_ppo_networks` 里面复杂的 Actor-Critic 和 Value 网络。
*   **`class HeightmapEncoder(nn.Module):`**
    *   **功能**：纯视觉骨干。使用 Flax 构建的类似 ResNet-18 的架构。输入 `[Batch, H, W, 1]` 的高程图，提取出抽象的隐表示 `[Batch, latent_dim]` 的特征 $Z$。
*   **`class ImplicitSDFDecoder(nn.Module):`**
    *   **功能**：几何解码器。将隐特征 $Z$ 和 3D 查询点 $Q_{local}$ 拼接。核心注意：**隐层激活函数必须用 `nn.swish` (SiLU)**，保证平滑性，摒弃 ReLU。输出 SDF 预测值。
*   **`class EndToEndSDFNetwork(nn.Module):`**
    *   **功能**：包装类。负责连接 Encoder 和 Decoder。提供针对单点解码的方法，为了后续 `jax.grad` 优雅求导做铺垫。

#### `sdf_pretrain/models/losses.py`
*   **`def compute_eikonal_loss(model: EndToEndSDFNetwork, params, z: jnp.ndarray, q_local: jnp.ndarray) -> jnp.ndarray:`**
    *   **功能**：最能体现 JAX 价值的极简求导函数。利用 `jax.vmap(jax.grad(model.decode_point, argnums=1))` 计算输出对于输入坐标 $Q_{local}$ 的一阶导数，并强制让其 L2 模长逼近 $1.0$。
*   **`def sdf_total_loss(params, batch_data) -> Tuple[float, Dict]:`**
    *   **功能**：MSE 损失与 Eikonal 损失的加权求和总控。

---

### 模块 C. 训练大闭环 (The Trainer)

#### `sdf_pretrain/train.py`
**对标参考仓库**：彻底抛弃 `phase_guided_terrain_traversal/training/train.py` 中依赖的 `brax.training.agents.ppo.train` 函数及其内部封装的环境迭代器。
*   **`def create_train_step(model, optimizer):`**
    *   **功能**：构建带有 `@jax.jit` 装饰器的极致加速更新步。它接受网络的参数状态和上述生成模块产出的批量数据，调用 Optax 优化器算出梯度并更新。
*   **`def main():`**
    *   **功能**：主入口。
        1. 实例化 `SDFOnlineGenerator`。
        2. 实例化 Flax 网络和 Optax 优化器。
        3. 用一个极简的朴素 `while` 循环（如 `for step in range(num_epochs)`）：
            * 调用生成器拿到一个批次的 $(X, Q_{local}, \text{SDF}_{gt})$。
            * 把数据扔进 `train_step()`。
            * 在终端打印 Loss 和进度。

---

## 下一步行动提示 (Action Required)

王甫12138，你的拆分思路极其专业！上述这份清单已经把所有的“黑盒子”全部打开了，每一个输入输出都非常清晰。

现在的提议：**我们把这份长长的清单里最吃物理理解的部分先通过测试脚本落地。**
如果我们继续在 `scene_terrain_test.xml` 上工作，我们可以先建一个测试临时脚本：
1. 实现 `parse_xml_to_boxes` 解析出 XML 里的各种 Box。
2. 实现 `get_camera_transforms` 并手动写一个随机相机位姿。
3. 实现 `local_to_global_points` 把一个相机的局部测试点映射回全局。
4. 调用 `compute_sdf_box_jax` 算出这个点到底有没有碰到地面或者箱子。

这个功能测试完全独立于网络，纯验证我们在空间的几何标定。是否批准先写这个 `geometry_test.py`？
