# 端到端视觉可微 MPC 模块开发 TODO 列表

## 模块 1：MPC 求解器 & MuJoCo 仿真基石 (MPC Solver & MuJoCo Foundation)
- [x] **任务 1.1：可导性验证单元测试** ✅ 2026-03-03
  - **目标**：在整个系统集成前，直接测试并验证 `jax.grad(mpc_solve)` 能够稳定回传梯度。
  - **具体动作**：编写一个极为简化的物理追踪任务（如质点追踪某轨迹）。利用当前项目已有的 `MPX` 求解器底座提取输出操作，计算对代价矩阵权重参数的求导。
  - **产出**：
    - `test/mpc_gradient_concept.py` - 概念验证脚本
    - `mpx/primal_dual_ilqr/tests/test_differentiability.py` - 正式单元测试套件
  - **关键发现**：
    - `primal_dual_ilqr` 使用 `lax.while_loop`，JAX 反向模式微分不直接支持
    - 解决方案1：使用固定迭代次数的简化版求解器（已实现，可用）
    - 解决方案2：完整的 IFT（隐函数定理）实现（需要更多工作）
  - **验证结果**：
    - 简化版可微 MPC ✅ 通过
    - 梯度正确性验证 ✅ 通过
    - 梯度稳定性测试 ⚠️ 大权重时可能爆炸（需要调参）
    - 端到端学习测试 ✅ 通过
  - **📦 开源情况**：无需额外开源库，直接使用本代码库中已包含的 `MPX` 算法实现（及 `test/IFT.py` 作为隐函数定理求导实现参考）即可。
  - **✅ 模块 1 完成 (2026-03-04)**：概念验证 + IFT 全量求解器 + 梯度稳定性均已通过，底座就绪。

**🔥 模块 1 后续攻坚任务 (死磕可导底座)**
- [x] **任务 1.2：完善全量求解器的隐函数定理 (IFT) 反向传播** ✅ 2026-03-04
  - **目标**：彻底解决 `lax.while_loop` 导致的不可导问题，使原装 `primal_dual_ilqr` 自身具备求导能力。
  - **具体动作**：在 `primal_dual_ilqr` 外部使用 `@custom_vjp` 进行包装，前向传播使用原求解器求解，反向传播提取最后一次迭代的 KKT 矩阵或 Hessian，利用隐式梯度公式 $-\left[ \frac{\partial^2 L}{\partial X^2} \right]^{-1} \frac{\partial^2 L}{\partial X \partial W}$ 解出解析梯度。
  - **产出**：`mpx/utils/differentiable_mpc.py` (新建)
  - **关键设计**：复用 `tvlqr` backward sweep 求解伴随变量，避免直接大矩阵求逆
- [x] **任务 1.3：解决大权重下的梯度爆炸与数值陷阱** ✅ 2026-03-04
  - **目标**：保证任何地形代价网络回传的极端梯度都被妥善平息。
  - **具体动作**：对数空间参数化 $W = \exp(w)$，随元素梯度裁剪 (`jnp.clip`)，Tikhonov 正则化提升 Hessian 条件数
  - **产出**：`test/test_differentiable_mpc_ift.py` (新建)
  - **验证结果**：5/5 测试全通过，包括权重=$10^5$ 时梯度仍然稳定无 NaN/Inf

## 模块 2：地形生成器 & 地形代价获取 (Terrain Generator & SDF Cost)
- [ ] **任务 2.1：高度图（Heightfield）与复杂地形生成器**
  - **目标**：在 MuJoCo 运行时能动态生成带碎石、台阶、斜坡的环境。
  - **具体动作**：基于正弦波叠加和 Perlin Noise 构建随机高度矩阵，并挂载到 MJCF 模型的 `hfield` 内。
  - **📦 开源情况**：已有 `utils/terrain_generator.py` 的基础。如果需要复杂噪声生成，JAX 社区的 Perlin Noise 开源代码或 Google DeepMind 的 `mujoco_playground` (其中包含丰富的程序化地形脚本) 可以极大加速进度。
- [ ] **任务 2.2：连续可微 SDF (Signed Distance Field) 计算**
  - **目标**：解决传统高程图不可导的问题，实现对 $(x,y)$ 坐标可微分的实时高度距离查询。
  - **具体动作**：利用输入的地形网格，通过 JAX 的 `jax.scipy.ndimage.map_coordinates`（并指定 `order=1` 双线性插值）构建 SDF 函数：$h(x, y)$。
  - **📦 开源情况**：这部分是核心创新逻辑，且所需算子在 JAX 标准库就已涵盖，因此建议**自己从头实现**，以保证定制化和求导稳定度。
- [ ] **任务 2.3：地形安全损失（Terrain Safety Loss）构建**
  - **目标**：将地形碰擦/碰撞的代价融入 MPC 的损失函数。
  - **具体动作**：在 `utils/objectives.py` 里添加 $J_{terrain} = \text{ReLU}(z_{foot\_ref} - SDF(x, y))$ 等防撞代价惩罚项，作为地形代价值，进而反向引导视觉模块的特征图。

## 模块 4：MuJoCo 中的视觉模块 (MuJoCo Perception)
- [ ] **任务 3.1：RGB-D 相机流与渲染框架封装**
  - **目标**：在仿真环境中获取机器人第一人称或环境视角的深度图（Depth Image）。
  - **具体动作**：在机器人的 MJCF 定义中添加 `<camera>` geom。通过 `mujoco.Renderer` 设置离屏渲染上下文并提取深度通道；或调研 `MJX` 原生提供的 GPU Raycasting 功能 (`mjx.ray`) 来极大地降低多环境并行时的渲染开销。
  - **📦 开源情况**：`DeepMind/mujoco_playground` 中对于 MJX / MuJoCo 视觉相机处理的例程是最好的学习和提取来源，其 API 契合我们的并行化要求。直接借用他们相机绑定的初始化逻辑再作修改是最佳路线。
- [ ] **任务 3.2：视觉管线前处理器 (Vision Pre-processor)**
  - **目标**：让渲染出的深度图符合模型的输入标注，并保持整个管线的可微或纯显存传递。
  - **具体动作**：在网络推断前对深度图进行归一化、降采样或裁剪。严禁中间使用 numpy / OpenCV CPU 函数，必须使用 `jax.image.resize` 等 JAX 原生操作确保张量不掉出 GPU 设备节点。

## 🛠 开发架构与代码规范约束 (Architecture Rules)
- **解耦性**：即使是端到端，也需确保目录里分别通过 `vision_encoder.py`，`dynamics_model.py` 以及核心 `mpc_wrapper.py` 隔离责任。
- **文档同步修订**：严格遵循用户要求（王甫12138的指示），以上任务在相关文件夹完成后，需要及时同步去修改其子目录的 `README.md`。
