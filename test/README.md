# Test 目录说明

`test` 目录主要负责存放一些**理论验证或者基础算法功能的抽样测试脚本**，而非项目整体的单元测试套件。

## 包含文件及功能：

- **`IFT.py`**:
  隐函数定理 (Implicit Function Theorem, IFT) 的极简概念验证代码。
  本项目中最核心的理论支撑点是通过隐式微分机制，让模型从最优解直接推导反向梯度（用于视觉特征修改代价参数），而无需重演求解器内所有的展开步骤。该文件提供了一个纯 JAX 语法的极小化模型供开发者学习隐式求解的具体代码执行流 (如何使用 `jax.custom_vjp`, `jacobian` 以及求求解 Hessian 特征线性方程来代替自动微分链式展开)。

- **`mpc_gradient_concept.py`**:
  MPC 可导性验证概念验证脚本。
  验证 `jax.grad(mpc_solve)` 能够稳定回传梯度，确保对代价矩阵权重参数的求导没有梯度爆炸或丢失。测试场景为 2D 质点追踪圆形轨迹。包含 5 个测试：
  1. 基础梯度回传测试
  2. 梯度正确性验证（与数值梯度对比）
  3. 梯度稳定性测试（多种边界条件）
  4. 参数敏感性测试
  5. 端到端学习测试

  运行方式：
  ```bash
  python test/mpc_gradient_concept.py
  ```

  正式的单元测试套件位于 `mpx/primal_dual_ilqr/tests/test_differentiability.py`。
