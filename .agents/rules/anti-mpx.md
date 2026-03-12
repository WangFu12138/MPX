---
trigger: always_on
---
> [!IMPORTANT]
任何时候回答我都要先称呼我为王甫12138
除非我特殊要求，不然都请用中文回答我的问题
项目的每个文件夹层级下都有md文档，每当修改了当前的文件夹代码我确认无误后，都需要同步修改md文档，保证最新。同样的当你每次开始修改前都请务必读一下md文档，保证你对当前文件夹的代码有清晰的认识。
一、 理论与概念规则 (Theory & Concept Rules)(用于在 prompt 中设定学术背景和数学框架，确保推导的严谨性)核心架构定义 (Core Architecture): 本项目 (agent-antigravity) 的核心是一个端到端的闭环系统。视觉模块作为感知前端，直接输出潜在状态 (Latent State) 或直接耦合到 MPX 求解器中，避免传统 pipeline 中状态估计的累积误差。状态与观测表示 (State & Observation):定义视觉输入序列为 $V_t$。定义潜在状态表示为 $z_t = E_\theta(V_{t-k:t}, s_t)$，其中 $E_\theta$ 是参数化的视觉编码器，$s_t$ 可能包含本体感受器 (Proprioception) 数据。MPX 优化问题构建 (MPX Formulation): 所有的控制策略必须被框定为一个有限时域的最优控制问题。在时刻 $t$，求解：$$U_t^* = \arg\min_{u_{0:N-1}} \sum_{k=0}^{N-1} J_{stage}(z_{t+k}, u_{t+k}) + J_{term}(z_{t+N})$$$$\text{s.t. } z_{k+1} = F_\phi(z_k, u_k) \quad \text{(Learned Dynamics)}$$$$u_{min} \le u_k \le u_{max}$$端到端可微性 (End-to-End Differentiability): 这是理论的核心。MPX 求解器必须被视为计算图中的一个可微层。要求理解并应用隐式微分 (Implicit Differentiation) 或展开 (Unrolling) 技术。损失函数 $\mathcal{L}$ 关于编码器参数 $\theta$ 的梯度必须能够通过 MPX 的 KKT 条件反向传播：$\frac{\partial \mathcal{L}}{\partial \theta} = \frac{\partial \mathcal{L}}{\partial U^*} \frac{\partial U^*}{\partial z} \frac{\partial z}{\partial \theta}$。Antigravity 概念的数学映射: 确保动力学模型 $F_\phi$ 能够捕捉高度非线性和欠驱动特性（如接触、腾空阶段）。成本函数 $J_{stage}$ 需要包含对姿态稳定、质心 (CoM) 轨迹和接触力的惩罚项，以实现“抗重力”的稳定运动。