#!/usr/bin/env python3
"""简单的 MuJoCo 场景加载示例"""
import mujoco
import mujoco.viewer

# 1. 加载模型
model = mujoco.MjModel.from_xml_path("mpx/data/r2-1024/mjcf/scene_terrain_test.xml")
data = mujoco.MjData(model)

# 2. 重置到初始姿态
mujoco.mj_resetDataKeyframe(model, data, 0)

print(f"场景加载成功!")
print(f"  - 主体数量: {model.nbody}")
print(f"  - 地形方块: {sum(1 for i in range(model.nbody) if model.body(i).name.startswith('box_'))}")
print(f"  - 机器人初始高度: {data.qpos[2]:.3f}m")

# 3. 启动可视化查看器
with mujoco.viewer.launch_passive(model, data) as viewer:
    # 设置相机视角
    viewer.cam.distance = 8.0
    viewer.cam.elevation = -30
    viewer.cam.azimuth = 45
    viewer.cam.lookat[:] = [0, 0, 0.5]

    # 主循环
    while viewer.is_running():
        # 运行物理仿真
        mujoco.mj_step(model, data)
        viewer.sync()

print("可视化结束")
