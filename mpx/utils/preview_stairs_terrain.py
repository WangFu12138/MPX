"""
楼梯地形预览工具 - 参考 preview_terrain_dynamic.py 的正确配置

运行方式:
    python mpx/utils/preview_stairs_terrain.py
"""

import mujoco
import mujoco.viewer
import numpy as np


def generate_stairs_terrain(size=(512, 512), num_steps=10, step_height=0.08, step_width=None, add_noise=0.005, seed=42):
    """
    生成楼梯地形
    """
    np.random.seed(seed)
    nrow, ncol = size

    if step_width is None:
        step_width = 6.0 / num_steps

    x = np.linspace(0, 6, ncol)
    y = np.linspace(0, 6, nrow)
    X, Y = np.meshgrid(x, y)

    height = np.zeros_like(X)

    # 生成楼梯
    for i in range(num_steps):
        x_start = i * (6.0 / num_steps)
        x_end = (i + 1) * (6.0 / num_steps)

        step_mask = (X >= x_start) & (X < x_end)
        current_height = (i + 1) * step_height

        transition_width = 0.05
        height[step_mask & (X < x_end - transition_width)] = current_height

        ramp_mask = step_mask & (X >= x_end - transition_width)
        if np.any(ramp_mask):
            ramp_progress = (X[ramp_mask] - (x_end - transition_width)) / transition_width
            height[ramp_mask] = current_height + ramp_progress * step_height

    # 添加轻微噪声
    if add_noise > 0:
        noise = np.random.randn(nrow, ncol) * add_noise
        try:
            from scipy.ndimage import gaussian_filter
            noise = gaussian_filter(noise, sigma=2)
        except ImportError:
            # 如果没有 scipy，就用简单的平滑
            pass
        height += noise

    height = np.maximum(height, 0)
    return height


def main():
    # 场景配置
    num_steps = 10
    step_height = 0.1

    xml = f"""
    <mujoco model="stairs_terrain">
      <statistic center="0 0 1" extent="2"/>
      <option timestep="0.002" iterations="50" solver="Newton"/>

      <visual>
        <headlight diffuse="0.6 0.6 0.6" ambient="0.3 0.3 0.3" specular="0 0 0"/>
        <rgba haze="0.15 0.25 0.35 1"/>
        <global azimuth="120" elevation="-15"/>
      </visual>

      <asset>
        <texture type="skybox" builtin="gradient" rgb1="0.3 0.5 0.7" rgb2="0 0 0" width="512" height="3072"/>
        <material name="stairs" rgba="0.6 0.6 0.6 1" reflectance="0.2" specular="0.1" shininess="0.2"/>

        <!-- 高度场配置 -->
        <hfield name="stairs_terrain" nrow="512" ncol="512" size="6 6 1.2 0.05"/>
      </asset>

      <worldbody>
        <light pos="1 1 4" dir="0 0 -1" directional="true"/>
        <geom name="terrain" type="hfield" hfield="stairs_terrain"
              material="stairs" contype="2" conaffinity="1"/>

        <!-- 使用滑动关节 + 旋转关节的球体（6 自由度，可以滚动） -->
        <body name="ball" pos="0 0 1.2">
          <!-- 平移关节 -->
          <joint name="slide_x" type="slide" axis="1 0 0" damping="10"/>
          <joint name="slide_y" type="slide" axis="0 1 0" damping="10"/>
          <joint name="slide_z" type="slide" axis="0 0 1" damping="10"/>
          <!-- 旋转关节（绕球心旋转） -->
          <joint name="rot_x" type="hinge" axis="1 0 0" damping="1"/>
          <joint name="rot_y" type="hinge" axis="0 1 0" damping="1"/>
          <joint name="rot_z" type="hinge" axis="0 0 1" damping="1"/>
          <geom name="body" type="sphere" size="0.15" mass="5" rgba="0.8 0.3 0.3 1"
                friction="0.3 0.001 0.0001"/>
        </body>
      </worldbody>
    </mujoco>
    """

    print("\n" + "="*50)
    print("生成楼梯地形...")
    print(f"楼梯数量: {num_steps}")
    print(f"每级高度: {step_height}m")
    print(f"总高度: {num_steps * step_height:.2f}m")
    print("="*50 + "\n")

    # 加载模型
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)

    # 生成地形
    nrow = model.hfield_nrow[0]
    ncol = model.hfield_ncol[0]
    print(f"高度场尺寸: {nrow} x {ncol}")

    height_data = generate_stairs_terrain(
        size=(nrow, ncol),
        num_steps=num_steps,
        step_height=step_height,
        add_noise=0.002,
        seed=42
    )

    print(f"地形高度范围: [{height_data.min():.3f}, {height_data.max():.3f}] 米")

    # 填充到 MuJoCo
    hfield_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_HFIELD, "stairs_terrain")
    adr = model.hfield_adr[hfield_id]
    n_elem = nrow * ncol
    model.hfield_data[adr:adr + n_elem] = height_data.flatten()

    print("\n控制:")
    print("  - 鼠标左键拖动: 旋转视角")
    print("  - 鼠标右键拖动: 平移视角")
    print("  - 鼠标滚轮: 缩放")
    print("  - Ctrl+C: 退出\n")

    # 启动查看器
    with mujoco.viewer.launch_passive(model, data) as viewer:
        print("查看器已启动...\n")

        viewer.cam.azimuth = 120
        viewer.cam.elevation = -15
        viewer.cam.distance = 6
        viewer.cam.lookat[:] = [3, 0, 0.5]

        step = 0
        while viewer.is_running():
            mujoco.mj_step(model, data)
            viewer.sync()
            step += 1

            if step % 500 == 0:
                # 6 个自由度：3 平移 + 3 旋转
                ball_pos = data.qpos[:3]
                ball_rot = data.qpos[3:6]
                print(f"球体位置: ({ball_pos[0]:.2f}, {ball_pos[1]:.2f}, {ball_pos[2]:.2f}) | "
                      f"旋转: ({ball_rot[0]:.2f}, {ball_rot[1]:.2f}, {ball_rot[2]:.2f}) rad")


if __name__ == '__main__':
    main()
