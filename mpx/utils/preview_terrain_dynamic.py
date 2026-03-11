"""
MuJoCo 动态地形预览工具 - 在运行时生成崎岖地形

用法:
    python mpx/utils/preview_terrain_dynamic.py
"""

import mujoco
import mujoco.viewer
import numpy as np


def generate_perlin_terrain(size=(64, 64), scale=0.3, height_scale=0.8, seed=42):
    """生成类 Perlin 噪声地形 - 超级崎岖版本"""
    np.random.seed(seed)

    x = np.linspace(0, scale * 4 * np.pi, size[1])
    y = np.linspace(0, scale * 4 * np.pi, size[0])
    X, Y = np.meshgrid(x, y)

    height = np.zeros_like(X)
    # 更多频率分量，更复杂的地形
    frequencies = [1, 2, 4, 8, 16, 32]
    amplitudes = [1.0, 0.8, 0.5, 0.3, 0.15, 0.08]

    for freq, amp in zip(frequencies, amplitudes):
        phase_x = np.random.rand() * 2 * np.pi
        phase_y = np.random.rand() * 2 * np.pi
        # 使用更复杂的波形组合
        height += amp * np.sin(freq * X + phase_x) * np.cos(freq * Y + phase_y)
        height += amp * 0.5 * np.sin(freq * X * 1.3 + phase_x) * np.sin(freq * Y * 0.7 + phase_y)

    # 添加更多"山峰"特征
    for _ in range(15):
        cx = np.random.rand() * size[1]
        cy = np.random.rand() * size[0]
        radius = np.random.randint(3, 12)
        peak_height = np.random.uniform(0.5, 1.5) * height_scale

        gaussian = peak_height * np.exp(-((X - cx * size[1]/64)**2 + (Y - cy * size[0]/64)**2) / (2 * (radius/12)**2))
        height += gaussian

    # 添加一些深坑
    for _ in range(5):
        cx = np.random.rand() * size[1]
        cy = np.random.rand() * size[0]
        radius = np.random.randint(8, 20)
        pit_depth = np.random.uniform(0.3, 0.6) * height_scale

        gaussian = -pit_depth * np.exp(-((X - cx * size[1]/64)**2 + (Y - cy * size[0]/64)**2) / (2 * (radius/10)**2))
        height += gaussian

    height = (height - height.min()) / (height.max() - height.min())
    height = height * height_scale
    return height


def main():
    # 创建基本的场景 XML（使用空的高度场）
    xml = """
    <mujoco model="rough_terrain_preview">
      <statistic center="0 0 1" extent="2"/>
      <option timestep="0.002" iterations="50" solver="Newton"/>

      <visual>
        <headlight diffuse="0.6 0.6 0.6" ambient="0.3 0.3 0.3" specular="0 0 0"/>
        <rgba haze="0.15 0.25 0.35 1"/>
        <global azimuth="120" elevation="-15"/>
      </visual>

      <asset>
        <texture type="skybox" builtin="gradient" rgb1="0.3 0.5 0.7" rgb2="0 0 0" width="512" height="3072"/>
        <!-- 纯色地面材质 -->
        <material name="ground_terrain" rgba="0.5 0.45 0.35 1" reflectance="0.1" specular="0" shininess="0.1"/>

        <!-- 空的高度场，稍后在代码中填充 -->
        <!-- MuJoCo 3.4.0 需要 4 个值: x_size y_size z_max bottom_offset -->
        <!-- 超高分辨率 512x512 让地形非常平滑真实 -->
        <hfield name="rough_terrain" nrow="512" ncol="512" size="6 6 1.0 0.05"/>
      </asset>

      <worldbody>
        <light pos="1 1 4" dir="0 0 -1" directional="true"/>
        <geom name="terrain" type="hfield" hfield="rough_terrain"
              material="ground_terrain" contype="2" conaffinity="1"/>

        <!-- 添加一个简单的球体机器人用于测试 -->
        <body name="robot" pos="0 0 1.2">
          <joint name="free_x" type="slide" axis="1 0 0" damping="10"/>
          <joint name="free_y" type="slide" axis="0 1 0" damping="10"/>
          <joint name="free_z" type="slide" axis="0 0 1" damping="10"/>
          <geom name="body" type="sphere" size="0.15" mass="5" rgba="0.8 0.3 0.3 1"/>
        </body>
      </worldbody>
    </mujoco>
    """

    print("\n" + "="*50)
    print("生成动态崎岖地形...")
    print("="*50 + "\n")

    # 加载模型
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)

    # 生成地形数据
    nrow = model.hfield_nrow[0]
    ncol = model.hfield_ncol[0]
    print(f"高度场尺寸: {nrow} x {ncol}")

    # 生成 Perlin 噪声地形 - 超级崎岖的参数
    height_data = generate_perlin_terrain(size=(nrow, ncol), scale=0.4, height_scale=0.8, seed=42)

    # 填充高度场数据（MuJoCo 需要按行主序，并且要翻转垂直方向）
    # MuJoCo hfield 数据是按行主序存储的
    height_flat = height_data.flatten()

    # 找到 hfield 数据的地址
    hfield_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_HFIELD, "rough_terrain")
    adr = model.hfield_adr[hfield_id]
    n_elem = nrow * ncol

    # 设置高度数据
    model.hfield_data[adr:adr + n_elem] = height_flat

    print(f"地形高度范围: [{height_data.min():.3f}, {height_data.max():.3f}] 米")
    print(f"地形物理尺寸: 6m x 6m")
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
        viewer.cam.distance = 5
        viewer.cam.lookat[:] = [0, 0, 0.2]

        step = 0
        while viewer.is_running():
            mujoco.mj_step(model, data)
            viewer.sync()
            step += 1

            if step % 1000 == 0:
                # 打印球体位置
                ball_pos = data.qpos[:3]
                ball_height = ball_pos[2]
                print(f"球体位置: ({ball_pos[0]:.2f}, {ball_pos[1]:.2f}, {ball_height:.2f})")


if __name__ == '__main__':
    main()
