"""
地形生成器 - 为 MPX 创建崎岖地面高度场

用法:
    python mpx/utils/terrain_generator.py --type perlin --output terrain.png
    python mpx/utils/terrain_generator.py --type steps --output steps.png
    python mpx/utils/terrain_generator.py --type rubble --output rubble.png
"""

import numpy as np
import argparse
from PIL import Image
import os


def generate_perlin_terrain(size=(256, 256), scale=0.1, height_scale=0.3, seed=42):
    """
    生成 Perlin 噪声地形（自然起伏）

    Args:
        size: 地形尺寸 (行, 列)
        scale: 噪声缩放因子（控制地形粗糙度）
        height_scale: 高度缩放（米）
        seed: 随机种子
    """
    np.random.seed(seed)

    # 使用多个频率的正弦波叠加模拟 Perlin 噪声
    x = np.linspace(0, scale * 2 * np.pi, size[1])
    y = np.linspace(0, scale * 2 * np.pi, size[0])
    X, Y = np.meshgrid(x, y)

    # 多层噪声叠加
    height = np.zeros_like(X)
    frequencies = [1, 2, 4, 8]
    amplitudes = [1.0, 0.5, 0.25, 0.125]

    for freq, amp in zip(frequencies, amplitudes):
        phase_x = np.random.rand() * 2 * np.pi
        phase_y = np.random.rand() * 2 * np.pi
        height += amp * np.sin(freq * X + phase_x) * np.sin(freq * Y + phase_y)

    # 归一化到 [0, 1] 然后缩放到高度范围
    height = (height - height.min()) / (height.max() - height.min())
    height = height * height_scale

    return height


def generate_steps_terrain(size=(256, 256), step_height=0.08, n_steps=5, seed=42):
    """
    生成阶梯地形

    Args:
        size: 地形尺寸 (行, 列)
        step_height: 每级台阶高度（米）
        n_steps: 台阶数量
        seed: 随机种子
    """
    np.random.seed(seed)

    height = np.zeros(size)
    step_width = size[1] // (n_steps + 1)

    for i in range(n_steps):
        start_x = i * step_width + np.random.randint(-10, 10)
        end_x = (i + 1) * step_width + np.random.randint(-10, 10)
        start_x = np.clip(start_x, 0, size[1])
        end_x = np.clip(end_x, 0, size[1])

        # 添加斜坡过渡
        transition = np.linspace(0, 1, min(20, end_x - start_x))

        for col in range(size[1]):
            if col < start_x:
                continue
            elif col < start_x + len(transition):
                height[:, col] = (i + 1) * step_height * transition[col - start_x]
            elif col < end_x:
                height[:, col] = (i + 1) * step_height
            elif col < end_x + len(transition):
                height[:, col] = (i + 1) * step_height * (1 - transition[col - end_x])

    return height


def generate_rubble_terrain(size=(256, 256), num_rocks=50, max_rock_height=0.15, seed=42):
    """
    生成碎石/障碍物地形

    Args:
        size: 地形尺寸 (行, 列)
        num_rocks: 障碍物数量
        max_rock_height: 最大障碍物高度（米）
        seed: 随机种子
    """
    np.random.seed(seed)

    height = np.zeros(size)
    x = np.arange(size[1])
    y = np.arange(size[0])
    X, Y = np.meshgrid(x, y)

    for _ in range(num_rocks):
        # 随机选择障碍物中心
        cx = np.random.randint(0, size[1])
        cy = np.random.randint(0, size[0])

        # 随机大小和高度
        radius_x = np.random.randint(5, 30)
        radius_y = np.random.randint(5, 30)
        rock_height = np.random.uniform(0.02, max_rock_height)

        # 创建高斯形状的障碍物
        gaussian = rock_height * np.exp(-((X - cx)**2 / (2 * radius_x**2) + (Y - cy)**2 / (2 * radius_y**2)))
        height = np.maximum(height, gaussian)

    return height


def generate_slope_terrain(size=(256, 256), slope_angle=15, seed=42):
    """
    生成斜坡地形

    Args:
        size: 地形尺寸 (行, 列)
        slope_angle: 斜坡角度（度）
        seed: 随机种子
    """
    np.random.seed(seed)

    # 计算斜坡高度差
    slope_radians = np.radians(slope_angle)
    max_height = size[1] * np.tan(slope_radians) * (4.0 / size[1])  # 假设地形宽度 4 米

    height = np.zeros(size)
    for col in range(size[1]):
        height[:, col] = col / size[1] * max_height

    # 添加轻微噪声使其更自然
    noise = generate_perlin_terrain(size, scale=0.05, height_scale=0.02)
    height = height + noise

    return height


def save_height_field(height, output_path, flip_vertical=True):
    """
    保存高度场为 PNG 图像

    MuJoCo hfield 需要的格式：
    - PNG 灰度图
    - 像素值映射到高度 [0, 255]
    - 注意：MuJoCo 会从下到上读取，所以可能需要垂直翻转

    Args:
        height: 高度数组 (H, W)
        output_path: 输出路径
        flip_vertical: 是否垂直翻转（MuJoCo 需要）
    """
    if flip_vertical:
        height = np.flipud(height)

    # 归一化到 [0, 255]
    height_normalized = (height - height.min()) / (height.max() - height.min() + 1e-10)
    image_array = (height_normalized * 255).astype(np.uint8)

    # 保存为 PNG
    img = Image.fromarray(image_array, mode='L')
    img.save(output_path)

    print(f"地形已保存到: {output_path}")
    print(f"  高度范围: [{height.min():.3f}, {height.max():.3f}] 米")
    print(f"  尺寸: {height.shape}")

    return height.min(), height.max()


def generate_mjx_hfield_code(height, size_meters=(4, 4), max_height=0.3, var_name="hfield_data"):
    """
    生成可直接在 MuJoCo XML 中使用的嵌入代码

    MuJoCo 支持在 XML 中直接嵌入 CSV 格式的高度数据
    """
    # 转换为 CSV 格式（每行一行）
    csv_lines = []
    for row in height:
        csv_lines.append(','.join([f'{v:.4f}' for v in row]))

    csv_data = '\\\n'.join(csv_lines)

    xml_template = f'''
<!-- 在 <asset> 标签中添加 -->
<hfield name="custom_terrain" nrow="{height.shape[0]}" ncol="{height.shape[1]}"
        size="{size_meters[0]} {size_meters[1]} {max_height}">
  {csv_data}
</hfield>
'''
    return xml_template


def main():
    parser = argparse.ArgumentParser(description='生成崎岖地形高度场')
    parser.add_argument('--type', type=str, default='perlin',
                        choices=['perlin', 'steps', 'rubble', 'slope'],
                        help='地形类型')
    parser.add_argument('--output', type=str, default='terrain.png',
                        help='输出 PNG 文件路径')
    parser.add_argument('--size', type=int, default=256,
                        help='地形尺寸（像素）')
    parser.add_argument('--height', type=float, default=0.3,
                        help='最大高度（米）')
    parser.add_argument('--seed', type=int, default=42,
                        help='随机种子')

    args = parser.parse_args()

    size = (args.size, args.size)

    print(f"正在生成 {args.type} 地形...")

    if args.type == 'perlin':
        height = generate_perlin_terrain(size, height_scale=args.height, seed=args.seed)
    elif args.type == 'steps':
        height = generate_steps_terrain(size, step_height=args.height / 5, seed=args.seed)
    elif args.type == 'rubble':
        height = generate_rubble_terrain(size, max_rock_height=args.height, seed=args.seed)
    elif args.type == 'slope':
        height = generate_slope_terrain(size, slope_angle=15, seed=args.seed)

    # 保存 PNG
    output_dir = os.path.dirname(args.output) or '.'
    os.makedirs(output_dir, exist_ok=True)
    save_height_field(height, args.output)

    # 如果是在 mpx 项目中，也可以生成 scene 配置
    if 'mpx' in os.getcwd():
        scene_xml = f"""<!-- 在 mpx/data/r2-1024/mjcf/ 目录下创建 scene_{args.type}.xml -->
<asset>
  <hfield name="{args.type}_terrain" nrow="{args.size}" ncol="{args.size}"
          size="4 4 {args.height}" file="{args.output}"/>
</asset>

<worldbody>
  <geom name="terrain" type="hfield" hfield="{args.type}_terrain"
        pos="0 0 0" size="4 4 {args.height} 0.1"/>
</worldbody>
"""
        print("\n=== MuJoCo XML 配置 ===")
        print(scene_xml)


if __name__ == '__main__':
    main()
