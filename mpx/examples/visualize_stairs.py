#!/usr/bin/env python3
"""
可视化台阶场景
"""

import pickle
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def load_stairs_data(data_path):
    """加载并可视化台阶轨迹数据"""
    with open(data_path, 'rb') as data:
        print(f"加载数数据: {data_path}")

        # 提取数据
        time = data.get('time')
        base_pos = data.get('base_position')
        base_vel = data.get('base_velocity')
        joint_pos = data.get('joint_position')
        joint_vel = data.get('joint_velocity')
        joint_torques = data.get('joint_torques')
        foot_pos = data.get('foot_position')
        ref_vel = data.get('reference_velocity')

        # 建议使用相对时间
        time_steps = np.arange(len(time))
        base_pos = np.array(base_pos)
        base_vel = np.array(base_vel)
        joint_pos = np.array(joint_pos)
        joint_vel = np.array(joint_vel)
        joint_torques = np.array(joint_torques)
        ref_vel = np.array(ref_vel)

        # 绘制基座位置 XY平面
        plt.figure(figsize=(14, 8))
        plt.title('R2机器人台阶场景测试', fontsize=16)
        plt.xlabel('基座位置轨迹', fontsize=12)

        # 3D轨迹
        ax = fig.add_subplot(1, 1, 1, projection='3d')
        ax.plot(base_pos[:, 0], base_pos[:, 1], base_pos[:, 2], 'b-', linewidth=2)
        ax.set_xlabel('X (米)')
        ax.set_ylabel('Y(米)')
        ax.set_zlabel('Z(米)')
        ax.set_title('基座位置 XY')
        ax.legend()
        ax.grid(True)

        # 绘制基座高度
        plt.figure()
        plt.plot(time, base_pos[:, 2], label='基座高度')
        plt.xlabel('时间(秒)')
        plt.ylabel('高度(米)')
        plt.legend()
        plt.grid(True)

        # 绘制XY轨迹
        plt.figure()
        plt.plot(base_pos[:, 0], base_pos[:, 1], 'r-', label='XY轨迹')
        plt.xlabel('X(米)')
        plt.ylabel('Y(米)')
        plt.legend()
        plt.grid(True)

        plt.tight_layout()
        plt.show()


        print(f"\n可视化完成！ 数据已加载自: {data_path}")
        print(f"时间范围: {time[-1]:.1f} - {time[-1]:.1f} 秒")
        print(f"基座位置范围: X: {base_pos[:, 0].min():.3f}, {base_pos[:, 0].max():.3f} 秒")
        print(f"基座高度范围: {base_pos[:, 2].min():.3f}, {base_pos[:, 2].max():.3f} 秒")
        print(f"最大高度: {base_pos[:, 2].max():.3f} 秒")
        print("\n轨迹数据可视化完成！ 图片已保存到: {fig_path}")

    except FileNotFoundError:
        print(f"错误: 找不到数据文件 {data_path}")
        return

    else:
        print(f"错误: 加载数数据 {data_path}")
        return

    else:
        print("可视化脚本不存在，退出。)

if __name__ == '__main__':
    import sys
    import os

    data_dir = '/home/wzn/双足/mpx/data/trajectories'
    latest_file = find_latest_stairs_data(os.path.join(data_dir, 'stairs_trajectory.pkl'))
    visualize_data(latest_file)
