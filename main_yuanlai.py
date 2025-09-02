import math
import numpy as np
import matplotlib.pyplot as plt
from get_canshu import CanShu
from transfor_to_huaxiang import transfor_to_huaxiang
from houxu_control import houxu_control


class main_simulation():
    def __init__(self):
        self.R0 = 6371e3  # 地球半径 (m)
        self.g0 = 9.8  # 重力加速度 (m/s^2)
        self.V_scale = math.sqrt(self.g0 * self.R0)  # 速度无量纲化系数
        self.T_scale = math.sqrt(self.R0 / self.g0)  # 时间无量纲化系数

        # 初始状态（无量纲）
        self.initial_r = 1 + 80e3 / self.R0
        self.initial_v = 7100 / self.V_scale
        self.initial_jingdu = np.deg2rad(10)
        self.initial_weidu = np.deg2rad(-20)
        self.initial_theta = np.deg2rad(-1)
        self.initial_psi = np.deg2rad(45)

        # 目标状态
        self.lambda_target = np.deg2rad(57)  # 目标经度 (弧度)
        self.phi_target = np.deg2rad(15)  # 目标纬度 (弧度)
        self.v_target = 2000 / self.V_scale  # 目标速度 (无量纲)
        self.h_target = 35e3 / self.R0  # 目标高度 (无量纲)
        self.r_target = 1 + self.h_target  # 目标地心距 (无量纲)

        # 初始控制量
        self.initial_eta = 1
        self.initial_gongjiao = 20
        self.initial_qingce_rad = np.deg2rad(0)

        # 仿真参数
        sum_time = 2000  # 总时间 (s)
        self.to_wulianggang_sum_time = sum_time / self.T_scale  # 无量纲总时间

        # 初始化类
        self.canshu = CanShu()
        self.transfor = transfor_to_huaxiang()
        self.control = houxu_control()

    def dynamics(self, state, eta, gongjiao, qingce_rad):
        """
        动力学方程 - 论文公式(2)
        state: [r, lambda, phi, v, theta, psi]
        返回状态导数
        """
        r, lambda_val, phi, v, theta, psi = state
        h = r - 1

        # 计算升力和阻力加速度
        L, D = self.canshu.calculate_L_D(v, h, eta, gongjiao)

        # 状态方程 - 论文公式(2)
        r_dot = v * math.sin(theta)
        lambda_dot = v * math.cos(theta) * math.sin(psi) / (r * math.cos(phi))
        phi_dot = v * math.cos(theta) * math.cos(psi) / r
        v_dot = -D - math.sin(theta) / (r * r)
        theta_dot = (1 / v) * (L * math.cos(qingce_rad) + (v * v - 1 / r) * math.cos(theta) / r)
        psi_dot = (1 / v) * (L * math.sin(qingce_rad) / math.cos(theta) +
                             (v * v / r) * math.cos(theta) * math.sin(psi) * math.tan(phi))

        return np.array([r_dot, lambda_dot, phi_dot, v_dot, theta_dot, psi_dot])

    def runge_kutta_4(self, state, eta, gongjiao, qingce_rad, dt):
        """
        四阶龙格库塔积分
        """
        k1 = self.dynamics(state, eta, gongjiao, qingce_rad)
        k2 = self.dynamics(state + dt / 2 * k1, eta, gongjiao, qingce_rad)
        k3 = self.dynamics(state + dt / 2 * k2, eta, gongjiao, qingce_rad)
        k4 = self.dynamics(state + dt * k3, eta, gongjiao, qingce_rad)

        return state + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)

    def check_terminal_conditions(self, state):
        """
        检查终端条件
        """
        r, lambda_val, phi, v, theta, psi = state
        h = r - 1

        # 检查是否到达目标
        if (abs(h - self.h_target) < 0.001 and  # 高度误差小于1km（无量纲）
                abs(v - self.v_target) < 0.01 and  # 速度误差
                abs(lambda_val - self.lambda_target) < np.deg2rad(1) and  # 经度误差
                abs(phi - self.phi_target) < np.deg2rad(1)):  # 纬度误差
            return True

        # 检查是否到达最低高度
        if h < 0.001:  # 高度过低
            return True

        return False

    def main(self):
        # 时间数组设置
        dt = 0.001   # 无量纲时间步长
        time_array = np.arange(0, self.to_wulianggang_sum_time, dt)
        N = len(time_array)

        # 初始化状态和控制量数组
        states = np.zeros((N, 6))  # [r, lambda, phi, v, theta, psi]
        controls = np.zeros((N, 3))  # [eta, gongjiao, qingce_rad]

        # 设置初始状态
        states[0] = [self.initial_r, self.initial_jingdu, self.initial_weidu,
                     self.initial_v, self.initial_theta, self.initial_psi]
        controls[0] = [self.initial_eta, self.initial_gongjiao, self.initial_qingce_rad]

        # 仿真变量
        in_glide_phase = False
        r_point = 0
        v_point = 0
        guidance_counter = 0
        last_transform_result = 0

        print("开始仿真...")

        for i in range(1, N):
            current_state = states[i - 1]
            current_eta = 1
            current_gongjiao = controls[i - 1, 1]
            current_qingce_rad = controls[i - 1, 2]

            # 检查是否进入滑翔段
            transform_result = self.transfor.transfor_to_huaxiang(
                current_state, current_eta, current_gongjiao, current_qingce_rad
            )

            # 判断从下降段转入滑翔段
            if last_transform_result == 0 and transform_result == 1:
                in_glide_phase = True
                r_point = current_state[0]
                v_point = current_state[3]
                guidance_counter = 0
                print(f"进入滑翔段，时间: {time_array[i] * self.T_scale:.2f}s")

            last_transform_result = transform_result

            # 控制量更新
            if not in_glide_phase:
                # 下降段：使用恒定控制量
                new_eta = 1
                new_gongjiao = self.initial_gongjiao
                new_qingce_rad = self.initial_qingce_rad
            else:
                # 滑翔段：每10个dt更新一次控制量
                guidance_counter += 1
                if guidance_counter >= 10:
                    try:
                        new_eta, new_gongjiao, new_qingce_rad = self.control.houxu_control(
                            current_state, current_eta, current_gongjiao,
                            current_qingce_rad, r_point, v_point
                        )
                        new_eta = 1
                        guidance_counter = 0
                    except Exception as e:
                        print(f"控制量计算出错: {e}")
                        new_eta =1
                        new_gongjiao = current_gongjiao
                        new_qingce_rad = current_qingce_rad
                else:
                    new_eta = 1
                    new_gongjiao = current_gongjiao
                    new_qingce_rad = current_qingce_rad

            # 四阶龙格库塔积分
            try:
                new_state = self.runge_kutta_4(current_state, new_eta, new_gongjiao, new_qingce_rad, dt)

                # 状态约束
                new_state[0] = max(new_state[0], 1.0)  # r不能小于1
                new_state[3] = max(new_state[3], 0.1)  # v不能小于0.1

                states[i] = new_state
                controls[i] = [new_eta, new_gongjiao, new_qingce_rad]

            except Exception as e:
                print(f"积分计算出错: {e}")
                states[i] = current_state
                controls[i] = [current_eta, current_gongjiao, current_qingce_rad]

            # 检查终端条件
            if self.check_terminal_conditions(states[i]):
                print(f"到达终端条件，时间: {time_array[i] * self.T_scale:.2f}s")
                states = states[:i + 1]
                controls = controls[:i + 1]
                time_array = time_array[:i + 1]
                break

            # 进度显示
            if i % 1000 == 0:
                h_real = (states[i][0] - 1) * self.R0 / 1000  # km
                v_real = states[i][3] * self.V_scale  # m/s
                print(f"进度: {i / N * 100:.1f}%, 高度: {h_real:.1f}km, 速度: {v_real:.1f}m/s")

        print("仿真完成，开始绘图...")

        # 恢复量纲并绘图
        self.plot_results(time_array, states, controls)

        return time_array, states, controls

    def plot_results(self, time_array, states, controls):
        """
        恢复量纲并绘制结果
        """
        # 恢复量纲
        time_real = time_array * self.T_scale  # 秒
        height_real = (states[:, 0] - 1) * self.R0 / 1000  # 千米
        velocity_real = states[:, 3] * self.V_scale  # m/s
        longitude_real = np.rad2deg(states[:, 1])  # 度
        latitude_real = np.rad2deg(states[:, 2])  # 度
        theta_real = np.rad2deg(states[:, 4])  # 航迹角，度
        psi_real = np.rad2deg(states[:, 5])  # 航向角，度

        eta_real = controls[:, 0]  # 展长变形量
        gongjiao_real = controls[:, 1]  # 攻角，度
        qingce_real = np.rad2deg(controls[:, 2])  # 倾侧角，度

        # 创建图形
        plt.rcParams['font.sans-serif'] = ['SimHei']  # 支持中文
        plt.rcParams['axes.unicode_minus'] = False

        fig, axes = plt.subplots(3, 3, figsize=(15, 12))

        # 1. 高度-时间
        axes[0, 0].plot(time_real, height_real)
        axes[0, 0].set_xlabel('时间 (s)')
        axes[0, 0].set_ylabel('高度 (km)')
        axes[0, 0].set_title('高度-时间曲线')
        axes[0, 0].grid(True)

        # 2. 速度-时间
        axes[0, 1].plot(time_real, velocity_real)
        axes[0, 1].set_xlabel('时间 (s)')
        axes[0, 1].set_ylabel('速度 (m/s)')
        axes[0, 1].set_title('速度-时间曲线')
        axes[0, 1].grid(True)

        # 3. 经度-纬度轨迹
        axes[0, 2].plot(longitude_real, latitude_real)
        axes[0, 2].plot(longitude_real[0], latitude_real[0], 'go', markersize=8, label='起点')
        axes[0, 2].plot(longitude_real[-1], latitude_real[-1], 'ro', markersize=8, label='终点')
        axes[0, 2].plot(np.rad2deg(self.lambda_target), np.rad2deg(self.phi_target),
                        'b*', markersize=12, label='目标点')
        axes[0, 2].set_xlabel('经度 (°)')
        axes[0, 2].set_ylabel('纬度 (°)')
        axes[0, 2].set_title('地面轨迹')
        axes[0, 2].legend()
        axes[0, 2].grid(True)

        # 4. 展长变形量-时间
        axes[1, 0].plot(time_real, eta_real)
        axes[1, 0].set_xlabel('时间 (s)')
        axes[1, 0].set_ylabel('展长变形量')
        axes[1, 0].set_title('展长变形量-时间曲线')
        axes[1, 0].grid(True)

        # 5. 攻角-时间
        axes[1, 1].plot(time_real, gongjiao_real)
        axes[1, 1].set_xlabel('时间 (s)')
        axes[1, 1].set_ylabel('攻角 (°)')
        axes[1, 1].set_title('攻角-时间曲线')
        axes[1, 1].grid(True)

        # 6. 倾侧角-时间
        axes[1, 2].plot(time_real, qingce_real)
        axes[1, 2].set_xlabel('时间 (s)')
        axes[1, 2].set_ylabel('倾侧角 (°)')
        axes[1, 2].set_title('倾侧角-时间曲线')
        axes[1, 2].grid(True)

        # 7. 航迹角-时间
        axes[2, 0].plot(time_real, theta_real)
        axes[2, 0].set_xlabel('时间 (s)')
        axes[2, 0].set_ylabel('航迹角 (°)')
        axes[2, 0].set_title('航迹角-时间曲线')
        axes[2, 0].grid(True)

        # 8. 航向角-时间
        axes[2, 1].plot(time_real, psi_real)
        axes[2, 1].set_xlabel('时间 (s)')
        axes[2, 1].set_ylabel('航向角 (°)')
        axes[2, 1].set_title('航向角-时间曲线')
        axes[2, 1].grid(True)

        # 9. 高度-速度
        axes[2, 2].plot(velocity_real, height_real)
        axes[2, 2].set_xlabel('速度 (m/s)')
        axes[2, 2].set_ylabel('高度 (km)')
        axes[2, 2].set_title('高度-速度曲线')
        axes[2, 2].grid(True)

        plt.tight_layout()
        plt.show()

        # 打印最终结果
        print("\n=== 仿真结果 ===")
        print(f"仿真时间: {time_real[-1]:.2f} 秒")
        print(f"最终高度: {height_real[-1]:.2f} km (目标: {self.h_target * self.R0 / 1000:.2f} km)")
        print(f"最终速度: {velocity_real[-1]:.2f} m/s (目标: {self.v_target * self.V_scale:.2f} m/s)")
        print(f"最终经度: {longitude_real[-1]:.2f}° (目标: {np.rad2deg(self.lambda_target):.2f}°)")
        print(f"最终纬度: {latitude_real[-1]:.2f}° (目标: {np.rad2deg(self.phi_target):.2f}°)")

        # 计算误差
        height_error = abs(height_real[-1] - self.h_target * self.R0 / 1000) * 1000  # m
        velocity_error = abs(velocity_real[-1] - self.v_target * self.V_scale)  # m/s
        longitude_error = abs(longitude_real[-1] - np.rad2deg(self.lambda_target))  # deg
        latitude_error = abs(latitude_real[-1] - np.rad2deg(self.phi_target))  # deg

        print(f"\n=== 误差分析 ===")
        print(f"高度误差: {height_error:.2f} m")
        print(f"速度误差: {velocity_error:.2f} m/s")
        print(f"经度误差: {longitude_error:.4f}°")
        print(f"纬度误差: {latitude_error:.4f}°")


if __name__ == "__main__":
    sim = main_simulation()
    time_array, states, controls = sim.main()