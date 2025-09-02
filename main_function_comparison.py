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

    def simulate_single(self, aircraft_type):
        """
        单个飞行器仿真
        aircraft_type: '变形' 或 '传统'
        """
        # 时间数组设置
        dt = 0.001  # 无量纲时间步长
        time_array = np.arange(0, self.to_wulianggang_sum_time, dt)
        N = len(time_array)

        # 初始化状态和控制量数组
        states = np.zeros((N, 6))  # [r, lambda, phi, v, theta, psi]
        controls = np.zeros((N, 3))  # [eta, gongjiao, qingce_rad]

        # 设置初始状态
        states[0] = [self.initial_r, self.initial_jingdu, self.initial_weidu,
                     self.initial_v, self.initial_theta, self.initial_psi]

        # 根据飞行器类型设置初始eta值
        initial_eta = 2 if aircraft_type == '变形' else 1
        controls[0] = [initial_eta, self.initial_gongjiao, self.initial_qingce_rad]

        # 仿真变量
        in_glide_phase = False
        r_point = 0
        v_point = 0
        guidance_counter = 0
        last_transform_result = 0

        print(f"开始{aircraft_type}飞行器仿真...")

        for i in range(1, N):
            current_state = states[i - 1]
            current_eta = controls[i - 1, 0]
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
                print(f"{aircraft_type}飞行器进入滑翔段，时间: {time_array[i] * self.T_scale:.2f}s")

            last_transform_result = transform_result

            # 控制量更新
            if not in_glide_phase:
                # 下降段：使用恒定控制量
                if aircraft_type == '变形':
                    new_eta = 2
                else:  # 传统飞行器
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
                        # 对于传统飞行器，强制eta=1
                        if aircraft_type == '传统':
                            new_eta = 1
                        guidance_counter = 0
                    except Exception as e:
                        print(f"{aircraft_type}飞行器控制量计算出错: {e}")
                        new_eta = current_eta
                        new_gongjiao = current_gongjiao
                        new_qingce_rad = current_qingce_rad
                else:
                    new_eta = current_eta
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
                print(f"{aircraft_type}飞行器积分计算出错: {e}")
                states[i] = current_state
                controls[i] = [current_eta, current_gongjiao, current_qingce_rad]

            # 检查终端条件
            if self.check_terminal_conditions(states[i]):
                print(f"{aircraft_type}飞行器到达终端条件，时间: {time_array[i] * self.T_scale:.2f}s")
                states = states[:i + 1]
                controls = controls[:i + 1]
                time_array = time_array[:i + 1]
                break

            # 进度显示
            if i % 1000 == 0:
                h_real = (states[i][0] - 1) * self.R0 / 1000  # km
                v_real = states[i][3] * self.V_scale  # m/s
                print(f"{aircraft_type}飞行器进度: {i / N * 100:.1f}%, 高度: {h_real:.1f}km, 速度: {v_real:.1f}m/s")

        print(f"{aircraft_type}飞行器仿真完成")
        return time_array, states, controls

    def main(self):
        # 同时仿真两种飞行器
        print("=== 开始飞行器对比仿真 ===")

        # 变形飞行器仿真
        time_morphing, states_morphing, controls_morphing = self.simulate_single('变形')

        # 传统飞行器仿真
        time_traditional, states_traditional, controls_traditional = self.simulate_single('传统')

        # 输出对比结果
        self.print_comparison_results(time_morphing, states_morphing, controls_morphing,
                                      time_traditional, states_traditional, controls_traditional)

        # 绘制对比图
        self.plot_comparison_results(time_morphing, states_morphing, controls_morphing,
                                     time_traditional, states_traditional, controls_traditional)

        return (time_morphing, states_morphing, controls_morphing,
                time_traditional, states_traditional, controls_traditional)

    def print_comparison_results(self, time_morphing, states_morphing, controls_morphing,
                                 time_traditional, states_traditional, controls_traditional):
        """
        打印对比结果
        """
        # 恢复量纲
        time_m_real = time_morphing * self.T_scale
        height_m_real = (states_morphing[-1][0] - 1) * self.R0 / 1000
        velocity_m_real = states_morphing[-1][3] * self.V_scale
        longitude_m_real = np.rad2deg(states_morphing[-1][1])
        latitude_m_real = np.rad2deg(states_morphing[-1][2])

        time_t_real = time_traditional * self.T_scale
        height_t_real = (states_traditional[-1][0] - 1) * self.R0 / 1000
        velocity_t_real = states_traditional[-1][3] * self.V_scale
        longitude_t_real = np.rad2deg(states_traditional[-1][1])
        latitude_t_real = np.rad2deg(states_traditional[-1][2])

        # 目标值
        target_height = self.h_target * self.R0 / 1000
        target_velocity = self.v_target * self.V_scale
        target_longitude = np.rad2deg(self.lambda_target)
        target_latitude = np.rad2deg(self.phi_target)

        print("\n" + "=" * 60)
        print("                   飞行器性能对比结果")
        print("=" * 60)
        print(f"{'参数':<15} {'变形飞行器':<20} {'传统飞行器':<20} {'目标值':<15}")
        print("-" * 70)
        print(f"{'仿真时间(s)':<15} {time_m_real[-1]:<20.2f} {time_t_real[-1]:<20.2f} {'--':<15}")
        print(f"{'最终高度(km)':<15} {height_m_real:<20.2f} {height_t_real:<20.2f} {target_height:<15.2f}")
        print(f"{'最终速度(m/s)':<15} {velocity_m_real:<20.2f} {velocity_t_real:<20.2f} {target_velocity:<15.2f}")
        print(f"{'最终经度(°)':<15} {longitude_m_real:<20.4f} {longitude_t_real:<20.4f} {target_longitude:<15.4f}")
        print(f"{'最终纬度(°)':<15} {latitude_m_real:<20.4f} {latitude_t_real:<20.4f} {target_latitude:<15.4f}")

        print("\n" + "=" * 60)
        print("                     误差分析")
        print("=" * 60)

        # 计算误差
        height_error_m = abs(height_m_real - target_height) * 1000  # m
        velocity_error_m = abs(velocity_m_real - target_velocity)  # m/s
        longitude_error_m = abs(longitude_m_real - target_longitude)  # deg
        latitude_error_m = abs(latitude_m_real - target_latitude)  # deg

        height_error_t = abs(height_t_real - target_height) * 1000  # m
        velocity_error_t = abs(velocity_t_real - target_velocity)  # m/s
        longitude_error_t = abs(longitude_t_real - target_longitude)  # deg
        latitude_error_t = abs(latitude_t_real - target_latitude)  # deg

        print(f"{'参数':<15} {'变形飞行器误差':<20} {'传统飞行器误差':<20}")
        print("-" * 55)
        print(f"{'高度误差(m)':<15} {height_error_m:<20.2f} {height_error_t:<20.2f}")
        print(f"{'速度误差(m/s)':<15} {velocity_error_m:<20.2f} {velocity_error_t:<20.2f}")
        print(f"{'经度误差(°)':<15} {longitude_error_m:<20.6f} {longitude_error_t:<20.6f}")
        print(f"{'纬度误差(°)':<15} {latitude_error_m:<20.6f} {latitude_error_t:<20.6f}")
        print("=" * 60)

    def plot_comparison_results(self, time_morphing, states_morphing, controls_morphing,
                                time_traditional, states_traditional, controls_traditional):
        """
        绘制对比结果
        """
        # 恢复量纲 - 变形飞行器
        time_m_real = time_morphing * self.T_scale
        height_m_real = (states_morphing[:, 0] - 1) * self.R0 / 1000
        velocity_m_real = states_morphing[:, 3] * self.V_scale
        longitude_m_real = np.rad2deg(states_morphing[:, 1])
        latitude_m_real = np.rad2deg(states_morphing[:, 2])
        theta_m_real = np.rad2deg(states_morphing[:, 4])
        psi_m_real = np.rad2deg(states_morphing[:, 5])
        eta_m_real = controls_morphing[:, 0]
        gongjiao_m_real = controls_morphing[:, 1]
        qingce_m_real = np.rad2deg(controls_morphing[:, 2])

        # 恢复量纲 - 传统飞行器
        time_t_real = time_traditional * self.T_scale
        height_t_real = (states_traditional[:, 0] - 1) * self.R0 / 1000
        velocity_t_real = states_traditional[:, 3] * self.V_scale
        longitude_t_real = np.rad2deg(states_traditional[:, 1])
        latitude_t_real = np.rad2deg(states_traditional[:, 2])
        theta_t_real = np.rad2deg(states_traditional[:, 4])
        psi_t_real = np.rad2deg(states_traditional[:, 5])
        eta_t_real = controls_traditional[:, 0]
        gongjiao_t_real = controls_traditional[:, 1]
        qingce_t_real = np.rad2deg(controls_traditional[:, 2])

        # 创建图形
        plt.rcParams['font.sans-serif'] = ['SimHei']  # 支持中文
        plt.rcParams['axes.unicode_minus'] = False

        fig, axes = plt.subplots(3, 3, figsize=(18, 14))

        # 1. 高度-时间
        axes[0, 0].plot(time_m_real, height_m_real, 'b-', linewidth=2, label='变形飞行器')
        axes[0, 0].plot(time_t_real, height_t_real, 'r--', linewidth=2, label='传统固定外形飞行器')
        axes[0, 0].set_xlabel('时间 (s)')
        axes[0, 0].set_ylabel('高度 (km)')
        axes[0, 0].set_title('高度-时间曲线')
        axes[0, 0].legend()
        axes[0, 0].grid(True)

        # 2. 速度-时间
        axes[0, 1].plot(time_m_real, velocity_m_real, 'b-', linewidth=2, label='变形飞行器')
        axes[0, 1].plot(time_t_real, velocity_t_real, 'r--', linewidth=2, label='传统固定外形飞行器')
        axes[0, 1].set_xlabel('时间 (s)')
        axes[0, 1].set_ylabel('速度 (m/s)')
        axes[0, 1].set_title('速度-时间曲线')
        axes[0, 1].legend()
        axes[0, 1].grid(True)

        # 3. 经度-纬度轨迹
        axes[0, 2].plot(longitude_m_real, latitude_m_real, 'b-', linewidth=2, label='变形飞行器')
        axes[0, 2].plot(longitude_t_real, latitude_t_real, 'r--', linewidth=2, label='传统固定外形飞行器')
        axes[0, 2].plot(longitude_m_real[0], latitude_m_real[0], 'go', markersize=8, label='起点')
        axes[0, 2].plot(np.rad2deg(self.lambda_target), np.rad2deg(self.phi_target),
                        'k*', markersize=12, label='目标点')
        axes[0, 2].set_xlabel('经度 (°)')
        axes[0, 2].set_ylabel('纬度 (°)')
        axes[0, 2].set_title('地面轨迹')
        axes[0, 2].legend()
        axes[0, 2].grid(True)

        # 4. 展长变形量-时间
        axes[1, 0].plot(time_m_real, eta_m_real, 'b-', linewidth=2, label='变形飞行器')
        axes[1, 0].plot(time_t_real, eta_t_real, 'r--', linewidth=2, label='传统固定外形飞行器')
        axes[1, 0].axhline(y=1, color='k', linestyle=':', alpha=0.7, label='基准线(η=1)')
        axes[1, 0].set_xlabel('时间 (s)')
        axes[1, 0].set_ylabel('展长变形量')
        axes[1, 0].set_title('展长变形量-时间曲线')
        axes[1, 0].legend()
        axes[1, 0].grid(True)

        # 5. 攻角-时间
        axes[1, 1].plot(time_m_real, gongjiao_m_real, 'b-', linewidth=2, label='变形飞行器')
        axes[1, 1].plot(time_t_real, gongjiao_t_real, 'r--', linewidth=2, label='传统固定外形飞行器')
        axes[1, 1].set_xlabel('时间 (s)')
        axes[1, 1].set_ylabel('攻角 (°)')
        axes[1, 1].set_title('攻角-时间曲线')
        axes[1, 1].legend()
        axes[1, 1].grid(True)

        # 6. 倾侧角-时间
        axes[1, 2].plot(time_m_real, qingce_m_real, 'b-', linewidth=2, label='变形飞行器')
        axes[1, 2].plot(time_t_real, qingce_t_real, 'r--', linewidth=2, label='传统固定外形飞行器')
        axes[1, 2].set_xlabel('时间 (s)')
        axes[1, 2].set_ylabel('倾侧角 (°)')
        axes[1, 2].set_title('倾侧角-时间曲线')
        axes[1, 2].legend()
        axes[1, 2].grid(True)

        # 7. 航迹角-时间
        axes[2, 0].plot(time_m_real, theta_m_real, 'b-', linewidth=2, label='变形飞行器')
        axes[2, 0].plot(time_t_real, theta_t_real, 'r--', linewidth=2, label='传统固定外形飞行器')
        axes[2, 0].set_xlabel('时间 (s)')
        axes[2, 0].set_ylabel('航迹角 (°)')
        axes[2, 0].set_title('航迹角-时间曲线')
        axes[2, 0].legend()
        axes[2, 0].grid(True)

        # 8. 航向角-时间
        axes[2, 1].plot(time_m_real, psi_m_real, 'b-', linewidth=2, label='变形飞行器')
        axes[2, 1].plot(time_t_real, psi_t_real, 'r--', linewidth=2, label='传统固定外形飞行器')
        axes[2, 1].set_xlabel('时间 (s)')
        axes[2, 1].set_ylabel('航向角 (°)')
        axes[2, 1].set_title('航向角-时间曲线')
        axes[2, 1].legend()
        axes[2, 1].grid(True)

        # 9. 高度-速度
        axes[2, 2].plot(velocity_m_real, height_m_real, 'b-', linewidth=2, label='变形飞行器')
        axes[2, 2].plot(velocity_t_real, height_t_real, 'r--', linewidth=2, label='传统固定外形飞行器')
        axes[2, 2].set_xlabel('速度 (m/s)')
        axes[2, 2].set_ylabel('高度 (km)')
        axes[2, 2].set_title('高度-速度曲线')
        axes[2, 2].legend()
        axes[2, 2].grid(True)

        plt.tight_layout()
        plt.suptitle('变形飞行器与传统固定外形飞行器性能对比', fontsize=16, y=0.98)
        plt.subplots_adjust(top=0.94)
        plt.show()


if __name__ == "__main__":
    sim = main_simulation()
    (time_morphing, states_morphing, controls_morphing,
     time_traditional, states_traditional, controls_traditional) = sim.main()