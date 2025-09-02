import math
import numpy as np
from get_canshu import CanShu
from PredictTerminalState import PredictTerminalState
from calculate_angle import get_angle


class houxu_control():
    def __init__(self):
        self.R0 = 6371e3  # 地球半径 (m)
        self.g0 = 9.8  # 重力加速度 (m/s^2)
        self.V_scale = math.sqrt(self.g0 * self.R0)  # 速度无量纲化系数
        self.T_scale = math.sqrt(self.R0 / self.g0)
        self.initial_h = 80e3 / self.R0

        self.target_v = 2000 / self.V_scale
        self.target_h = 35e3 / self.R0
        self.target_r = self.target_h + 1

        self.canshu = CanShu()
        self.PredictTerminalState = PredictTerminalState()
        self.get_angle = get_angle()

    def houxu_control(self, state, eta, gongjiao, qingce_rad, r_point, v_point, simulation_time=1500):
        try:
            # 使用牛顿迭代法求解新的控制量
            new_eta, new_gongjiao, new_qingce_rad = self.PredictTerminalState.newton_iteration_solve(
                state, eta, gongjiao, qingce_rad)

            # 根据论文公式(22)，添加高度反馈
            simulation_time = simulation_time / self.T_scale
            h_dot = (self.target_h - self.initial_h) / simulation_time
            r = state[0]
            v = state[3]
            h = r - 1
            theta = state[4]
            h_dot_ref = v * math.sin(theta)

            # 高度反馈系数，根据论文公式(23)
            k0 = 40
            kf = 10

            e0 = 1 / r_point - v_point * v_point / 2
            ef = 1 / self.target_r - self.target_v * self.target_v / 2
            em = 0.99 * ef
            e = 1 / r - v * v / 2

            # 计算反馈系数k
            if (e >= e0) and (e <= em):
                k = k0 + ((e0 - e) / (e0 - em)) * (kf - k0)
            elif e > em:
                k = 0
            else:
                k = k0

            # 获取升力和阻力
            L, D = self.canshu.calculate_L_D(v, h, eta, gongjiao)

            # 确保L不为零，避免除零错误
            if abs(L) < 1e-10:
                L = 1e-10

            # 计算高度反馈后的倾侧角
            cos_qingce_before = math.cos(new_qingce_rad)
            cos_qingce_feedback = cos_qingce_before - k * (h_dot - h_dot_ref) / L

            # 限制cos值在[-1, 1]范围内，避免math domain error
            cos_qingce_feedback = max(-1.0, min(1.0, cos_qingce_feedback))
            qingce_rad_act = math.acos(cos_qingce_feedback)

            # 滤波参数
            a = 0.1
            b = 0.1

            # 对高度反馈后的结果进行滤波
            qingce_magnitude = abs(qingce_rad_act)
            eta_filtered = a * new_eta + (1 - a) * eta
            qingce_filtered = b * qingce_magnitude + (1 - b) * abs(qingce_rad)

            new_eta = eta_filtered
            new_qingce_rad = qingce_filtered

            # 确定倾侧角符号
            sign = self.get_angle.get_sign(state, qingce_rad)
            new_qingce_rad = sign * new_qingce_rad

            # 限制控制量在合理范围内
            new_eta = max(1.0, min(3.0, new_eta))
            new_gongjiao = max(0.0, min(20.0, new_gongjiao))
            new_qingce_rad = max(-math.pi / 2, min(math.pi / 2, new_qingce_rad))

            return new_eta, new_gongjiao, new_qingce_rad

        except Exception as e:
            # 如果出现错误，返回当前值
            print(f"控制量计算警告: {e}")
            return eta, gongjiao, qingce_rad