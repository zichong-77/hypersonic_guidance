import numpy as np
import math
from get_canshu import CanShu
from calculate_gongjiao import GetGongjiao


class PredictTerminalState:
    def __init__(self):
        self.R0 = 6371e3  # 地球半径 (m)
        self.g0 = 9.8  # 重力加速度 (m/s^2)
        self.V_scale = math.sqrt(self.g0 * self.R0)  # 速度无量纲化系数
        self.lambda_target = np.deg2rad(57)  # 目标经度 (弧度)
        self.phi_target = np.deg2rad(15)  # 目标纬度 (弧度)
        self.v_target = 2000 / self.V_scale  # 目标速度 (无量纲)
        self.h_target = 35e3 / self.R0  # 目标高度 (无量纲)
        self.r_target = 1 + self.h_target  # 目标地心距 (无量纲)

        # 初始化参数计算类和攻角计算类
        self.canshu = CanShu()
        self.gongjiao = GetGongjiao()

    def get_ideal_remaining_range_height(self, lambda_current, phi_current):
        """
        计算理想剩余航程和理想终端高度
        """
        # 计算大圆弧距离作为理想剩余航程（使用Haversine公式）
        delta_lambda = self.lambda_target - lambda_current
        delta_phi = self.phi_target - phi_current

        # Haversine公式计算大圆弧距离（无量纲）
        a = (math.sin(delta_phi / 2) ** 2 +
             math.cos(phi_current) * math.cos(self.phi_target) *
             math.sin(delta_lambda / 2) ** 2)

        # 确保a在[0,1]范围内，避免数值误差
        a = max(0.0, min(1.0, a))
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

        # 理想剩余航程（无量纲，以地球半径为单位）
        s_ideal = c

        # 理想终端高度（无量纲）
        h_ideal = self.h_target

        return s_ideal, h_ideal

    def _integrate_trajectory(self, initial_state, eta, gongjiao, qingce_rad):
        """
        积分轨迹方程预测终端状态
        """
        try:
            # 解包状态变量
            r, lambda_val, phi, v, theta, psi = initial_state
            h = r - 1

            # 初始化积分变量
            e_initial = 1 / r - v * v / 2
            e_final = 1 / self.r_target - self.v_target * self.v_target / 2

            # 如果初始能量已经大于等于终端能量，直接返回
            if e_initial >= e_final:
                return 0.0, h

            # 积分步长
            de_step = min(0.001, (e_final - e_initial) / 100)
            e_current = e_initial

            # 当前状态
            r_current = r
            v_current = v
            theta_current = theta
            s_current = 0.0

            max_iterations = 1000
            iteration = 0

            # 积分直到达到终端能量
            while e_current < e_final and iteration < max_iterations:
                iteration += 1

                # 从当前能量计算速度
                # e = 1/r - V²/2, 所以 V = sqrt(2(1/r - e))
                try:
                    v_current = math.sqrt(2 * max(0, 1 / r_current - e_current))
                except:
                    v_current = 0.1  # 设置最小速度

                # 获取当前状态下的升力和阻力
                h_current = r_current - 1
                L, D = self.canshu.calculate_L_D(v_current, h_current, eta, gongjiao)

                # 避免除零
                if abs(D) < 1e-10:
                    break

                # 根据论文公式(17)计算导数
                cos_theta = math.cos(theta_current)
                sin_theta = math.sin(theta_current)
                cos_sigma = math.cos(qingce_rad)

                # 限制三角函数值
                cos_theta = max(-1.0, min(1.0, cos_theta))
                sin_theta = max(-1.0, min(1.0, sin_theta))
                cos_sigma = max(-1.0, min(1.0, cos_sigma))

                ds_de = cos_theta / (r_current * D)
                dr_de = sin_theta / D

                # 计算theta导数时要小心
                if abs(v_current) > 1e-10:
                    dtheta_de = (L * cos_sigma + (v_current * v_current - 1 / r_current) * cos_theta / r_current) / (
                                D * v_current * v_current)
                else:
                    dtheta_de = 0

                # 更新状态变量
                s_current += ds_de * de_step
                r_current += dr_de * de_step
                theta_current += dtheta_de * de_step

                # 限制状态变量在合理范围内
                r_current = max(1.001, r_current)  # 不能小于地球半径
                theta_current = max(-math.pi / 2, min(math.pi / 2, theta_current))

                # 更新能量
                e_current += de_step

                # 防止无限循环
                if abs(ds_de) < 1e-10 and abs(dr_de) < 1e-10:
                    break

            # 预测剩余航程和终端高度
            predict_sgo = abs(s_current)
            predict_hgo = r_current - 1

            return predict_sgo, predict_hgo

        except Exception as e:
            print(f"轨迹积分错误: {e}")
            # 返回默认值
            return 0.1, self.h_target

    def get_predict_remaining_range_height(self, state, eta, gongjiao, qingce_rad):
        """
        预测剩余航程和终端高度
        """
        return self._integrate_trajectory(state, eta, gongjiao, qingce_rad)

    def newton_iteration_solve(self, state, pre_eta, pre_gongjiao, pre_qingce_rad):
        """
        使用牛顿迭代法求解倾侧角和展长变形量
        """
        try:
            r = state[0]
            v = state[3]

            # 根据速度计算攻角
            new_gongjiao = self.gongjiao.get_gongjiao(v)

            # 预测值和真实值
            predict_sgo, predict_hgo = self._integrate_trajectory(state, pre_eta, pre_gongjiao, pre_qingce_rad)
            real_sgo, real_hgo = self.get_ideal_remaining_range_height(state[1], state[2])

            F = predict_sgo - real_sgo
            G = predict_hgo - real_hgo

            # 如果误差已经很小，直接返回
            if abs(F) < 0.01 and abs(G) < 0.001:
                return pre_eta, new_gongjiao, abs(pre_qingce_rad)

            # 扰动量
            delta_eta = 0.1
            delta_sigma = np.deg2rad(2)

            # 使用扰动值来计算雅可比矩阵
            predict_sgo_delta_eta, predict_hgo_delta_eta = self._integrate_trajectory(
                state, pre_eta + delta_eta, pre_gongjiao, pre_qingce_rad)
            predict_sgo_delta_sigma, predict_hgo_delta_sigma = self._integrate_trajectory(
                state, pre_eta, pre_gongjiao, pre_qingce_rad + delta_sigma)

            # 计算F和G对各个变量的偏导数
            F_delta_eta = predict_sgo_delta_eta - real_sgo
            G_delta_eta = predict_hgo_delta_eta - real_hgo
            F_delta_sigma = predict_sgo_delta_sigma - real_sgo
            G_delta_sigma = predict_hgo_delta_sigma - real_hgo

            # 计算导数
            F_d_delta_eta = (F_delta_eta - F) / delta_eta if abs(delta_eta) > 1e-10 else 0
            G_d_delta_eta = (G_delta_eta - G) / delta_eta if abs(delta_eta) > 1e-10 else 0
            F_d_delta_sigma = (F_delta_sigma - F) / delta_sigma if abs(delta_sigma) > 1e-10 else 0
            G_d_delta_sigma = (G_delta_sigma - G) / delta_sigma if abs(delta_sigma) > 1e-10 else 0

            # 构建雅可比矩阵
            jacobian_matrix = np.array([[F_d_delta_eta, F_d_delta_sigma],
                                        [G_d_delta_eta, G_d_delta_sigma]])

            # 检查矩阵条件数，避免奇异矩阵
            det = np.linalg.det(jacobian_matrix)
            if abs(det) < 1e-10:
                # 矩阵接近奇异，添加正则化
                jacobian_matrix += np.eye(2) * 1e-6

            # 求解牛顿方程
            try:
                delta = np.linalg.solve(jacobian_matrix, np.array([-F, -G]))
            except:
                # 如果求解失败，使用伪逆
                delta = np.linalg.pinv(jacobian_matrix) @ np.array([-F, -G])

            # 更新变量，添加步长限制
            step_limit_eta = 0.5
            step_limit_sigma = np.deg2rad(10)

            delta[0] = max(-step_limit_eta, min(step_limit_eta, delta[0]))
            delta[1] = max(-step_limit_sigma, min(step_limit_sigma, delta[1]))

            new_eta = pre_eta + delta[0]
            new_qingce_rad = abs(pre_qingce_rad) + delta[1]

            # 限制在合理范围内
            new_eta = max(1.0, min(3.0, new_eta))
            new_qingce_rad = max(0.0, min(math.pi / 2, new_qingce_rad))

            return new_eta, new_gongjiao, new_qingce_rad

        except Exception as e:
            print(f"牛顿迭代求解错误: {e}")
            # 返回安全值
            return max(1.0, min(3.0, pre_eta)), new_gongjiao, min(math.pi / 2, abs(pre_qingce_rad))