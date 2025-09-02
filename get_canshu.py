import numpy as np
import math


class CanShu:
    def __init__(self):
        self.R = 287.053  # 干空气比气体常数 (J/(kg·K))
        self.gamma = 1.4  # 干空气比热比

        # 物理参数
        self.m = 996.4  # 质量 (kg)
        self.S = 0.54 * 8  # 参考面积 (m^2)
        self.k_Q = 9.4369e-5
        self.R0 = 6371e3
        self.g0 = 9.8

    def new_get_CL_CD(self, eta, alpha, Ma):
        """
        计算升力系数和阻力系数
        """
        try:
            # 确保输入为float类型并进行数值范围检查
            eta = float(np.clip(eta, 1.0, 3.0))  # 限制eta范围
            alpha = float(np.clip(alpha, 0, 20))  # 限制攻角范围
            Ma = float(np.clip(Ma, 0.1, 30))  # 限制马赫数范围

            # 阻力系数参数
            p00D = 0.07815
            p10D = 0.0002955
            p01D = 0.001565
            p11D = -0.0003823
            p02D = 0.00083

            # 计算阻力系数
            CD = p00D + p10D * Ma + p01D * alpha + p11D * Ma * alpha + p02D * alpha ** 2

            # 升力系数参数
            p00L = 0.05875
            p10L = -0.006656
            p01L = 0.04488

            # 计算升力系数
            CL = p00L + p10L * Ma + p01L * alpha

            # 伸缩率修正
            k_cl = 0.025 * eta * 2 + 0.125 * eta + 0.85
            k_cd = 0.05 * eta + 1

            CL = 1.4 * k_cl * CL
            CD = 2.0 * k_cd * CD

            # 确保系数在合理范围内
            CL = max(0.001, min(CL, 5.0))
            CD = max(0.001, min(CD, 5.0))

            return float(CL), float(CD)

        except Exception as e:
            print(f"气动系数计算错误: {e}")
            return 0.5, 0.1  # 返回默认值

    def get_atmosphere_params(self, h):
        """
        计算给定高度的大气属性
        """
        try:
            H = h * self.R0
            H = float(np.clip(H, 0, 200000))  # 限制高度0-200km

            # 简化的分层大气模型
            if H <= 11000:  # 对流层
                T0 = 288.15
                p0 = 101325
                L = 0.0065

                T = T0 - L * H
                T = max(T, 150.0)  # 防止温度过低
                p = p0 * (T / T0) ** (9.81 / (self.R * L))

            elif H <= 20000:  # 平流层下部
                T = 216.65
                p = 22632 * math.exp(-9.81 * (H - 11000) / (self.R * T))

            elif H <= 32000:  # 平流层上部
                T0 = 216.65
                p0 = 5474.9
                L = 0.001
                h0 = 20000

                T = T0 + L * (H - h0)
                T = max(T, 150.0)
                p = p0 * (T / T0) ** (-9.81 / (self.R * L))

            else:  # 更高层
                T = max(200.0, 270.65 - 0.002 * (H - 32000))
                p = max(1e-6, 868.02 * math.exp(-9.81 * (H - 32000) / (self.R * T)))

            # 计算密度和声速
            rho = max(p / (self.R * T), 1e-10)
            shengsu = max(math.sqrt(self.gamma * self.R * T), 100.0)

            return float(rho), float(shengsu)

        except Exception as e:
            print(f"大气参数计算错误: {e}")
            return 1e-6, 300.0  # 返回默认值

    def calculate_L_D(self, v, h, eta, alpha_deg):
        """
        计算升力加速度和阻力加速度
        """
        try:
            # 恢复有量纲速度和高度
            V = v * math.sqrt(self.R0 * self.g0)
            V = float(np.clip(V, 100, 12000))
            H = h * self.R0
            H = float(np.clip(H, 0, 200000))
            eta = float(np.clip(eta, 1.0, 3.0))
            alpha_deg = float(np.clip(alpha_deg, 0, 20))

            # 获取大气参数
            rho, shengsu = self.get_atmosphere_params(h)

            # 计算马赫数
            Ma = V / shengsu if shengsu > 0 else 0.1

            # 计算升力系数和阻力系数
            CL, CD = self.new_get_CL_CD(eta, alpha_deg, Ma)

            # 计算动压
            q = 0.5 * rho * (V ** 2)

            # 计算升力加速度和阻力加速度
            L = q * self.S * CL / self.m  # 升力加速度
            D = q * self.S * CD / self.m  # 阻力加速度

            # 无量纲化
            L = L / self.g0
            D = D / self.g0

            # 确保结果在合理范围内
            L = max(0.001, min(L, 10.0))
            D = max(0.001, min(D, 10.0))

            return float(L), float(D)

        except Exception as e:
            print(f"升力阻力计算错误: {e}")
            return 0.1, 0.05  # 返回默认值