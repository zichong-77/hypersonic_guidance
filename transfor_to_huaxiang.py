import numpy as np
import math
from get_canshu import CanShu


class transfor_to_huaxiang():
    def __init__(self):
        # 地球参数
        self.mu = 3.986004418e14  # 地球引力常数 (m³/s²)
        self.R0 = 6.371e6  # 地球半径 (m)
        self.g0 = 9.80665  # 重力加速度 (m/s²)
        self.R0 = 6371e3

        # 转换判断阈值（按照论文要求设置）
        self.min_altitude = 30000.0  # 最小转换高度 (m) - 论文中提到滑翔段在较低高度
        self.max_altitude = 85000.0  # 最大转换高度 (m)
        self.min_velocity = 2000.0  # 最小速度 (m/s) - 接近目标速度时开始滑翔

        self.canshu = CanShu()

    def transfor_to_huaxiang(self, state, eta, gongjiao, qingce_rad):
        r = state[0]
        v = state[3]
        h = r - 1
        theta = state[4]



        R = r * self.R0
        V = v * math.sqrt(self.R0 * self.g0)
        altitude = R - self.R0

        L,D = self.canshu.calculate_L_D (v,h,eta,gongjiao)


        # 转换条件判断（按照论文条件）

        # 条件1: 高度范围检查
        if altitude < self.min_altitude or altitude > self.max_altitude:
            return 0


        # 准平衡滑翔条件：L*cos(σ) + (V²-1/r)*cos(θ)/r ≥ 0

        QEGC_condition = L * math.cos(qingce_rad) + (v * v - 1 / r) * math.cos(theta) / r

        if QEGC_condition >= 0:
            return 1
        else:
            return 0

