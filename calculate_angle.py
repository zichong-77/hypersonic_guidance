import numpy as np
import math

class get_angle():
    def __init__(self):
        self.delta_psi = np.deg2rad(10)
        self.target_jingdu = np.deg2rad(57)
        self.target_weidu = np.deg2rad(15)

    def get_angle(self, jingdu1, weidu1, jingdu2, weidu2):
        """
        计算两个地理点之间的视线角（方位角）。
        """
        try:
            # 计算经度差
            delta_jingdu = jingdu2 - jingdu1

            # 经度差归一化到[-π, π]
            while delta_jingdu > math.pi:
                delta_jingdu -= 2 * math.pi
            while delta_jingdu < -math.pi:
                delta_jingdu += 2 * math.pi

            # 使用正确的方位角公式
            y = math.sin(delta_jingdu) * math.cos(weidu2)
            x = (math.cos(weidu1) * math.sin(weidu2) -
                 math.sin(weidu1) * math.cos(weidu2) * math.cos(delta_jingdu))

            # 避免除零情况
            if abs(x) < 1e-10 and abs(y) < 1e-10:
                return 0.0

            angle_rad = math.atan2(y, x)

            # 确保角度在[-π, π]范围内
            while angle_rad > math.pi:
                angle_rad -= 2 * math.pi
            while angle_rad < -math.pi:
                angle_rad += 2 * math.pi

            return angle_rad

        except Exception as e:
            print(f"角度计算错误: {e}")
            return 0.0

    def get_sign(self, state, pre_qingce_rad):
        """
        确定倾侧角符号
        """
        try:
            # 注意：state[1]是经度lambda，state[2]是纬度phi
            psi_los = self.get_angle(state[1], state[2], self.target_jingdu, self.target_weidu)
            psi_now = state[5]

            # 计算航向角误差
            psi_error = psi_now - psi_los

            # 归一化航向角误差到[-π, π]
            while psi_error > math.pi:
                psi_error -= 2 * math.pi
            while psi_error < -math.pi:
                psi_error += 2 * math.pi

            # 根据论文公式(26)确定符号
            if psi_error > self.delta_psi:
                sign = -1
            elif psi_error < -self.delta_psi:
                sign = 1
            else:
                # 保持上一次的符号
                if abs(pre_qingce_rad) < 1e-10:
                    sign = 1  # 默认符号
                else:
                    sign = 1 if pre_qingce_rad > 0 else -1

            return sign

        except Exception as e:
            print(f"符号计算错误: {e}")
            # 返回默认符号
            return 1 if pre_qingce_rad >= 0 else -1