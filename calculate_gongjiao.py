import math

class GetGongjiao:
    def __init__(self):
        self.V1 = 6500
        self.V2 = 5000
        self.alpha_max = 20  # 最大攻角，假设值
        self.alpha_LD = 10  # 低速攻角，假设值
        self.g0 = 9.8
        self.R0 = 6371e3
        self.V_scale = math.sqrt(self.g0 * self.R0)

    def get_gongjiao(self, v):
        """
        根据给定的速度V计算攻角
        :param V: 当前无量纲速度
        :return: 攻角
        """

        V = v * self.V_scale


        # 计算攻角
        if V > self.V1:
            alpha = self.alpha_max
        elif V <= self.V1 and V > self.V2:
            alpha = (self.alpha_LD - self.alpha_max) / (self.V2 - self.V1) * (V - self.V1) + self.alpha_max
        else:
            alpha = self.alpha_LD

        return alpha
