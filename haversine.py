import numpy as np
import math

class Haversine:
    def __init__(self):
        self.R0 = 6371e3  # 地球半径，单位：米

    def haversine(self, jingdu1, weidu1, jingdu2, weidu2):
        """
        计算两点间的大圆距离（Haversine公式）

        :param jingdu1: 第一个点的经度（弧度）
        :param weidu1: 第一个点的纬度（弧度）
        :param jingdu2: 第二个点的经度（弧度）
        :param weidu2: 第二个点的纬度（弧度）
        :return: 两点间真实距离（无量纲，单位：地球半径倍数）
        """
        # 经度和纬度的差值
        delta_jingdu = jingdu2 - jingdu1
        delta_weidu = weidu2 - weidu1

        # Haversine公式
        a = math.sin(delta_weidu / 2) ** 2 + math.cos(weidu1) * math.cos(weidu2) * math.sin(delta_jingdu / 2) ** 2
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

        # 计算无量纲距离
        distance = c  # 距离单位为地球半径倍数

        return distance
