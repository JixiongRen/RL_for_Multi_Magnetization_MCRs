import numpy as np

class Magnet:
    """
    一个用于构建磁环对象的类

    :param length: 磁体长度 (m) (float)
    :param outer_diam: 磁体外径 (m) (float)
    :param inner_diam: 磁体内径 (m) (float)
    :param remanence: 磁体剩磁 (Tesla) (float)
    :param color: 用于可视化的磁体颜色 [r, g, b, alpha] (list[float])
    """
    def __init__(self,
                 length: float,
                 outer_diam:float,
                 inner_diam: float,
                 remanence: float,
                 color=None) -> None:
        if color is None:
            color = [0.2, 0.2, 0.2, 1.0]

        # 永磁体磁环几何参数
        self.length = length            # 磁体长度 (m)
        self.outer_diam = outer_diam    # 磁体外径 (m)
        self.inner_diam = inner_diam    # 磁体内径 (m)
        self.color = color

        # 磁体体积
        self.volume = self.length * np.pi * ((self.outer_diam / 2.0) ** 2 - (self.inner_diam / 2.0) ** 2)

        # 磁体物理参数
        self.remanence = remanence          # 剩磁 (T)
        self.mu_0 = (4.0 * np.pi) * 1e-7    # 真空磁导率 (H/m)
        self.dipole_moment = (1.0 / self.mu_0) * self.remanence * self.volume  # 磁偶极矩 (A*m^2), m = V * B / mu_0