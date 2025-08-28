import Sofa
import numpy as np

from multi_magnetization_mcr_sim import multi_magnetization_mcr_supiee, multi_magnetization_mcr_catheter
from splib.numerics import Quat, Vec3
from scipy.spatial.transform import Rotation as R

class MagController(Sofa.Core.Controller):
    """
    用于接收期望的磁场输入并计算施加在磁性器械磁体上的力矩的类
    这些力矩会在每个时间步施加到 SOFA 的机械模型上

    :param supiee: 磁驱动器对象 (SUPIEE)
    :param catheter: 多点磁控导管对象 (MultiMagnetizationMCRCatheter)
    :param T_sim_mns: 定义 sofa_sim 坐标系中心在 Supiee 坐标系中的位姿变换 [x, y, z, qx, qy, qz, qw] (list[float])
    :param field_des: 期望的磁场 (Tesla) (np.ndarray)
    """
    def __init__(
            self,
            supiee: multi_magnetization_mcr_supiee.SUPIEE,
            catheter: multi_magnetization_mcr_catheter.MultiMagnetizationMCRCatheter,
            T_sim_mns: list[float],
            field_des: np.ndarray = np.array([0.0, 0.0, 0.0]),
            *args,
            **kwargs,
    ) -> None:
        Sofa.Core.Controller.__init__(self, *args, **kwargs)

        self.supiee = supiee
        self.catheter = catheter
        self.T_sim_mns = T_sim_mns
        self.field_des = field_des

        # 读取有效的磁体磁矩: 从第一个非零掩码项获取
        # 假设嵌入导管的磁体具有相同的磁矩参数
        self.magnet_moment = 0.
        if hasattr(catheter, "magnets"):
            for m in catheter.magnets:
                if m not in (0, None):  # 若未定义或为零则 fallback为 0(不会产生扭矩)
                    self.magnet_moment = getattr(m, "dipole_moment", 0.0)
                    break

        self.BG = [0., 0., 0., 0., 0., 0., 0., 0., 0.]  # 初始化 Bx, By, Bz, dBx/dx, dBx/dy, dBx/dz, dBy/dy, dBy/dz

        # 获取节点数量和磁体数量
        self.num_nodes = len(self.catheter.MO.position)
        self.num_magnets = len(self.catheter.index_mag)

        # 入口点位置
        self.initPos = np.array([self.T_sim_mns[0], self.T_sim_mns[1], self.T_sim_mns[2]])

    def onAnimateBeginEvent(self, event) -> None:
        """
        计算并施加磁场力矩到导管上的每一个磁体节点
        :param event: SOFA 动画事件
        :return: None
        """
        # 获取最新的节点数量和磁体数量
        self.num_nodes = len(self.catheter.MO.position)
        self.num_magnets = len(self.catheter.index_mag)

        magnetic_field = None

        for i in range(self.num_magnets):
            # 获取磁体在柔性段中的索引
            magnet_index = self.catheter.index_mag[i]

            # 计算该磁体在 MO.position 数组中的实际位置索引
            actual_index = self.num_nodes - self.catheter.num_elem_tip + magnet_index  # TODO: 检查此处逻辑

            if actual_index >= 0 and actual_index < self.num_nodes:
                pos = self.catheter.MO.position[actual_index]
                quat = Quat(pos[3], pos[4], pos[5], pos[6])

                # 更新磁体在 Supiee 坐标系中的位置
                actualPos = np.array([pos[0], pos[1], pos[2]]) + np.array([self.T_sim_mns[0], self.T_sim_mns[1], self.T_sim_mns[2]])

                # field_des -> currents_des
                currents = self.supiee.field_to_currents(field=self.field_des, position=actualPos)

                self.BG = self.supiee.currents_to_field(currents=currents, position=actualPos)  # 更新 BG

                # 计算磁力矩: 将 B 向量按照磁力矩缩放
                B = Vec3(self.BG[0] * self.magnet_moment, self.BG[1] * self.magnet_moment, self.BG[2] * self.magnet_moment)
                magnetic_field = B

                r = R.from_quat(quat)
                X = r.apply(np.array([1.0, 0.0, 0.0]))
                T = Vec3()  # [0.0, 0.0, 0.0]
                T = T.cross(X, magnetic_field)

                # 更新力矩
                # TODO: 实际上并没有考虑磁力，仅考虑了力矩
                force_index = magnet_index
                if force_index < len(self.catheter.CFF.forces):
                    self.catheter.CFF.forces[force_index][:] = [0, 0, 0, T[0], T[1], T[2]]

        # 可视化最后一个磁场
        # TODO: 如何可视化所有磁体的磁场?
        if magnetic_field is not None:
            self.catheter.CFF_visu.force = [magnetic_field[0], magnetic_field[1], magnetic_field[2], 0, 0, 0]









