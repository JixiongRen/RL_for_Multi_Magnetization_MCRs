import Sofa
import Sofa.Core
import numpy as np

from multi_magnetization_mcr_sim import multi_magnetization_mcr_mag_controller, \
    multi_magnetization_mcr_supiee, multi_magnetization_mcr_catheter

from scipy.spatial.transform import Rotation as R

# 定义: 使用弧度制表示的磁场角增量
DFIELD_ANGLE = 3.0 * np.pi / 180.0

class ControllerSofa(Sofa.Core.Controller):
    """
    SOFA环境中多磁化MCR(磁性连续体机器人)的控制器类。

    该控制器负责处理键盘事件输入，控制磁场方向和导管的插入/回撤操作。
    它连接SOFA仿真环境与磁场控制器，实现对磁性导管的实时控制。

    :param root_node: SOFA仿真的根节点 (Sofa.Core.Node)
    :param supiee: SUPIEE磁场发生器实例 (multi_magnetization_mcr_supiee.SUPIEE)
    :param catheter: 多磁化MCR导管实例 (multi_magnetization_mcr_catheter.MultiMagnetizationMCRCatheter)
    :param T_sim_mns: 仿真到磁导航系统(MNS)的变换矩阵 (list[float])
    :param mag_field_init: 初始磁场向量 (np.ndarray)
    :param dfield_angle: 磁场角度变化量 (float)
    :param mag_controller: 磁场控制器实例 (multi_magnetization_mcr_mag_controller.MagController)
    :param invalid_action: 标记当前动作是否有效 (bool)
    """
    def __init__(
            self,
            root_node: Sofa.Core.Node,
            supiee: multi_magnetization_mcr_supiee.SUPIEE,
            catheter: multi_magnetization_mcr_catheter.MultiMagnetizationMCRCatheter,
            T_sim_mns: list[float],
            mag_field_init: np.ndarray=np.array([0.01, 0.01, 0.0]),
            *args,
            **kwargs,
    ) -> None:
        Sofa.Core.Controller.__init__(self, *args, **kwargs)

        self.root_node = root_node
        self.supiee = supiee
        self.catheter = catheter
        self.T_sim_mns = T_sim_mns
        self.mag_field_init = mag_field_init

        self.dfield_angle = 0.0

        self.mag_controller = multi_magnetization_mcr_mag_controller.MagController(
            root_node=self.root_node,
            supiee=self.supiee,
            catheter=self.catheter,
            T_sim_mns=self.T_sim_mns,
        )
        self.root_node.addObject(self.mag_controller)

        self.mag_controller.field_des = self.mag_field_init
        self.invalid_action = False


    def onKeypressedEvent(self, event):
        """
        当按键被按下时处理键盘输入事件。

        实现了以下键盘映射:
        - J键(76): 绕Z轴正向旋转磁场
        - L键(74): 绕Z轴负向旋转磁场
        - I键(73): 绕X轴正向旋转磁场
        - K键(75): 绕X轴负向旋转磁场

        :param event: 包含按键信息的事件字典
        """
        key = event["key"]
        # J 键：z 轴旋转 +
        if ord(key) == 76:
            self.rotateZ(-1)

        # L 键：z 轴旋转 -
        if ord(key) == 74:
            self.rotateZ(1)

        # I 键：x 轴旋转 +
        if ord(key) == 73:
            self.rotateX(-1)

        # K 键：x 轴旋转 -
        if ord(key) == 75:
            self.rotateX(1)


    def rotateZ(self, val: int) -> None:
        """
        围绕Z轴旋转磁场。

        :param val: 旋转方向系数，+1表示顺时针，-1表示逆时针
        """
        rot_vec = val * DFIELD_ANGLE * np.array([0.0, 0.0, 1.0])
        r = R.from_rotvec(rot_vec)
        self.mag_controller.field_des = r.apply(self.mag_controller.field_des)
        print(f"Rotated magnetic field: {self.mag_controller.field_des}")


    def rotateX(self, val: int) -> None:
        """
        围绕X轴旋转磁场。

        :param val: 旋转方向系数，+1表示顺时针，-1表示逆时针
        """
        rot_vec = val * DFIELD_ANGLE * np.array([1.0, 0.0, 0.0])
        r = R.from_rotvec(rot_vec)
        self.mag_controller.field_des = r.apply(self.mag_controller.field_des)
        print(f"Rotated magnetic field: {self.mag_controller.field_des}")


    def insertRetract(self, val: float):
        """
        控制导管的插入或回撤操作。

        :param val: 插入/回撤的方向和幅度，正值表示插入，负值表示回撤

        说明:
            - 每步移动距离为0.0015单位
            - 如果操作会使导管尖端超出安全范围(>1.01)，将标记为无效操作
        """
        d_step = 0.0015
        irc_xtip = self._getXTipValue()
        if (float(irc_xtip) + (val * d_step)) > 1.01:
            self.invalid_action = True
        else:
            self.invalid_action = False
            self.catheter.IRC.xtip[0] += val * d_step


    def _getXTipValue(self):
        """
        获取导管尖端当前的插入/回撤位置值。

        :return 导管尖端的当前位置值
        """
        return self.catheter.IRC. xtip[0]


    def reset(self) -> None:
        """
        重置控制器状态，包括磁场和导管位置

        重置操作会:
        - 将磁场恢复到初始值
        - 重置磁场角度变化量
        - 将导管回撤到初始位置
        - 清除无效操作标记
        """
        super().reset()
        self.mag_controller.field_des = self.mag_field_init
        self.dfield_angle = 0.0

        # 重置插入/回撤状态
        self.catheter.IRC.xtip[0] = 0.0
        self.invalid_action = False


    def get_mag_field_des(self):
        """
        获取当前设定的期望磁场向量
        :return numpy.ndarray: 三维磁场向量
        """
        return self.mag_controller.field_des


    def get_pos_catheter(self, num_points):
        """
        获取导管在指定数量采样点处的位置。采样点均匀分布在导管上，通过插值方式选择
        :param num_points: 需要获取的采样点数量
        :return numpy.ndarray: 包含所有采样点位置的数组
        """
        pos_catheter = ()
        factor = int(31 / num_points) + 1
        for i in range(num_points):
            pos_catheter = np.append(pos_catheter, self.catheter.MO.position.array()[i * factor][:3])
        return pos_catheter


    def get_pos_quat_catheter_tip(self):
        """
        获取导管尖端的位置和姿态四元数。
        :return 导管尖端的位置和姿态信息
        """
        return self.catheter.MO.position.array()[32]
