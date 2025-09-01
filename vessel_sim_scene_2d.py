"""
vessel_sim_scene_2d — 构建基于 SOFA 的 2D 平面血管场景。

功能概述:
- 加载平面血管 STL 模型 (meshes/flat_models/flat_model_circles.stl)。
- 配置相机与光照以进行离屏渲染或交互调试。
- 基于梁模型构建带三段永磁体的导管, 并设定初始位姿。
- 连接 SUPIEE 与 SOFA 控制器, 返回场景关键对象句柄。

主要依赖:
- SOFA 与 sofa_env (Camera)。
- multi_magnetization_mcr_sim.* 子模块 (环境、导管、控制器、磁体、仿真器、SUPIEE)。
- 本仓库 calib 与 meshes 资源。

坐标系约定:
- 环境到仿真: 由 T_env_sim 指定 (平移+四元数)。
- 器械入口位姿: 由 T_start_env 转换为 T_start_sim 并用于放置导管。
- 磁导航(mns)与仿真(sim)默认重合 (T_sim_mns 为单位变换)。
"""


from splib.numerics import Quat, Vec3
import math
from scipy.spatial.transform import Rotation as R
from typing import Optional, Tuple
from pathlib import Path
import numpy as np

import Sofa.Core

from multi_magnetization_mcr_sim import \
    multi_magnetization_mcr_enviroment, \
    multi_magnetization_mcr_catheter, \
    multi_magnetization_mcr_supiee, \
    multi_magnetization_mcr_simulator, \
    multi_magnetization_mcr_controller_sofa, \
    multi_magnetization_mcr_magnet

from sofa_env.sofa_templates.camera import Camera, CAMERA_PLUGIN_LIST

from multi_magnetization_mcr_env import FLAT_CATHETER_DESTINATION_EXIT_POINT


HERE = Path(__file__).resolve().parent
PLUGIN_LIST = [
    "SofaPython3",
    "SoftRobots",
    "BeamAdapter",
] + CAMERA_PLUGIN_LIST


def createScene(
        root_node: Sofa.Core.Node,
        image_shape: Tuple[Optional[int], Optional[int]] = (None, None),
        debug_rendering: bool = False,
        positioning_camera: bool = False,
):
    """
    创建并装配 2D 血管仿真场景。

    :param root_node: SOFA 根节点, 场景会在该节点下创建所有对象。
    :param image_shape: (宽, 高) 像素, 用于相机离屏渲染尺寸;为 (None, None) 时使用默认。
    :param debug_rendering: 若为 True, 显示相机实体等调试可视化。
    :param positioning_camera: 若为 True, 将 Camera 节点加入图中以便交互定位。

    :return dict, 包含以下键:
      - "multi_magnetization_mcr_controller_sofa": 控制器对象。
      - "multi_magnetization_mcr_environment": 环境对象。
      - "camera": 相机对象。

    说明:
    - 环境 STL 通过 T_env_sim 放置到仿真坐标系中。
    - 器械入口位姿由 T_start_env 变换得到 T_start_sim 后用于初始化导管。
    - 导管采用主体段+柔性段梁模型 (num_elem_tip=10), magnets_layout 以物理间距配置三段永磁体。
    """

    # =============== 校准文件路径 ================
    cal_path = str(HERE / "calib/Navion_2_Calibration_24-02-2020.yaml")

    # =============== 导管器械参数 ===============
    young_modulus_body = 170e6  # (Pa)
    young_modulus_tip = 21e6  # (Pa)
    length_body = 1.  # (m)
    length_tip = 0.05  # (m)
    outer_diam = 0.00133  # (m)
    inner_diam = 0.0008  # (m)
    length_init = 0.35  # (m)

    # =============== 环境模型路径 ===============
    environment_stl = str(HERE / "meshes/flat_models/flat_model_circles.stl")

    # =============== 磁体参数 ===============
    magnet_length = 5e-3  # (m)
    magnet_id = 0.86e-3  # (m)
    magnet_od = 1.33e-3  # (m)
    magnet_remanence = 1.45 # (Tesla)

    # =============== SOFA 梁模型参数 ===============
    nume_nodes_viz = 600    # 可视化节点数
    num_elem_body = 30      # 导管主体段(刚性段)单元数
    num_elem_tip = 10       # 导管尖端段(柔性段)单元数

    # =============== 坐标变换参数 ===============
    """
    下面结合项目内各文件, 说明该段变换代码中三者的坐标系与相对位姿关系。

    全局与局部坐标系
    - SOFA 仿真全局坐标系(sim): 场景根节点所在的世界坐标系, `scene_description_2d.py` 中最终生成的所有对象都放置在此坐标系下。
    - 环境坐标系(env): 血管模型(场景几何与障碍物)天然定义所在的坐标系, 通过 `T_env_sim` 放置到 sim。
    - 磁导航/控制坐标系(mns): 代码中给出 `T_sim_mns` 为单位变换, 表示 mns 与 sim 对齐(默认重合)。
    
    变换与位姿构造
    - 环境到仿真:  
      `T_env_sim = [transl_env_sim, quat_env_sim]`。位置用 `transl_env_sim`, 方向由欧拉角 `rot_env_sim` 构造 `quat_env_sim`。本项目默认设为全零, 因而环境坐标系与仿真全局坐标系重合。
    - 器械(导管)初始位姿的坐标系切换:  
      在环境坐标系中给定入口位姿 `T_start_env = [p_env, q_env]`, 然后按下式转换到仿真坐标系：
      - 位置: p_sim = R_env_sim · p_env + t_env_sim  
      - 姿态: q_sim = q_env_sim ⊗ q_env  
      代码中用 `R.from_euler("xyz", rot_env_sim)` 旋转位置, 用 `Quat.createFromEuler(rot_env_sim)` 与 `q.rotateFromQuat(qrot)` 组合姿态, 语义即“先施加环境→仿真的旋转, 再叠加器械在环境中的初始旋转”。在默认零变换时, p_sim = p_env, q_sim = q_env。
    - mns 与仿真:  
      `T_sim_mns` 为单位四元数与零平移, 表示 mns 坐标与 sim 坐标一致, 便于在控制与物理场计算时无需再做坐标切换。
    
    这些位姿在场景中的落地
    - 血管模型: 在 `scene_description_2d.py` 中, 血管/环境几何按环境坐标定义, 再由 `T_env_sim` 放入仿真全局坐标系。默认零变换时, 血管原点与仿真原点一致。
    - 器械初始位姿: 由上式得到的 `T_start_sim` 提供给器械构造。可在 `mcr_sim/mcr_instrument.py` 中看到, `InterventionalRadiologyController` 的 `startingPos` 即使用该 `T_start_sim`。
    - 环境封装: 在 `mcr_env.py` 的 `MCREnv._init_sim()` 中, 场景加载后拿到控制器与环境对象, 器械和血管最终都在 sim 坐标系下实例化。
    
    默认数值下的直观关系
    - `rot_env_sim = 0`、`transl_env_sim = 0` 时, env ≡ sim, mns ≡ sim。
    - 器械初始位置即 `[-0.04, 0.01, 0.002]` m, 相对于血管模型与仿真原点的偏移一致;初始朝向为单位四元数(在代码默认下与 env→sim 的旋转相同为零), 因此器械与血管坐标轴对齐, 从该入口位姿插入场景。
    
    文件关联
    - 位姿构造于 `scene_description_2d.py`, 并传入器械;  
    - 器械使用位置与姿态见 `mcr_sim/mcr_instrument.py`(`startingPos=T_start_sim`);  
    - 环境和控制封装见 `mcr_env.py`(初始化后获取 `mcr_controller_sofa`、`mcr_environment` 并运行)。
    """

    # 环境(血管模型)在仿真坐标系下的变换
    rot_env_sim = [0, 0, 0]  # Roll-Pitch-Yaw 角
    transl_env_sim = [0, 0, 0]  # 平移

    quat_env_sim = Quat.createFromEuler([rot_env_sim[0] * np.pi / 180, rot_env_sim[1] * np.pi / 180, rot_env_sim[2] * np.pi / 180])
    T_env_sim = [transl_env_sim[0], transl_env_sim[1], transl_env_sim[2],               # 平移
                 quat_env_sim[0], quat_env_sim[1], quat_env_sim[2], quat_env_sim[3]]    # 四元数

    # 器械(导管)在环境坐标系下的入口位姿
    T_start_env = [-0.04, 0.01, 0.002, 0.0, 0.0, 0.0, 1.0]  # 定义器械在环境坐标系下的入口位姿 [x, y, z, qx, qy, qz, qw]
    X = Vec3(T_start_env[0], T_start_env[1], T_start_env[2])  # 位置
    r = R.from_euler("xyz", rot_env_sim, degrees=True)  # 定义从环境坐标系到仿真坐标系的旋转
    X = r.apply(X)  # 位置变换
    q = Quat.createFromEuler([rot_env_sim[0] * math.pi / 180, rot_env_sim[1] * math.pi / 180, rot_env_sim[2] * math.pi / 180])  # 环境到仿真的旋转四元数
    qrot = Quat(T_start_env[3], T_start_env[4], T_start_env[5], T_start_env[6])  # 器械在环境中的初始旋转四元数
    q.rotateFromQuat(qrot)  # 姿态变换, 得到器械在仿真坐标系下的初始旋转四元数

    # 器械在仿真坐标系下的入口位姿
    T_start_sim = [X[0] + transl_env_sim[0], X[1] + transl_env_sim[1], X[2] + transl_env_sim[2],  # 平移
                   q[0], q[1], q[2], q[3]]  # 四元数

    # supiee 在仿真坐标系下的变换
    T_sim_supiee = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1]  # supiee 坐标系与仿真坐标系重合

    # =============== 相机和光照参数 ===============
    root_node.addObject("RequiredPlugin", name="Sofa.GL.Component.Shader")
    root_node.addObject("RequiredPlugin", name="Sofa.Component.Visual")
    root_node.addObject("RequiredPlugin", name="Sofa.Component.ODESolver.Backward")
    root_node.addObject("RequiredPlugin", name="Sofa.Component.LinearSolver.Iterative")
    root_node.addObject("RequiredPlugin", name="Sofa.Component.Constraint.Projective")

    root_node.addObject("LightManager", listening=True, ambient=(0.8, 0.8, 0.8, 0.8))
    root_node.addObject("DirectionalLight", direction=(1, 1, 0), color=(0.8, 0.8, 0.8, 1.0))
    root_node.addObject("DirectionalLight", direction=(-1, 1, 0), color=(0.8, 0.8, 0.8, 1.0))

    placement_kwargs = {
        "position": [-0.0, 0.0, 1.1],
        "lookAt": [0.0, -0.001, 0.0],
        "orientation": [0.0, 0.0, 0.0, 0.0],
    }

    light_source_kwargs = {
        "cutoff": 45.0 / 1.2,
        "color": [0.8] * 4,
        "attenuation": 0.0,
        "exponent": 1.0,
        "shadowsEnabled": False,
    }
    camera = Camera(
        root_node=root_node,
        placement_kwargs=placement_kwargs,
        with_light_source=True,
        show_object=debug_rendering,
        show_object_scale=1.0,
        light_source_kwargs=light_source_kwargs,
        vertical_field_of_view=12,
        width_viewport=image_shape[0],
        height_viewport=image_shape[1],
        z_near=1.0,
        z_far=1000.0,
    )

    if positioning_camera:
        root_node.addObject(camera)

    # =============== 场景对象构造 ===============
    multi_magnetization_mcr_simulator.Simulator(root_node=root_node)

    # =============== Supiee 对象 ===============
    supiee = multi_magnetization_mcr_supiee.SUPIEE(
        name="Supiee",
        calibration_path=cal_path,
    )

    # =============== 仿真物理环境对象 ===============
    environment = multi_magnetization_mcr_enviroment.Environment(
        root_node=root_node,
        environment_stl=environment_stl,
        T_env_sim=T_env_sim,
        color=[1.0, 0.0, 0.0, 0.3],
    )

    # =============== 磁体对象 ===============
    magnet = multi_magnetization_mcr_magnet.Magnet(
        length=magnet_length,
        outer_diam=magnet_od,
        inner_diam=magnet_id,
        remanence=magnet_remanence,
        color=[0.2, 0.2, 0.2, 1.0],
    )

    # =============== 器械(导管)对象 ===============
    # 使用物理布局: 柔性段 0.05m、num_elem_tip=10; 每个单元约 5mm
    # 下面的三段设置使三个磁体中心分别落在三个单元中:
    # gap_before 依次为 15mm; 15mm; 5mm, 磁体长度 5mm
    magnets_layout = [
        (0.015, magnet),  # 柔性1 -> 磁体1
        (0.015, magnet),  # 柔性2 -> 磁体2
        (0.005, magnet),  # 柔性3 -> 磁体3
    ]

    catheter = multi_magnetization_mcr_catheter.MultiMagnetizationMCRCatheter(
        name="mag_gw",
        root_node=root_node,
        length_body=length_body,
        length_tip=length_tip,
        outer_diam=outer_diam,
        inner_diam=inner_diam,
        young_modulus_body=young_modulus_body,
        young_modulus_tip=young_modulus_tip,
        magnets_layout=magnets_layout,  # 切换为物理布局
        num_elem_body=num_elem_body,
        num_elem_tip=num_elem_tip,
        nume_nodes_viz=nume_nodes_viz,
        T_start_sim=T_start_sim,
        fixed_directions=[0, 0, 1, 0, 0, 0],
        color=[0.2, 0.8, 1.0, 1.0],
    )

    # =============== 控制器对象 ===============
    controller_sofa = multi_magnetization_mcr_controller_sofa.ControllerSofa(
        root_node=root_node,
        supiee=supiee,
        catheter=catheter,
        environment=environment,
        length_init=length_init,
        T_sim_mns=T_sim_supiee,
    )
    root_node.addObject(controller_sofa)

    # =============== 可视化标记小球 ===============
    # 起点标记小球（绿色）
    start_marker = root_node.addChild("start_marker")
    start_marker.addObject('MechanicalObject', 
                          name="mstate", 
                          template="Rigid3", 
                          position=[T_start_sim[0], T_start_sim[1], T_start_sim[2], 0, 0, 0, 1],
                          showObject=True,
                          showObjectScale=0.01,  # 显示为可见的球体
                          drawMode=1)  # 使用球体绘制模式
    
    # 终点标记小球（红色）
    # 使用2D场景的默认目标位置
    target_position = FLAT_CATHETER_DESTINATION_EXIT_POINT
    end_marker = root_node.addChild("end_marker")
    end_marker.addObject('MechanicalObject', 
                        name="mstate", 
                        template="Rigid3", 
                        position=[target_position[0], target_position[1], target_position[2], 0, 0, 0, 1],
                        showObject=True,
                        showObjectScale=0.01,  # 显示为可见的球体
                        drawMode=1)  # 使用球体绘制模式

    # ================ 返回数据 ===============
    scene_creation_result = {
        "multi_magnetization_mcr_controller_sofa": controller_sofa,
        "multi_magnetization_mcr_environment": environment,
        "camera": camera,
    }
    return scene_creation_result

