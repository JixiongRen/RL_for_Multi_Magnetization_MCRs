"""
vessel_sim_scene_3d — 构建基于 SOFA 的 3D 血管场景。

功能概述:
- 加载 3D 血管解剖 STL 模型 (meshes/anatomies/J2-Naviworks.stl)。
- 配置相机与光照，支持定位调试与渲染。
- 基于梁模型构建带三段永磁体的导管，并以给定入口位姿放置。
- 连接 SUPIEE、控制器与仿真器，返回关键对象句柄。

主要依赖:
- SOFA 与 sofa_env (Camera)。
- multi_magnetization_mcr_sim.* 子模块 (环境、导管、控制器、磁体、仿真器、SUPIEE)。
- 本仓库 calib 与 meshes 资源。

坐标系约定:
- 环境到仿真: 由 T_env_sim 指定 (平移+四元数)。本脚本默认让血管沿 X 轴约 90° 旋转并在 Y 上平移，以获得合适的 3D 视角与插入方向。
- 器械入口位姿: 在环境坐标系下给定 T_start_env，经 R.from_quat(rot_env_sim) 与四元数组合转换为 T_start_sim。
- 磁导航(mns)与仿真(sim)默认重合 (T_sim_mns 为单位变换)。
"""

from multi_magnetization_mcr_sim import multi_magnetization_mcr_supiee
from splib.numerics import Quat, Vec3
from scipy.spatial.transform import Rotation as R
from typing import Optional, Tuple
from pathlib import Path

import Sofa.Core
from multi_magnetization_mcr_sim import \
    multi_magnetization_mcr_enviroment, \
    multi_magnetization_mcr_catheter, \
    multi_magnetization_mcr_supiee, \
    multi_magnetization_mcr_simulator, \
    multi_magnetization_mcr_controller_sofa, \
    multi_magnetization_mcr_magnet

from sofa_env.sofa_templates.camera import Camera, CAMERA_PLUGIN_LIST

HERE = Path(__file__).resolve().parent
PLUGIN_LIST = [
    "SofaPython3",
    "SoftRobots",
    "BeamAdapter"
] + CAMERA_PLUGIN_LIST


def createScene(
        root_node: Sofa.Core.Node,
        image_shape: Tuple[Optional[int], Optional[int]]=(None, None),
        debug_rendering: bool=False,
        positioning_camera: bool=False,
):
    """
    创建并装配 3D 血管仿真场景。

    :param root_node: SOFA 根节点，场景会在该节点下创建所有对象。
    :param image_shape: (宽, 高) 像素，用于相机离屏渲染尺寸；为 (None, None) 时使用默认。
    :param debug_rendering: 若为 True，显示相机实体等调试可视化。
    :param positioning_camera: 若为 True，将 Camera 节点加入图中以便交互定位。

    :return dict，包含以下键:
      - "multi_magnetization_mcr_controller_sofa": 控制器对象。
      - "multi_magnetization_mcr_environment": 环境对象。
      - "camera": 相机对象。

    说明:
    - 环境 STL 通过 T_env_sim (四元数+平移) 放置到仿真坐标系中，必要时可通过 filp_normals 翻转法线以获得正确可视化。
    - 器械入口位姿由 T_start_env 结合 rot_env_sim 的四元数变换得到 T_start_sim，用于初始化导管。
    - 导管采用主体段+柔性段梁模型 (num_elem_tip=10)，magnets_layout 以物理间距配置三段永磁体。
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
    environment_stl = str(HERE / "meshes/anatomies/J2-Naviworks.stl")

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
    # 环境 (血管模型) 在仿真坐标系下的变换
    rot_env_sim = [-0.7071068, 0, 0, 0.7071068]
    transl_env_sim = [0.0, -0.45, 0.0]
    T_env_sim = [transl_env_sim[0], transl_env_sim[1], transl_env_sim[2],
                 rot_env_sim[0], rot_env_sim[1], rot_env_sim[2], rot_env_sim[3]]

    # 器械(导管)在环境坐标系下的入口位姿
    T_start_env = [-0.075, -0.001, -0.020, 0.0, -0.3826834, 0.0, 0.9238795]
    X = Vec3(T_start_env[0], T_start_env[1], T_start_env[2])
    r = R.from_quat(rot_env_sim)
    X = r.apply(X)
    q = Quat(rot_env_sim)
    qrot = Quat(T_start_env[3], T_start_env[4], T_start_env[5], T_start_env[6])
    q.rotateFromQuat(qrot)

    # 器械在仿真坐标系下的入口位姿
    T_start_sim = [X[0] + transl_env_sim[0], X[1] + transl_env_sim[1], X[2] + transl_env_sim[2],
                   q[0], q[1], q[2], q[3]]

    # supiee 在仿真坐标系下的变换
    T_sim_supiee = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1]

    # =============== 相机和光照参数 ===============
    root_node.addObject("RequiredPlugin", name="Sofa.GL.Component.Shader")
    root_node.addObject("RequiredPlugin", name="Sofa.Component.Visual")

    root_node.addObject(
        "LightManager",
        listening=True,
        ambient=(
            0.8,
            0.8,
            0.8,
            0.8,
        ),
    )

    root_node.addObject("DirectionalLight", direction=(1, 1, 0), color=(0.8, 0.8, 0.8, 1.0))
    root_node.addObject("DirectionalLight", direction=(-1, 1, 0), color=(0.8, 0.8, 0.8, 1.0))

    placement_kwargs = {"position": [-0.35, -1.0, -1.0], "lookAt": [0.0, -0.3, 0.0], "orientation": [0.0, 0.0, 0.0, 0.0]}

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
        vertical_field_of_view=17,
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
    supiee = multi_magnetization_mcr_supiee.SUPIEE(name="Supiee", calibration_path=cal_path)

    # =============== 仿真物理环境对象 ===============
    environment = multi_magnetization_mcr_enviroment.Environment(
        root_node=root_node,
        environment_stl=environment_stl,
        name="aortic_arch",
        T_env_sim=T_env_sim,
        filp_normals=True,
        color=[1.0, 0.0, 0.0, 0.3]
    )

    # =============== 磁体对象 ===============
    magnet = multi_magnetization_mcr_magnet.Magnet(
        length=magnet_length,
        outer_diam=outer_diam,
        inner_diam=inner_diam,
        remanence=magnet_remanence
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
        name="mcr",
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
        color=[0.2, 0.8, 1.0, 1.0],
    )

    # =============== 控制器对象 ===============
    controller_sofa = multi_magnetization_mcr_controller_sofa.ControllerSofa(
        root_node=root_node,
        supiee=supiee,
        catheter=catheter,
        length_init=length_init,
        T_sim_mns=T_sim_supiee,
    )
    root_node.addObject(controller_sofa)

    # ================ 返回数据 ===============
    scene_creation_result = {
        "multi_magnetization_mcr_controller_sofa": controller_sofa,
        "multi_magnetization_mcr_environment": environment,
        "camera": camera,
    }
    return scene_creation_result

