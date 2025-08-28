import Sofa
from scipy.spatial.transform import Rotation as R
import numpy as np

class Environment(Sofa.Core.Controller):
    """
    用于定义环境对象并构建该环境的 SOFA 碰撞与可视化模型的类。

    :param root_node: SOFA 根节点
    :type root_node:
    :param environment_stl: 环境 STL 网格文件的路径
    :type environment_stl: str
    :param name: 环境名称
    :type name: str
    :param T_env_sim: 定义环境相对于仿真坐标系的位姿变换 [x, y, z, qx, qy, qz, qw]
    :type T_env_sim:
    :param flip_normals: 控制网格法向方向的标志
    :type flip_normals: bool
    :param color: 环境可视化使用的颜色 [r, g, b, alpha]
    :type color: list[float]
    """
    def __init__(
            self,
            root_node: Sofa.Core.Node,
            environment_stl: str,
            name: str="environment",
            T_env_sim: list[float]=[0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
            filp_normals: bool=False,
            color: list[float]=[0.0, 1.0, 0.0, 0.3],
            *args,
            **kwargs,
    ) -> None:
        Sofa.Core.Controller.__init__(self, *args, **kwargs)

        self.root_node = root_node
        self.environment_stl = environment_stl
        self.name_env = name

        self.color = color

        self.T_env_sim = T_env_sim
        r = R.from_quat(self.T_env_sim[3:7])
        rot_env_sim = (r.as_euler("xyz", degrees=True)).tolist()

        # 碰撞
        self.CollisionModel = root_node.addChild("CollisionModel")
        self.CollisionModel.addObject("RequiredPlugin", name="Sofa.Component.IO.Mesh")
        self.CollisionModel.addObject(
            "MeshSTLLoader",
            filename=self.environment_stl,
            flipNormals=filp_normals,
            triangulate=True,
            name="meshLoader",
            rotation=rot_env_sim,
            translation=self.T_env_sim[0:3],
            scale="0.001",
        )

        self.CollisionModel.addObject("RequiredPlugin", name="Sofa.Component.Topology.Container.Constant")
        self.CollisionModel.addObject("Mesh", position="@meshLoader.position", triangles="@meshLoader.triangles", drawTriangles="0")
        self.CollisionModel.addObject("RequiredPlugin", name="Sofa.Component.StateContainer")
        self.MO = self.CollisionModel.addObject("MechanicalObject", position=[0, 0, 0], scale=1, name="DOFs1")
        self.CollisionModel.addObject("RequiredPlugin", name="Sofa.Component.Collision.Geometry")
        self.CollisionModel.addObject("TriangleCollisionModel", moving=False, simulated=False)
        self.CollisionModel.addObject("LineCollisionModel", moving=False, simulated=False)
        self.CollisionModel.addObject("PointCollisionModel", moving=False, simulated=False)

        # 可视化模型环境
        VisuModel = self.CollisionModel.addChild("VisuModel")
        VisuModel.addObject("RequiredPlugin", name="Sofa.GL.Component.Rendering3D")
        VisuModel.addObject("OglModel", name="VisualOgl_model", src="@../meshLoader", color=self.color)

    def get_vessel_tree_positions(self) -> np.ndarray:
        """返回血管树网格点的位置数组"""
        positions_vessel_tree = self.MO.position.array()
        return positions_vessel_tree