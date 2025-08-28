import Sofa


class Simulator(Sofa.Core.Controller):
    """
    用于定义SOFA仿真中物理场和求解器的类

    :param root_node: SOFA 根节点 (Sofa.Core.Node)
    :param dt: 仿真时间步长 (float)
    :param gravity: 重力向量 [gx, gy, gz] (list[float])
    :param friction_coef: 摩擦系数 (float)
    """
    def __init__(
            self,
            root_node: Sofa.Core.Node,
            dt: float=0.01,
            gravity: list[float]=[0., 0., 0.],
            friction_coef: float=0.04,
            *args,
            **kwargs,
    ) -> None:
        Sofa.Core.Controller.__init__(self, *args, **kwargs)

        self.root_node = root_node
        self.dt = dt
        self.gravity = gravity
        self.friction_coef = friction_coef

        self.root_node.addObject("RequiredPlugin", name="ImportSoftRob", pluginName="SoftRobots")
        self.root_node.addObject("RequiredPlugin", name="ImportBeamAdapt", pluginName="BeamAdapter")
        self.root_node.addObject("RequiredPlugin", name="ImportSofaPython3", pluginName="SofaPython3")

        self.root_node.dt = self.dt
        self.root_node.animate = True
        self.root_node.gravity = self.gravity

        self.root_node.addObject("RequiredPlugin", name="Sofa.Component.Visual")
        self.root_node.addObject(
            "VisualStyle",
            displayFlags="showVisualModels hideBehaviorModels \
                hideCollisionModels hideMappings hideForceFields \
                    hideInteractionForceFields",
        )
        self.root_node.addObject("RequiredPlugin", name="Sofa.Component.AnimationLoop")
        self.root_node.addObject("FreeMotionAnimationLoop")
        self.root_node.addObject("RequiredPlugin", name="Sofa.Component.Constraint.Lagrangian.Solver")
        self.lcp_solver = self.root_node.addObject("LCPConstraintSolver", mu=str(friction_coef), tolerance="1e-6", maxIt="10000", build_lcp="false")
        self.root_node.addObject("RequiredPlugin", name="Sofa.Component.Collision.Response.Contact")
        self.root_node.addObject("RequiredPlugin", name="Sofa.Component.Collision.Detection.Algorithm")
        self.root_node.addObject("RequiredPlugin", name="Sofa.Component.Collision.Detection.Intersection")
        self.root_node.addObject("CollisionPipeline", draw="0", depth="6", verbose="1") \

        self.root_node.addObject("BruteForceBroadPhase", name="N2_1")
        self.root_node.addObject("BVHNarrowPhase", name="N2_2")
        self.root_node.addObject("LocalMinDistance", contactDistance="0.002", alarmDistance="0.003", name="localmindistance", angleCone="0.02")
        self.root_node.addObject("CollisionResponse", name="Response", response="FrictionContactConstraint")
        self.root_node.addObject("RequiredPlugin", name="SofaMiscCollision")
        self.root_node.addObject("DefaultCollisionGroupManager", name="Group")
        self.root_node.addObject("DefaultVisualManagerLoop", name="VisualLoop")

        # 设置背景色
        self.root_node.addObject("RequiredPlugin", name="Sofa.Component.Setting")
        self.root_node.addObject("BackgroundSetting", color="1 1 1")