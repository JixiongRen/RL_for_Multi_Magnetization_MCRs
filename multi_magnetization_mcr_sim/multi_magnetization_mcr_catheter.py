import Sofa
import numpy as np
from typing import List, Optional, Tuple, Union


class MultiMagnetizationMCRCatheter(Sofa.Core.Controller):
    """
    用于创建多磁化导管的 SOFA 控制器类

    :param root_node: SOFA 根节点 (Sofa.Core.Node)
    :param magnets: 磁体列表 (list)
                        - 长度为 `num_element_tip`, 元素为 0 或 Magnet 对象，表示柔性段每个离散单元是否放置磁体
    :param name: 导管名称 (str)
    :param length_body: 导管主体(刚性段)长度 (float, m)
    :param length_tip: 导管柔性段(含磁体)长度 (float, m)
    :param outer_diam: 导管外径 (float, m)
    :param inner_diam: 导管内径 (float, m)
    :param young_modulus_body: 导管主体(刚性段)杨氏模
    :param young_modulus_tip: 导管柔性段(此处不包含磁体)杨氏模量 (float, Pa)
    :param num_elem_body: 导管主体(刚性段)单元数 (int)
    :param num_elem_tip: 导管柔性段(含磁体)单元数 (int)
    :param nume_nodes_viz: 可视化节点数 (int)
    :param T_start_sim: 仿真开始时的位姿变换 [x, y, z, qx, qy, qz, qw] (list[float])
    :param fixed_directions: 用于固定节点自由度的参数 [tx, ty, yz, rx, ry, rz] (list[int])
    :param color: 导管可视化颜色 [r, g, b, alpha] (list[float])
    :param magnets_layout: 磁体布局，包含每个磁体的位移和磁体对象的列表
                          - 每个元素为 (gap_before, magnet) 的元组
                          - gap_before: 磁体前的间距 (float, m)
                          - magnet: 磁体对象或 None
                          - 例如: [(0.002, magnet1), (0.010, magnet2), (0.014, magnet3)]
    :param `*args`: 其余位置参数传递给 SofaCoreController
    :param `**kwargs`: 关键字参数传递给 SofaCoreController
    """

    def __init__(
            self,
            root_node: Sofa.Core.Node,
            magnets: Optional[List] = None,
            name: str = "multi_magnetization_mcr_catheter",
            length_body: float = 1.0,
            length_tip: float = 0.05,
            outer_diam: float = 0.00133,
            inner_diam: float = 0.0008,
            young_modulus_body: float = 170e6,
            young_modulus_tip: float = 21e6,
            num_elem_body: int = 30,
            num_elem_tip: int = 10,
            nume_nodes_viz: int = 600,
            T_start_sim: Optional[List[float]] = None,
            fixed_directions: Optional[List[int]] = None,
            color: Optional[List[float]] = None,

            magnets_layout: Optional[List[Union[dict, tuple]]] = None,
            *args,
            **kwargs,
    ):
        Sofa.Core.Controller.__init__(self, *args, **kwargs)

        self.root_node = root_node

        # ================ 1. 关键参数缺省值设置 ================
        if T_start_sim is None: T_start_sim = [0., 0., 0., 0., 0., 0., 1.]
        if fixed_directions is None: fixed_directions = [0, 0, 0, 0, 0, 0]
        if color is None: color = [0.2, 0.8, 1.0, 1.0]

        # ================ 2. 导管参数设置 ================
        self.outer_diam = outer_diam
        self.inner_diam = inner_diam

        self.num_elem_body = num_elem_body
        self.num_elem_tip = num_elem_tip

        self.length_body = length_body
        self.length_tip = length_tip

        self.color = color

        # 空心圆截面的惯性矩：I = (π/64) * (D^4 - d^4)，D为外径，d为内径
        # 实心圆截面的惯性矩：I = (π/64) * D_eq^4，D_eq为等效外径
        # 令两者惯性矩相等，解得：D_eq = ((D^4 - d^4))^(1/4)
        self.outer_diam_eq = ((outer_diam / 2.) ** 4. - (inner_diam / 2.) ** 4.) ** (1. / 4.)

        # ================ 3. 导管柔性段磁体布局 ================
        # 1) 优先使用原接口 magnets
        # 2) 若未定义 magnets 参数或 magnets 列表长度和 num_elem_tip 不匹配，则使用 magnets_layout
        #     - a. 将柔性段按照等长单元离散化: dl = length_tip / num_elem_tip
        #     - b. 对每个磁体，取其 [起点, 终点] 区间的中心点并映射到单元索引
        #     - c. TODO: 若一个磁体跨越多个单元，可根据需要扩展: 此处采用 "中心点落在哪个单元就放在哪个单元"
        if isinstance(magnets, list) and len(magnets) == num_elem_tip:
            self.magnets = magnets
        else:
            self.magnets = self._discretize_magnets_layout(
                length_tip=length_tip,
                num_elem_tip=num_elem_tip,
                magnets_layout=magnets_layout,
            )
        # 提取柔性段内含磁体的离散索引列表
        self.index_mag = np.nonzero(self.magnets)[0]

        # ================ 4. 可视化区间 ================
        first_mag_length = None
        for m in self.magnets:
            if m != 0 and m is not None:
                first_mag_length = getattr(m, "length", None)
                break
        if first_mag_length is None:
            # fallback: 若没有磁体定义，则使用柔性段长度的一小部分作为第一个磁体长度，但理论上不应该出现此情况
            first_mag_length = max(1e-6, length_tip / max(1, num_elem_tip))

        self.index_mag_visu = [
            int(nume_nodes_viz * (length_body + length_tip - first_mag_length)),
            int(nume_nodes_viz * (length_body + length_tip)),
        ]

        self.fixed_directions = fixed_directions

        # ================ 5. SOFA 节点和组件创建 ================
        """
        这段代码创建了多磁化导管仿真中的拓扑线模型，作为导管物理模拟和形变计算的基础骨架。
        
        1. 拓扑线节点创建:
            - 首先创建一个名为 `{name}_topo_lines` 的子节点作为容器: `topoLines_guide = self.root_node.addChild(name + "_topo_lines")`
        2. 导管几何特性定义:
            - 添加 `WireRestShape` 组件，定义导管的静态几何和物理特性: `topoLines_guide.addObject("WireRestShape", ...)`
        3. 关键参数含义：
            - `straightLength`: 导管刚性段长度
            - `length`: 导管总长度（刚性段+柔性段）
            - `youngModulus`/`youngModulusExtremity`: 分别定义刚性段和柔性段的材料刚度
            - `radius`/`radiusExtremity`: 导管截面半径
            - `numEdgesCollis`/`densityOfBeams`: 控制导管离散化密度，影响模拟精度
        4.拓扑结构和物理状态组件:
            - 添加一系列组件构建拓扑结构和物理模型：
                ```python
                topoLines_guide.addObject("EdgeSetTopologyContainer", name="meshLinesGuide")
                topoLines_guide.addObject("EdgeSetTopologyModifier", name="Modifier")
                topoLines_guide.addObject("MechanicalObject", name="dofTopo2", template="Rigid3d")
                ```
        5. 功能说明：
            - `EdgeSetTopologyContainer`: 存储边集拓扑信息，定义导管的离散结构
            - `EdgeSetTopologyModifier`: 提供动态修改拓扑结构的能力
            - `MechanicalObject`: 存储导管的物理状态（位置、速度等），使用刚体模板
        
        这些组件共同构成了导管的基础物理模型，为后续的插值计算、力学模拟和形变分析提供了必要框架。
        """
        # 5.1 topoLines_guide: 用于插值的拓扑线模型
        topoLines_guide = self.root_node.addChild(name + "_topo_lines")
        topoLines_guide.addObject(
            "WireRestShape",
            name="MMMCRsRestShape",
            straightLength=length_body,
            length=length_body + length_tip,
            numEdges=nume_nodes_viz,
            youngModulus=young_modulus_body,
            spireDiameter=250.,
            numEdgesCollis=[self.num_elem_body, self.num_elem_tip],
            printLog=True,
            template="Rigid3d",
            spireHeight=0.,
            radius=self.outer_diam_eq / 2.,
            radiusExtremity=self.outer_diam_eq / 2.,
            densityOfBeams=[self.num_elem_body, self.num_elem_tip],
            youngModulusExtremity=young_modulus_tip,
        )
        topoLines_guide.addObject("RequiredPlugin", name="Sofa.Component.Topology.Container.Dynamic")
        topoLines_guide.addObject("EdgeSetTopologyContainer", name="meshLinesGuide")
        topoLines_guide.addObject("EdgeSetTopologyModifier", name="Modifier")
        topoLines_guide.addObject("EdgeSetGeometryAlgorithms", name="GeomAlgo", template="Rigid3d")
        topoLines_guide.addObject("RequiredPlugin", name="Sofa.Component.StateContainer")
        topoLines_guide.addObject("MechanicalObject", name="dofTopo2", template="Rigid3d")

        # 5.2 InstrumentCombined: 导管的力学模型和控制器
        self.InstrumentCombined = self.root_node.addChild(name)
        # ---------- ODE 求解器 ----------
        self.InstrumentCombined.addObject("RequiredPlugin", name="Sofa.Component.ODESolver.Backward")
        self.InstrumentCombined.addObject("EulerImplicitSolver", rayleighStiffness=0.2, printLog=False,
                                          rayleighMass=0.0)

        # ---------- 线性求解器 ----------
        self.InstrumentCombined.addObject("RequiredPlugin", name="Sofa.Component.LinearSolver.Direct")
        self.InstrumentCombined.addObject("BTDLinearSolver", verification=False, subpartSolve=False, verbose=False)

        # ---------- 网格拓扑 ----------
        """
        这段代码在SOFA框架中创建一个规则网格(Regular Grid)拓扑，用于定义导管的离散化模型。具体功能如下：

            1. 第168行导入了SOFA的网格拓扑容器组件，这是创建RegularGrid对象的必要依赖
    
            2. 第169~182行创建并配置了RegularGrid对象:
               - 将对象存储在`self.RG`变量中以便后续访问
               - 设置网格名称为"meshLinesCombined"
               - 配置为本质上是一维网格：
                 - X轴方向有`self.num_elem_body + self.num_elem_tip`个单元 (导管主体和柔性段的单元总和)
                 - Y轴和Z轴方向都只有1个单元 (ny=1, nz=1)
                 - X轴范围为 0到 0.2，代表导管的长度方向
                 - Y轴和Z轴实际上没有延展 (ymin=ymax=0, zmin=zmax=1)
    
            这个网格实际上创建了一个沿X轴的一维离散线模型，用于表示导管的物理结构。
            它作为后续物理模拟和形变计算的基础几何结构，将与MechanicalObject组件结合使用来模拟导管的力学行为。
        """
        self.InstrumentCombined.addObject("RequiredPlugin", name="Sofa.Component.Topology.Container.Grid")
        self.RG = \
            self.InstrumentCombined.addObject(
                "RegularGrid",
                name="meshLinesCombined",
                zmax=1,
                zmin=1,
                nx=self.num_elem_body + self.num_elem_tip,
                ny=1,
                nz=1,
                xmax=0.2,
                xmin=0,
                ymin=0,
                ymax=0)

        # ---------- 力学模型 ----------
        """
        功能：向InstrumentCombined节点添加StateContainer组件依赖
        目的：为紧随其后的MechanicalObject组件提供必要支持
        StateContainer是SOFA框架中的核心组件，它：
            - 管理模拟对象的物理状态（位置、速度、力等）
            - 提供状态数据存储结构
        是后续创建的MechanicalObject（第204行）的必要依赖：
        self.MO = self.InstrumentCombined.addObject("MechanicalObject", showIndices=False, name="DOFs", template="Rigid3d")
        """
        self.InstrumentCombined.addObject("RequiredPlugin", name="Sofa.Component.StateContainer")

        """
        这段代码创建并初始化了SOFA框架中的核心组件 - MechanicalObject，用于存储和管理导管模型的物理状态。

        1. 创建组件
            - `self.MO = \`: 将创建的组件实例存储在类属性中，方便后续访问
            - `self.InstrumentCombined.addObject()`: 在导管节点树中添加新组件
            - 参数解析:
              - `MechanicalObject`: SOFA中用于表示物理状态的基础组件类型
              - `showIndices=False`: 禁用可视化中的节点索引显示
              - `name="DOFs"`: 组件名称设为"DOFs"
              - `template="Rigid3d"`: 使用刚体模板，表示每个节点有位置和旋转属性
        2. 初始化组件
            - `self.MO.init()`: 调用初始化方法，完成以下工作:
              - 分配内存空间
              - 建立内部数据结构
              - 准备物理状态变量（位置、速度、加速度等）
              - 连接组件间的依赖关系
              
        在多磁化MCR导管模拟中，此MechanicalObject的作用为:
            1. 存储导管离散化后每个节点的位姿信息（位置和方向）
            2. 作为力和运动计算的基础数据容器
            3. 为后续的力场计算、形变分析和可视化提供状态数据
        
        后续代码中可以看到，导管的初始姿态、受力情况和约束条件都是通过这个MechanicalObject实现的。
        """
        self.MO = \
            self.InstrumentCombined.addObject("MechanicalObject", showIndices=False, name="DOFs", template="Rigid3d")
        self.MO.init()

        restPos = []
        indicesAll = []
        i = 0
        for pos in self.MO.rest_position.value:
            restPos.append(T_start_sim)
            indicesAll.append(i)
            i += 1

        forcesList = ""
        for i in range(0, self.num_elem_body + self.num_elem_tip):
            forcesList += " 0 0 0 0 0 0 "  # (fx, fy, fz, tx, ty, tz)

        indicesList = list(range(0, self.num_elem_body + self.num_elem_tip))

        self.MO.rest_position.value = restPos

        self.IC = \
            self.InstrumentCombined.addObject(
                "WireBeamInterpolation",
                WireRestShape="@../" + name + "_topo_lines" + "/MMMCRsRestShape",
                radius=self.outer_diam_eq / 2.0,
                printLog=True,
                name="InterpolGuide"
            )

        self.InstrumentCombined.addObject(
            "AdaptiveBeamForceFieldAndMass",
            massDensity=155.0,
            name="GuideForceField",
            interpolation="@InterpolGuide"
        )

        # ---------- 添加磁体力矩控制器 ----------
        self.InstrumentCombined.addObject("RequiredPlugin", name="Sofa.Component.MechanicalLoad")

        # ---------- 恒定力场 ----------
        # 常量力场作用于指定离散索引
        # indices 覆盖整根导管，indexFromEnd=True 使索引从末端计数
        self.CFF = self.InstrumentCombined.addObject("ConstantForceField", name="CFF", indices=indicesList,
                                                     forces=forcesList, indexFromEnd=True)  # 设定恒定力场
        self.CFF_visu = self.InstrumentCombined.addObject("ConstantForceField", name="CFFVisu", indices=0,
                                                          force="0 0 0 0 0 0", showArrowSize=2.0e2)  # TODO: 这里可以修改箭头大小

        # ---------- 介入放射学控制器 ----------
        self.IRC = self.InstrumentCombined.addObject(
            "InterventionalRadiologyController",
            xtip=[0.001],
            name="m_ircontroller",
            instruments="InterpolGuide",
            step=0.0007,
            printLog=True,
            listening=True,
            template="Rigid3d",
            startingPos=T_start_sim,
            rotationInstrument=[0.0],
            speed=1e-12,
            mainDirection=[0, 0, 1],
            threshold=5e-9,
            controlledInstrument=0,
        )
        self.InstrumentCombined.addObject("RequiredPlugin", name="Sofa.Component.Constraint.Lagrangian.Correction")
        self.InstrumentCombined.addObject("LinearSolverConstraintCorrection", wire_optimization="true", printLog=False)
        self.InstrumentCombined.addObject("RequiredPlugin", name="Sofa.Component.Constraint.Projective")
        self.InstrumentCombined.addObject("FixedConstraint", indices=0, name="FixedConstraint")
        self.InstrumentCombined.addObject("RequiredPlugin", name="Sofa.Component.SolidMechanics.Spring")
        self.InstrumentCombined.addObject("RestShapeSpringsForceField", points="@m_ircontroller.indexFirstNode",
                                          angularStiffness=1e8, stiffness=1e8)

        # ---------- 对节点的约束 ----------
        self.InstrumentCombined.addObject("PartialFixedConstraint", indices=indicesAll,
                                          fixedDirections=self.fixed_directions, fixAll=True)

        # ---------- 碰撞模型 ----------
        Collis = self.InstrumentCombined.addChild(name + "_collis")
        Collis.activated = True
        Collis.addObject("RequiredPlugin", name="Sofa.Component.Topology.Container.Dynamic")
        Collis.addObject("EdgeSetTopologyContainer", name="collisEdgeSet")
        Collis.addObject("EdgeSetTopologyModifier", name="colliseEdgeModifier")
        Collis.addObject("RequiredPlugin", name="Sofa.Component.StateContainer")
        Collis.addObject("MechanicalObject", name="CollisionDOFs")
        Collis.addObject("MultiAdaptiveBeamMapping", controller="../m_ircontroller", useCurvAbs=True, printLog=False,
                         name="collisMap")
        Collis.addObject("RequiredPlugin", name="Sofa.Component.Collision.Geometry")
        Collis.addObject("LineCollisionModel", proximity=0.0, group=1)
        Collis.addObject("PointCollisionModel", proximity=0.0, group=1)

        # ---------- ROS 可视化 ----------
        CathVisuROS = self.InstrumentCombined.addChild("CathVisuROS")
        CathVisuROS.addObject("RequiredPlugin", name="Sofa.Component.Topology.Container.Grid")
        CathVisuROS.addObject("RegularGrid", name="meshLinesCombined", zmax=0.0, zmin=0.0, nx=nume_nodes_viz, ny=1,
                              nz=1, xmax=1.0, xmin=0.0, ymin=0.0, ymax=0.0)
        CathVisuROS.addObject("RequiredPlugin", name="Sofa.Component.StateContainer")
        self.MO_visu = CathVisuROS.addObject("MechanicalObject", name="ROSCatheterVisu", template="Rigid3d")
        CathVisuROS.addObject("AdaptiveBeamMapping", interpolation="@../InterpolGuide", printLog="1", useCurvAbs="1")

        # ---------- SOFA 可视化 ----------
        CathVisu = self.InstrumentCombined.addChild(name + "_viz")
        CathVisu.addObject("RequiredPlugin", name="Sofa.Component.StateContainer")
        CathVisu.addObject("MechanicalObject", name="QuadsCatheter")
        CathVisu.addObject("RequiredPlugin", name="Sofa.Component.Topology.Container.Dynamic")
        CathVisu.addObject("QuadSetTopologyContainer", name="ContainerCath")
        CathVisu.addObject("QuadSetTopologyModifier", name="Modifier")
        CathVisu.addObject("QuadSetGeometryAlgorithms", name="GeomAlgo", template="Vec3d")
        self.root_node.addObject("RequiredPlugin", name="Sofa.Component.Topology.Mapping")
        CathVisu.addObject("Edge2QuadTopologicalMapping", flipNormals="true",
                           input="@../../" + name + "_topo_lines" + "/meshLinesGuide", nbPointsOnEachCircle="10",
                           output="@ContainerCath", radius=self.outer_diam_eq / 2, tags="catheter")
        CathVisu.addObject("AdaptiveBeamMapping", interpolation="@../InterpolTube0", input="@../DOFs",
                           isMechanical="false", name="VisuMapCath", output="@QuadsCatheter", printLog="1",
                           useCurvAbs="1")
        VisuOgl = CathVisu.addChild("VisuOgl")
        VisuOgl.addObject("OglModel", quads="@../ContainerCath.quads", color=self.color,
                          material="texture Ambient 1 0.2 0.2 0.2 0.0 Diffuse 1 1.0 1.0 1.0 1.0 Specular 1 1.0 1.0 1.0 1.0 Emissive 0 0.15 0.05 0.05 0.0 Shininess 1 20",
                          name="VisualCatheter")
        VisuOgl.addObject("RequiredPlugin", name="Sofa.Component.Mapping.Linear")
        VisuOgl.addObject("IdentityMapping", input="@../QuadsCatheter", output="@VisualCatheter", name="VisuCathIM")

    def _discretize_magnets_layout(
            self,
            length_tip: float,
            num_elem_tip: int,
            magnets_layout: Optional[List[Union[dict, tuple]]]
    ) -> Optional[list[int]]:
        """
            将磁体的空间物理布局转化为离散化的、长度为 `num_elem_tip` 的掩码列表
            对于边界超出的情况: 超出柔性段的磁体将被截断/跳过并打印提示
            对于全空情况的处理: 若 `magnets_layout` 为空或无效，则返回全 0 掩码列表
            :param length_tip: 柔性段弧长，规定从 0 延伸到 length_tip
            :param num_elem_tip: 柔性段离散化单元数
            :param magnets_layout: 导管尖端磁体的物理布局, 可选用以下格式：
                - list[dict]: 每个字典包含 'gap_before' 和 'magnet' 键
                - list[tuple]: 每个元组为 (gap_before, magnet)
                其中
                    - gap_before: 此磁体与上一个磁体之间的纯柔性部分间距 (float, m)
                    - magnet: 磁体对象 (Magnet)
            :return: 长度为 `num_elem_tip` 的掩码列表
            """

        # 初始化/默认全 0 掩码列表
        mask = [0 for _ in range(num_elem_tip)]
        if not magnets_layout: return mask

        dl = float(length_tip) / float(num_elem_tip)  # 每个单元的长度
        s_cursor = 0.

        def _parse_item(item: Union[dict, tuple]) -> Tuple[float, Optional['Magnet']]:
            if isinstance(item, dict):
                gap = float(item.get("gap_before", item.get("gap", 0.0)))
                magnet = item.get("magnet", None)
                return gap, magnet
            elif isinstance(item, (tuple, list)) and len(item) >= 2:
                return float(item[0]), item[1]
            else:
                return 0., None

        for idx_item, it in enumerate(magnets_layout):
            gap, mag = _parse_item(it)
            if mag is None:
                print(f"[Instrument] magnets_layout item #{idx_item} have no invalid magnet, skipping.")
                continue
            s_start = s_cursor + max(0., gap)
            s_end = s_start + float(getattr(mag, "length", 0.))

            if s_start >= length_tip:
                print(
                    f"[Instrument] magnets_layout item #{idx_item} starts at {s_start:.3f} m, which is beyond the tip length {length_tip:.3f} m, skipping.")
                break  # 超出柔性段长度, 终止

            s_end_clamped = min(s_end, length_tip)  # 确保结束位置不超过柔性段长度
            c = 0.5 * (s_start + s_end_clamped)  # 计算中心点位置: 处于磁体首末端中心点位置

            # 将中心点映射到单元索引 (0 ... num_elem_tip-1)
            index = int(np.clip(np.floor(c / dl + 1e-9), 0, num_elem_tip - 1))
            mask[index] = mag  # 将磁体对象放入对应单元位置

            s_cursor = s_end  # 更新游标位置, 从磁体末端开始下一个循环
        return mask
