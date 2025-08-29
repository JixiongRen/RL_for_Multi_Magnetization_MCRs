# 多磁化磁控导管强化学习项目

这个项目实现了一个基于SOFA框架的多磁化磁控导管(MCR, Magnetic Continuum Robot)仿真环境，并提供了强化学习接口以训练智能控制器。该系统可用于模拟医疗场景中磁控导管在血管内的导航过程。

## 项目结构

### 主要模块

- **multi_magnetization_mcr_env.py**: 强化学习环境的主要实现，继承自`SofaEnv`，定义了观测空间、动作空间和奖励函数。
- **human_controller.py**: 人类操作界面，通过键盘输入控制导管运动。
- **vessel_sim_scene_2d.py**: 2D平面血管场景构建脚本。
- **vessel_sim_scene_3d.py**: 3D主动脉血管场景构建脚本。

### 核心功能模块 (multi_magnetization_mcr_sim)

- **multi_magnetization_mcr_catheter.py**: 定义导管物理模型和几何结构。
- **multi_magnetization_mcr_magnet.py**: 实现磁体的物理特性和参数。
- **multi_magnetization_mcr_controller_sofa.py**: 与SOFA框架交互的控制器实现。
- **multi_magnetization_mcr_enviroment.py**: 血管环境的物理建模。
- **multi_magnetization_mcr_mag_controller.py**: 磁场控制器，处理磁场和磁体之间的相互作用。
- **multi_magnetization_mcr_simulator.py**: SOFA仿真环境的基础设置。
- **multi_magnetization_mcr_supiee.py**: 电磁导航系统接口。

### 资源目录

- **meshes/**: 包含血管模型的STL文件。
    - **anatomies/**: 3D解剖学血管模型
    - **flat_models/**: 2D平面血管模型
- **calib/**: 电磁系统校准文件。

## 磁控导管结构

多磁化磁控导管由以下部分组成：

1. **主体段(刚性段)**：长度较长，较硬，用于传递推力。
    - 默认长度: 1.0 m
    - 默认杨氏模量: 170e6 Pa

2. **柔性段(尖端段)**：长度较短，较软，含有多个磁体。
    - 默认长度: 0.05 m
    - 默认杨氏模量: 21e6 Pa
    - 默认包含3个磁体，磁体布局可配置

3. **磁体配置**：
    - 默认磁体长度: 5mm
    - 默认外径: 1.33mm
    - 默认内径: 0.86mm
    - 默认剩磁: 1.45 T

导管通过SOFA的梁模型进行离散化表示，支持物理仿真和实时形变计算。

## 自定义配置

### 导管物理参数

在场景描述文件(`vessel_sim_scene_2d.py`或`vessel_sim_scene_3d.py`)中，可以修改以下参数：

```python
# 导管器械参数
young_modulus_body = 170e6  # 主体段杨氏模量 (Pa)
young_modulus_tip = 21e6    # 尖端段杨氏模量 (Pa)
length_body = 1.0           # 主体段长度 (m)
length_tip = 0.05           # 尖端段长度 (m)
outer_diam = 0.00133        # 外径 (m)
inner_diam = 0.0008         # 内径 (m)
```

### 磁体配置

可以自定义磁体的物理参数和布局：

```python
# 磁体参数
magnet_length = 5e-3       # 磁体长度 (m)
magnet_id = 0.86e-3        # 磁体内径 (m)
magnet_od = 1.33e-3        # 磁体外径 (m)
magnet_remanence = 1.45    # 剩磁 (Tesla)

# 磁体布局 - 以物理间距定义
magnets_layout = [
    (0.015, magnet),  # (间隔距离, 磁体对象)
    (0.015, magnet),
    (0.005, magnet),
]
```

### 环境和场景参数

可以修改血管模型和初始位置：

```python
# 环境模型路径
environment_stl = str(HERE / "meshes/flat_models/flat_model_circles.stl")

# 器械在环境坐标系下的入口位姿
T_start_env = [-0.04, 0.01, 0.002, 0.0, 0.0, 0.0, 1.0]  # [x, y, z, qx, qy, qz, qw]
```

## 运行项目

### 人工控制

通过键盘控制导管：

```bash
python human_controller.py --env_type FLAT  # 使用平面场景
# 或
python human_controller.py --env_type AORTIC  # 使用3D主动脉场景
```

键盘控制映射：
- W/S: 控制导管插入/回撤
- A/D: 控制磁场绕X轴旋转
- Q/E: 控制磁场绕Y轴旋转
- ESC: 退出控制

可选参数：
- `--record_video`: 记录视频
- `--record_trajectory`: 记录轨迹数据
- `--info "额外信息"`: 在元数据中添加信息

### 强化学习环境

要将环境用于强化学习，可以导入环境并进行自定义：

```python
from multi_magnetization_mcr_env import MultiMagnetizaitonMCREnv, EnvType, ObervationType

# 创建环境
env = MultiMagnetizaitonMCREnv(
    image_shape=(400, 400),
    observation_type=ObervationType.STATE,
    env_type=EnvType.FLAT,
    render_mode=RenderMode.HEADLESS,
    reward_amount_dict={
        "tip_pos_distance_to_dest_pos": -1.0,
        "delta_tip_pos_distance_to_dest_pos": -10.0,
        "workspace_constraint_violation": -5.0,
        "successful_task": 100.0,
    }
)

# 使用环境
obs, info = env.reset()
while True:
    action = agent.get_action(obs)  # 由强化学习算法提供
    obs, reward, terminated, truncated, info = env.step(action)
    if terminated or truncated:
        break
```

## 强化学习配置

### 观测空间

环境支持两种类型的观测：

1. **STATE**观测(默认)：
    - 导管尖端位置(3)和四元数(4)
    - 导管上的跟踪点位置(num_catheter_tracking_points * 3)
    - 磁场向量(3)
    - 目标位置(3)

2. **RGB**观测：
    - 渲染的图像(高度 × 宽度 × 3)

### 动作空间

环境使用连续动作空间，维度为3：
- 第一维：绕Z轴旋转磁场(-1.0到1.0)
- 第二维：绕X轴旋转磁场(-1.0到1.0)
- 第三维：导管插入/回撤(-1.0到1.0)

### 奖励函数

奖励由以下组件组成，可通过`reward_amount_dict`进行配置：

- **tip_pos_distance_to_dest_pos**: 导管尖端到目标的距离
- **delta_tip_pos_distance_to_dest_pos**: 距离变化量
- **workspace_constraint_violation**: 工作空间约束违反惩罚
- **successful_task**: 成功完成任务的奖励

### 自定义奖励权重

示例：
```python
reward_amount_dict = {
    "tip_pos_distance_to_dest_pos": -1.0,      # 距离越近越好
    "delta_tip_pos_distance_to_dest_pos": -5.0, # 鼓励减小距离
    "workspace_constraint_violation": -10.0,    # 严重惩罚违反工作空间约束
    "successful_task": 100.0,                   # 任务完成给予大量奖励
}
```

## 深度强化学习模型集成

要将此环境与深度强化学习框架(如Stable Baselines3或RLlib)集成，可参考以下步骤：

### 使用Stable Baselines3

```python
from multi_magnetization_mcr_env import MultiMagnetizaitonMCREnv, EnvType, ObervationType
from stable_baselines3 import SAC, PPO

# 创建环境
env = MultiMagnetizaitonMCREnv(
    observation_type=ObervationType.STATE,
    env_type=EnvType.FLAT,
)

# 创建模型
model = SAC("MlpPolicy", env, verbose=1)
# 训练模型
model.learn(total_timesteps=100000)

# 保存模型
model.save("mcr_sac_model")

# 加载和评估
model = SAC.load("mcr_sac_model")
obs, _ = env.reset()
while True:
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = env.step(action)
    if terminated or truncated:
        break
```

### 自定义网络架构

对于此类连续控制任务，推荐以下网络架构：

1. **Actor网络**：
    - 输入层: 观测维度
    - 隐藏层: 256-256个神经元，使用ReLU激活
    - 输出层: 3个神经元(动作维度)，使用tanh激活

2. **Critic网络**：
    - 输入层: 观测维度 + 动作维度
    - 隐藏层: 256-256个神经元，使用ReLU激活
    - 输出层: 1个神经元(Q值)，线性激活

TD3或SAC算法通常对这类连续控制任务表现良好。

## 注意事项

1. 系统对计算资源有较高要求，特别是3D主动脉场景。
2. 默认情况下，成功任务的判定阈值为导管尖端与目标位置的欧氏距离小于0.015m。
3. 可通过`target_position`参数自定义目标位置。

## 依赖库

- SOFA框架与sofa_env
- NumPy
- SciPy
- PyMag-Manip (磁场计算)
- Gymnasium
- OpenCV
- Pynput (键盘控制)