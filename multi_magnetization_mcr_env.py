from typing import Union, Tuple, Optional, Any, Dict
from pathlib import Path
from enum import Enum, unique
import gymnasium.spaces as spaces
import numpy as np
from collections import defaultdict

from numpy import floating
from sofa_env.base import SofaEnv, RenderMode, RenderFramework
from multi_magnetization_mcr_sim.multi_magnetization_mcr_controller_sofa import ControllerSofa

HERE = Path(__file__).resolve().parent              # 当前文件所在目录
FLAT_SCENE_DESCRIPTION_FILE_PATH = \
    HERE / "vessel_sim_scene_2d.py"                # 平面场景描述文件路径
AORTIC_SCENE_DESCRIPTION_FILE_PATH = \
    HERE / "vessel_sim_scene_3d.py"                # 主动脉场景描述文件路径
FLAT_CATHETER_DESTINATION_EXIT_POINT = \
    np.array([0.101129, 0.0238015, 0.002])          # 平面导管目标出口点
AORTIC_CATHETER_DESTINATION_EXIT_POINT = \
    np.array([-0.0101583, -0.180636, 0.0345185])    # 主动脉导管目标出口点


@unique
class ObervationType(Enum):
    RGB = 0
    STATE = 1


@unique
class ActionType(Enum):
    DISCRETE = 0
    CONTINUOUS = 1


@unique
class EnvType(Enum):
    FLAT = 0
    AORTIC = 1


class MultiMagnetizaitonMCREnv(SofaEnv):
    """
    磁性连续体机器人环境(MCREnv)

    该环境的目标是旋转并移动导管，使其沿动脉前进至指定目的地。可通过 create_scene_kwargs 调整工作空间大小；详见 scene_description.py。提供两种场景(平面或主动脉)，可通过参数 env_type 选择。
    
    参数：
        image_shape (Tuple[int, int])：渲染图像的高度与宽度。
        create_scene_kwargs (Optional[dict])：传递给 createScene 函数的额外关键字参数字典。
        observation_type (ObservationType)：返回 RGB 图像或状态数组作为观测。
        action_type (ActionType)：离散或连续动作，用于定义环境的动作空间。
        time_step (float)：仿真时间步长(秒)(默认：0.1)。
        frame_skip (int)：每次 step 调用执行的仿真步数(调用 \_do_action 并推进仿真)(默认：1)。
        settle_steps (int)：环境重置后，在返回观测前需要模拟的步数。
        render_mode (RenderMode)：创建窗口(RenderMode.HUMAN)、无头运行(RenderMode.HEADLESS)，或不创建渲染缓冲(RenderMode.NONE)。
        reward_amount_dict (dict)：用于对奖励函数各组成部分加权的字典。
        target_position (Optional[np.ndarray])：场景中导管尖端的目标位置。
        env_type (EnvType)：使用平面(EnvType.FLAT)或主动脉(EnvType.AORTIC)场景。
        target_distance_threshold (float)：奖励函数的距离阈值(默认：0.015)。
        num_catheter_tracking_points (int)：需要跟踪的导管点数量(默认：4)。
    """
    def __init__(
            self,
            image_shape: Tuple[int, int] = (400, 400),
            create_scene_kwargs: Optional[dict]=None,
            observation_type: ObervationType=ObervationType.STATE,
            action_type: ActionType=ActionType.CONTINUOUS,
            time_step: float=0.1,
            frame_skip: int=1,
            settle_steps: int=10,
            render_mode: RenderMode=RenderMode.HUMAN,
            render_framework: RenderFramework=RenderFramework.PYGLET,
            reward_amount_dict: dict[str, float]=None,
            target_position: Optional[np.ndarray]=None,
            env_type: EnvType=EnvType.FLAT,
            target_distance_threshold: float=0.015,
            num_catheter_tracking_points: int=4,
    ) -> None:
        if reward_amount_dict is None:
            reward_amount_dict = {
                "tip_pos_distance_to_dest_pos": -0.0,
                "delta_tip_pos_distance_to_dest_pos": -0.0,
                "workspace_constraint_violation": -0.0,
                "successful_task": 0.0,
            }

        # 传递图像长宽参数
        if not isinstance(create_scene_kwargs, dict):
            create_scene_kwargs = {}
        create_scene_kwargs["image_shape"] = image_shape

        self.target_distance_threshold = target_distance_threshold
        self.num_catheter_tracking_points = num_catheter_tracking_points

        self.env_type = env_type
        if self.env_type == EnvType.FLAT:
            # 若 target_position 为空，则使用默认的目标出口点
            self.target_position = target_position if target_position is not None else FLAT_CATHETER_DESTINATION_EXIT_POINT
            self.scene_path = FLAT_SCENE_DESCRIPTION_FILE_PATH  # 场景模型文件路径
        elif self.env_type == EnvType.AORTIC:
            self.target_position = target_position if target_position is not None else AORTIC_CATHETER_DESTINATION_EXIT_POINT
            self.scene_path = AORTIC_SCENE_DESCRIPTION_FILE_PATH  # 场景模型文件路径

        super().__init__(
            scene_path=self.scene_path,
            time_step=time_step,
            frame_skip=frame_skip,
            render_mode=render_mode,
            render_framework=render_framework,
            create_scene_kwargs=create_scene_kwargs,
        )

        self.observation_type = observation_type        # 环境观测空间的观测类型: STATE or RGB
        self._settle_steps = settle_steps               # 环境重置后，执行多少步的空动作以稳定环境

        ###########################
        # 1. 设置观测空间
        ###########################
        if self.observation_type == ObervationType.STATE:
            #  - 器械尖端位姿: 位置(3,) + 四元数(4,) = (7,)
            #  - 器械上若干个跟踪点的位置: num_catheter_tracking_points * 3
            #  - 磁场 B (3,)
            #  - 目标位置 (3,)
            observations_size = 3 + 4 + self.num_catheter_tracking_points * 3 + 3 + 3
            self.observation_space = spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=(observations_size,),
                dtype=np.float32,
            )
        elif self.observation_type == ObervationType.RGB:
            self.observation_space = spaces.Box(
                low=0,
                high=255,
                shape=(*image_shape, 3),
                dtype=np.uint8,
            )

        ######################
        # 2. 设置动作空间
        ######################
        action_dimensionality = 3
        self.action_type = action_type
        if self.action_type == ActionType.CONTINUOUS:
            self.action_space = spaces.Box(
                low=-1.0,
                high=1.0,
                shape=(action_dimensionality,),
                dtype=np.float32,
            )
        else:
            raise NotImplementedError("Only continuous action space is implemented for MultiMagnetizationMCREnv.")

        ######################
        # 3. 回合级变量
        ######################
        self.reward_info = {}
        self.reward_features = {}

        # 未传入的奖励项默认权重为0.0
        self.reward_amount_dict = defaultdict(float) | reward_amount_dict


    def _get_observation(
            self,
            image_observation: Union[np.ndarray, None]
    ) -> Union[np.ndarray, dict]:
        """ 根据 ``ObservationType`` 组装正确的观测 """

        if self.observation_type == ObervationType.RGB:
            return image_observation
        elif self.observation_type == ObervationType.STATE:
            obs = {
                # 尖端柔性段位置 + 四元数
                "position-quaternion-catheter-tip": self.multi_magnetization_mcr_controller_sofa.get_pos_quat_catheter_tip(),
                # 采样点位置
                "position-catheter": self.multi_magnetization_mcr_controller_sofa.get_pos_catheter(num_points=self.num_catheter_tracking_points),
                # 期望磁场
                "magnetic-field-des": self.multi_magnetization_mcr_controller_sofa.get_mag_field_des(),
                # 目标位置
                "target-position": self.target_position,
            }
            return np.concatenate(tuple(obs.values()))
        else:
            return {}


    def _get_reward_features(
            self,
            previous_reward_features: dict
    ) -> dict:
        """
        计算可用于组装奖励函数的特征, 包括:
            - successful_task (bool): 任务是否完成
            - tip_pos_distance_to_dest_pos (float): 导管尖端到目标点的距离
            - delta_tip_pos_distance_to_dest_pos (float): 该距离的变化量
            - workspace_constraint_violation (bool): 是否违反工作空间约束
        :param previous_reward_features: 上一步的奖励特征字典
        :return: 当前步骤的奖励特征字典
        """
        # 初始化奖励特征字典
        reward_features = {}

        # 1. 判断任务是否完成
        if previous_reward_features["tip_pos_distance_to_dest_pos"] < self.target_distance_threshold:
            reward_features["successful_task"] = True
        else:
            reward_features["successful_task"] = False

        # 2. 导管尖端到目标点的距离
        reward_features["tip_pos_distance_to_dest_pos"] = self._get_distance_tip_to_dest()

        # 3. 距离变化量
        reward_features["delta_tip_pos_distance_to_dest_pos"] = \
            reward_features["tip_pos_distance_to_dest_pos"] - previous_reward_features["tip_pos_distance_to_dest_pos"]

        # 4. 尖端脱离工作空间惩罚项
        reward_features["workspace_constraint_violation"] = self.multi_magnetization_mcr_controller_sofa.invalid_action

        return reward_features


    def _get_reward(self) -> float:
        """获取奖励特征并根据 ``reward_amount_dict`` 进行缩放加权 """
        reward = 0.
        self.reward_info = {}
        reward_features = self._get_reward_features(previous_reward_features=self.reward_features)
        self.reward_features = reward_features.copy()

        for key, value in reward_features.items():
            value = self.reward_amount_dict[key] * value
            if "distance" in key:
                value *= self.cartesian_scaling_factor
            self.reward_info[f"reward_{key}"] = value
            reward += self.reward_info[f"reward_{key}"]

        self.reward_info["reward"] = reward
        return float(reward)


    def _get_done(self) -> bool:
        """ 判断回合是否结束 """
        return self.reward_features["successful_task"]


    def _get_info(self) -> dict:
        """ 组装info字典 """
        self.info = {}
        self.episode_info = {}
        return {**self.info, **self.reward_info, **self.episode_info, **self.reward_features}


    def _get_distance_tip_to_dest(self) -> floating[Any]:
        """ 获取器械尖端到目标位置的距离 """
        pos_tip = self.multi_magnetization_mcr_controller_sofa.get_pos_quat_catheter_tip()[:3]
        distance = np.linalg.norm(pos_tip - self.target_position)
        return distance


    def _do_action(
            self,
            action: np.ndarray
    ) -> None:
        """ 将动作应用到仿真 """
        self.multi_magnetization_mcr_controller_sofa.rotateZ(action[0])
        self.multi_magnetization_mcr_controller_sofa.rotateX(action[1])
        self.multi_magnetization_mcr_controller_sofa.insertRetract(action[2])


    def _init_sim(self) -> None:
        """ 初始化仿真 """
        super()._init_sim()
        print("keys: ", sorted(self.scene_creation_result.keys()))
        self.multi_magnetization_mcr_controller_sofa: ControllerSofa = self.scene_creation_result["multi_magnetization_mcr_controller_sofa"]
        self.multi_magnetization_mcr_environment = self.scene_creation_result["multi_magnetization_mcr_environment"]
        vessel_positions = self.multi_magnetization_mcr_environment.get_vessel_tree_positions()
        self.cartesian_scaling_factor = 1 / np.linalg.norm(np.min(vessel_positions, axis=0) - np.max(vessel_positions, axis=0))


    def reset(
            self,
            seed: Union[int, np.random.SeedSequence, None]=None,
            options: Optional[Dict[str, Any]]=None,
    ) -> Tuple[Union[np.ndarray, None], Dict]:
        super().reset(seed)

        # 重置 multi_magnetization_mcr 控制器
        self.multi_magnetization_mcr_controller_sofa.reset()

        # 重置奖励信息字典
        self.reward_info = {}

        # 填充上一帧奖励特征的初始值
        self.reward_features = {}
        self.reward_features["tip_pos_distance_to_dest_pos"] = self._get_distance_tip_to_dest()

        # 重置后，执行若干步空动作以稳定环境
        for _ in range(self._settle_steps):
            self.sofa_simulation.animate(self._sofa_root_node, self._sofa_root_node.getDt())

        return self._get_observation(image_observation=self._maybe_update_rgb_buffer()), {}


    def step(self, action: np.ndarray) -> Tuple[Union[np.ndarray, dict], float, bool, bool, dict]:
        """ 环境的 step 函数: 应用动作到仿真，并返回观测、奖励、终止标志和信息 """
        image_observation = super().step(action)
        observation = self._get_observation(image_observation=image_observation)
        reward = self._get_reward()
        terminated = self._get_done()
        info = self._get_info()
        return observation, reward, terminated, False, info

if __name__ == "__main__":
    env = MultiMagnetizaitonMCREnv(env_type=EnvType.AORTIC)
    env.reset()
    while True:
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        print(reward)
        if terminated or truncated:
            break
    env.close()