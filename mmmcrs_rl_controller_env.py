from __future__ import annotations
from typing import Any, Dict, Optional, Tuple, Union
import numpy as np
import gymnasium as gym

# 复用现有环境, 固定观测/动作类型并添加回合步数截断
from multi_magnetization_mcr_env import (
    MultiMagnetizaitonMCREnv,
    EnvType,
    RenderMode,
    ObservationType as _ObsType,
    ActionType,
    FLAT_CATHETER_DESTINATION_EXIT_POINT,
    AORTIC_CATHETER_DESTINATION_EXIT_POINT,
)


class MMMCRsRLControllerEnv(MultiMagnetizaitonMCREnv):
    """
    针对RL训练的轻量封装:
    - 固定 observation_type=STATE、action_type=CONTINUOUS
    - 默认 RenderMode.NONE(可在可视化/录制时改为 HEADLESS/HUMAN)
    - 新增 max_episode_steps, 通过 step() 返回 truncated=True
    - 观测与动作统一为 np.float32
    """

    def __init__(
        self,
        env_type: EnvType = EnvType.FLAT,
        image_shape: Tuple[int, int] = (64, 64),
        time_step: float = 0.05,
        frame_skip: int = 1,
        settle_steps: int = 5,
        render_mode: RenderMode = RenderMode.NONE,
        reward_amount_dict: Optional[Dict[str, float]] = None,
        target_distance_threshold: float = 0.015,
        num_catheter_tracking_points: int = 4,
        max_episode_steps: int = 600,
        create_scene_kwargs: Optional[dict] = None,
    ) -> None:
        self._max_episode_steps = int(max(1, max_episode_steps))
        self._elapsed_steps = 0

        super().__init__(
            image_shape=image_shape,
            create_scene_kwargs=create_scene_kwargs,
            observation_type=_ObsType.STATE,
            action_type=ActionType.CONTINUOUS,
            time_step=time_step,
            frame_skip=frame_skip,
            settle_steps=settle_steps,
            render_mode=render_mode,
            reward_amount_dict=reward_amount_dict,
            target_position=None,
            env_type=env_type,
            target_distance_threshold=target_distance_threshold,
            num_catheter_tracking_points=num_catheter_tracking_points,
        )

        # Gymnasium风格的属性(便于与常见RL库适配)
        self.metadata = {"render_modes": [m.name for m in RenderMode], "render_fps": int(1.0 / max(1e-6, self.time_step))}
        self.spec = None  # 可按需填充 gymnasium.EnvSpec

    # gymnasium API: 可选的 seed 方法
    def seed(self, seed: Optional[int] = None) -> None:
        # SofaEnv 在 reset(seed) 中处理随机种子；此处保留接口
        self._np_random = np.random.default_rng(seed)

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        obs, info = super().reset(seed=seed, options=options)
        self._elapsed_steps = 0
        if obs is None:
            # STATE 模式不应为 None, 但做个兜底
            obs = np.zeros(self.observation_space.shape, dtype=np.float32)
        obs = np.asarray(obs, dtype=np.float32).reshape(self.observation_space.shape)
        return obs, info

    def step(self, action: Union[np.ndarray, Dict[int, float], list, tuple]) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        # 将动作转换为 numpy 数组, 确保 dtype/shape
        if isinstance(action, (list, tuple)):
            action = np.asarray(action, dtype=np.float32)
        elif isinstance(action, dict):
            # 若来自旧的人类控制接口(dict), 转换为数组
            arr = np.zeros(self.action_space.shape, dtype=np.float32)
            for i, v in enumerate(action.values()):
                if i >= arr.shape[0]:
                    break
                arr[i] = float(v)
            action = arr
        else:
            action = np.asarray(action, dtype=np.float32)

        action = np.clip(action, self.action_space.low, self.action_space.high).astype(np.float32)

        obs, reward, terminated, truncated, info = super().step(action)
        self._elapsed_steps += 1

        # 触发时间截断
        if not (terminated or truncated):
            if self._elapsed_steps >= self._max_episode_steps:
                truncated = True
                # 可记录最终观测
                info = dict(info)
                info.setdefault("TimeLimit.truncated", True)

        # 观测与奖励类型保障
        if obs is None:
            obs = np.zeros(self.observation_space.shape, dtype=np.float32)
        obs = np.asarray(obs, dtype=np.float32).reshape(self.observation_space.shape)
        reward = float(reward)

        return obs, reward, bool(terminated), bool(truncated), info


__all__ = [
    "MMMCRsRLControllerEnv",
    "EnvType",
    "RenderMode",
]
