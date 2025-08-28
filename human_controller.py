from sofa_env.wrappers.trajectory_recorder import TrajectoryRecorder
from sofa_env.wrappers.realtime import RealtimeWrapper
from multi_magnetization_mcr_env import MultiMagnetizaitonMCREnv as MMMCREnv, EnvType
import cv2
from typing import Dict, Tuple
import numpy as np
import time
from collections import deque
from pathlib import Path
import argparse
from pynput import keyboard
import threading
from typing import List


class KeyboardController:
    """
    键盘控制器类，用于通过键盘输入控制磁性导管仿真系统

    该类监听键盘事件并实时更新控制参数，支持以下功能:
    - 使用 W/S 键控制导管的插入和回撤
    - 使用 A/D 键控制磁场绕X轴旋转
    - 使用 Q/E 键控制磁场绕Y轴旋转
    - 使用 ESC 键退出控制

    控制值范围为 [-1.0, 0.0, 1.0]，表示不同方向和强度的控制输入

    :param id: 控制器的唯一标识符，默认为0 (int)
    """
    def __init__(self, id: int=0) -> None:
        self.id = id
        self.r_x = 0.
        self.r_y = 0.
        self.retract = 0.
        self.x = 0.

        self._monitor_thread = threading.Thread(
            target=self._monitor_controller,
            daemon=True
        )
        print("starting keyboard thread")
        self._monitor_thread.start()


    def read(self) -> List[float]:
        """
        读取当前控制参数值

        :return
            List[float]: 包含三个控制参数的列表 [r_x, r_y, retract]
            - r_x: 绕X轴旋转的控制值 (-1.0: 左转, 0.0: 停止, 1.0: 右转)
            - r_y: 绕Y轴旋转的控制值 (-1.0: 下转, 0.0: 停止, 1.0: 上转)
            - retract: 导管进退的控制值 (-1.0: 回撤, 0.0: 停止, 1.0: 插入)
        """
        return [self.r_x, self.r_y, self.retract]

    def is_alive(self) -> bool:
        """
        检查键盘监听线程是否仍在运行

        :return
            bool: 如果线程仍在运行则返回True，否则返回False
        """
        return self._monitor_thread.is_alive()

    def _monitor_controller(self) -> None:
        """
        内部方法：监控键盘事件并更新控制参数

        键位映射:
            - W: 设置retract=1.0 (导管插入)
            - S: 设置retract=-1.0 (导管回撤)
            - A: 设置r_x=-1.0 (左转)
            - D: 设置r_x=1.0 (右转)
            - Q: 设置r_y=1.0 (上转)
            - E: 设置r_y=-1.0 (下转)
            - ESC: 设置x=1.0 (退出控制)

        当按键释放时，对应的控制参数会重置为0.0
        """
        with keyboard.Events() as events:
            for event in events:
                if type(event) == (keyboard.Events.Press):
                    if event.key == keyboard.Key.esc:
                        self.x = 1.
                        break
                    if hasattr(event.key, 'char'):
                        if event.key.char == 'w':
                            self.retract = 1.
                        elif event.key.char == 's':
                            self.retract = -1.

                        if event.key.char == 'a':
                            self.r_x = -1.
                        if event.key.char == 'd':
                            self.r_x = 1.

                        if event.key.char == 'q':
                            self.r_y = 1.
                        elif event.key.char == 'e':
                            self.r_y = -1.

                elif type(event) == keyboard.Events.Release:
                    if hasattr(event.key, 'char') and (event.key.char == "w" or event.key.char == "s"):
                        self.retract = 0.0
                    if hasattr(event.key, 'char') and (event.key.char == "a" or event.key.char == "d"):
                        self.r_x = 0.0
                    if hasattr(event.key, 'char') and (event.key.char == "q" or event.key.char == "e"):
                        self.r_y = 0.0

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Set human input behavior.")
    parser.add_argument("-rv", "--record_video", action="store_true", help="Record a track video.")
    parser.add_argument("-rt", "--record_trajectory", action="store_true", help="Record the full track.")
    parser.add_argument("-i", "--info", action="store", type=str, help="Additional information to be stored in the metadata.")
    parser.add_argument("--env_type", type=str, choices=[e.name for e in EnvType], default=EnvType.FLAT.name, help="Select the scene: FLAT or AORTIC")
    args = parser.parse_args()

    controller = KeyboardController()
    time.sleep(0.1)
    if not controller.is_alive():
        raise RuntimeError("Could not start controller.")

    image_shape = (1024, 1024)
    image_shape_to_save = (256, 256)

    env = MMMCREnv(image_shape=image_shape, env_type=EnvType[args.env_type])

    env = RealtimeWrapper(env)

    if args.record_video:
        video_folder = Path("videos")
        video_folder.mkdir(exist_ok=True)
        video_name = time.strftime("%Y%m%d-%H%M%S")
        video_path = video_folder / f"{video_name}.mp4"
        video_writer = cv2.VideoWriter(
            str(video_path),
            cv2.VideoWriter_fourcc(*"mp4v"),  # 类型: 忽略[attr-defined]
            1 / (env.time_step / env.frame_skip),
            image_shape[::-1],
            )
    else:
        video_writer = None

    if args.record_trajectory:

        def store_rgb_obs(self: TrajectoryRecorder, shape: Tuple[int, int] = image_shape_to_save):
            observation = self.env.render()
            observation = cv2.resize(
                observation,
                shape,
                interpolation=cv2.INTER_AREA,
            )
            self.trajectory["rgb"].append(observation)

        metadata = {
            "frame_skip": env.frame_skip,
            "time_step": env.time_step,
            "observation_type": env.observation_type.name,
            "reward_amount_dict": env.reward_amount_dict,
            "user_info": args.info,
        }

        env = TrajectoryRecorder(
            env,
            log_dir="trajectories",
            metadata=metadata,
            store_info=True,
            save_compressed_keys=["observation", "terminal_observation", "rgb", "info"],
            after_step_callbacks=[store_rgb_obs],
            after_reset_callbacks=[store_rgb_obs],
        )

    reset_obs, reset_info = env.reset()
    if video_writer is not None:
        video_writer.write(env.render()[:, :, ::-1])  # type: ignore[index]

    done = False

    fps_list = deque(maxlen=100)

    while not done:
        start = time.perf_counter()

        r_z, r_x, retract = controller.read()
        print(f"r_z: {r_z}    r_x: {r_x}    retract: {retract}")

        sample_action: Dict = env.action_space.sample()

        # 绕 z 轴旋转
        sample_action[0] = r_z
        # 绕 x 轴旋转
        sample_action[1] = r_x
        # 插入/回撤
        sample_action[2] = retract

        obs, reward, terminated, truncated, info = env.step(sample_action)
        done = terminated or truncated
        if video_writer is not None:
            video_writer.write(env.render()[:, :, ::-1])  # type: ignore[index]

        if controller.x:
            cv2.imwrite("exit_image.png", env.render()[:, :, ::-1])  # type: ignore[index]
            break

        end = time.perf_counter()
        fps = 1 / (end - start)
        fps_list.append(fps)
        print(f"FPS mean: {np.mean(fps_list):.5f}    std: {np.std(fps_list):.5f}")

    if video_writer is not None:
        video_writer.release()