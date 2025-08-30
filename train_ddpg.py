from __future__ import annotations
import argparse
import time
from pathlib import Path
import numpy as np
from collections import deque

from mmmcrs_rl_controller_env import MMMCRsRLControllerEnv, EnvType, RenderMode
from rl.ddpg import DDPG, DDPGConfig
from rl.replay_buffer import ReplayBuffer


def evaluate(env: MMMCRsRLControllerEnv, agent: DDPG, episodes: int = 5) -> tuple[float, float]:
    returns = []
    for _ in range(episodes):
        obs, _ = env.reset()
        done = False
        ep_ret = 0.0
        while not done:
            action = agent.act(obs, noise_scale=0.0)
            obs, r, term, trunc, _ = env.step(action)
            ep_ret += r
            done = term or trunc
        returns.append(ep_ret)
    return float(np.mean(returns)), float(np.std(returns))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--env_type", type=str, choices=[e.name for e in EnvType], default=EnvType.FLAT.name)
    parser.add_argument("--total_steps", type=int, default=100_000)
    parser.add_argument("--max_episode_steps", type=int, default=600)
    parser.add_argument("--buffer_size", type=int, default=1_000_000)
    parser.add_argument("--start_steps", type=int, default=10_000)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--tau", type=float, default=0.005)
    parser.add_argument("--lr_actor", type=float, default=3e-4)
    parser.add_argument("--lr_critic", type=float, default=3e-4)
    parser.add_argument("--eval_interval", type=int, default=10_000)
    parser.add_argument("--eval_episodes", type=int, default=3)
    parser.add_argument("--save_dir", type=str, default="checkpoints")
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--time_step", type=float, default=0.05)
    parser.add_argument("--frame_skip", type=int, default=1)
    parser.add_argument("--settle_steps", type=int, default=5)
    # parser.add_argument("--render_mode", type=str, choices=[m.name for m in RenderMode], default=RenderMode.NONE.name)
    parser.add_argument("--render_mode", type=str, choices=[m.name for m in RenderMode], default=RenderMode.HUMAN.name)
    parser.add_argument("--noise_init", type=float, default=0.2)
    parser.add_argument("--noise_final", type=float, default=0.05)
    # 日志控制
    parser.add_argument("--log_interval", type=int, default=1000, help="步级日志打印间隔 (步): 0 表示不按固定间隔打印")
    parser.add_argument("--step_log_on_violation", action="store_true", help="当出现工作空间违规时立即打印步级日志")
    parser.add_argument("--print_reward_components", action="store_true", help="在步级日志中打印奖励各组成项")
    args = parser.parse_args()

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # 环境
    env = MMMCRsRLControllerEnv(
        env_type=EnvType[args.env_type],
        image_shape=(512, 512),
        time_step=args.time_step,
        frame_skip=args.frame_skip,
        settle_steps=args.settle_steps,
        render_mode=RenderMode[args.render_mode],
        max_episode_steps=args.max_episode_steps,
        reward_amount_dict={
            "tip_pos_distance_to_dest_pos": -1.0,
            "delta_tip_pos_distance_to_dest_pos": -5.0,
            "workspace_constraint_violation": -5.0,
            "successful_task": 100.0,
        },
    )

    obs, _ = env.reset(seed=args.seed)
    s_dim = env.observation_space.shape[0]
    a_dim = env.action_space.shape[0]

    # Agent
    cfg = DDPGConfig(
        s_dim=s_dim,
        a_dim=a_dim,
        lr_actor=args.lr_actor,
        lr_critic=args.lr_critic,
        gamma=args.gamma,
        tau=args.tau,
        device=args.device,
    )
    agent = DDPG(cfg)

    # Replay buffer
    buffer = ReplayBuffer(s_dim, a_dim, capacity=args.buffer_size)

    # 探索噪声线性衰减
    def noise_scale(step: int) -> float:
        if args.total_steps <= 0:
            return args.noise_final
        frac = 1.0 - min(1.0, step / max(1, args.total_steps))
        return args.noise_final + (args.noise_init - args.noise_final) * frac

    # 日志统计
    ep_idx = 0
    ep_ret, ep_len = 0.0, 0
    ep_min_dist = float("inf")
    ep_last_dist = float("inf")
    ep_violations = 0
    ep_last_info = {}

    # 平滑损失日志
    last_actor_losses = deque(maxlen=200)
    last_critic_losses = deque(maxlen=200)

    best_eval = -np.inf
    start_time = time.perf_counter()

    for step in range(1, args.total_steps + 1):
        cur_noise = noise_scale(step)
        if step < args.start_steps:
            action = env.action_space.sample().astype(np.float32)
        else:
            action = agent.act(obs, noise_scale=cur_noise).astype(np.float32)

        next_obs, rew, term, trunc, info = env.step(action)
        done = term or trunc
        buffer.store(obs, action, rew, next_obs, done)

        # 步级统计（距离/违规）
        tip_dist = float(info.get("tip_pos_distance_to_dest_pos", np.nan))
        ep_min_dist = min(ep_min_dist, tip_dist) if not np.isnan(tip_dist) else ep_min_dist
        ep_last_dist = tip_dist if not np.isnan(tip_dist) else ep_last_dist
        if bool(info.get("workspace_constraint_violation", False)):
            ep_violations += 1
        ep_last_info = info

        obs = next_obs
        ep_ret += rew
        ep_len += 1

        # 更新
        if step >= args.start_steps and len(buffer) >= args.batch_size:
            batch = buffer.sample_batch(args.batch_size)
            actor_loss, critic_loss = agent.train_step(batch)
            last_actor_losses.append(actor_loss)
            last_critic_losses.append(critic_loss)

        # 步级日志
        should_log_step = (args.log_interval and (step % max(1, args.log_interval) == 0)) or (args.step_log_on_violation and bool(info.get("workspace_constraint_violation", False)))
        if should_log_step:
            msg = [
                f"Step {step}/{args.total_steps}",
                f"Ep {ep_idx + 1}",
                f"r {rew:.3f}",
                f"dist {tip_dist:.4f}" if not np.isnan(tip_dist) else "dist nan",
                f"Δdist {info.get('delta_tip_pos_distance_to_dest_pos', float('nan')):.5f}" if 'delta_tip_pos_distance_to_dest_pos' in info else "Δdist n/a",
                f"succ {bool(info.get('successful_task', False))}",
                f"viol {bool(info.get('workspace_constraint_violation', False))}",
                f"buf {len(buffer)}",
                f"noise {cur_noise:.3f}",
            ]
            if last_actor_losses and last_critic_losses:
                msg += [
                    f"a_loss {np.mean(last_actor_losses):.3f}",
                    f"c_loss {np.mean(last_critic_losses):.3f}",
                ]
            print(" | ".join(msg))

            if args.print_reward_components:
                comp_keys = [k for k in info.keys() if k.startswith("reward_")]
                comp_brief = {k: round(float(info[k]), 5) for k in sorted(comp_keys)}
                print(f"  reward_components: {comp_brief}")

        if done:
            ep_idx += 1
            elapsed = time.perf_counter() - start_time
            success = bool(ep_last_info.get("successful_task", False)) or term
            term_flag = bool(term)
            trunc_flag = bool(trunc)
            mean_a_loss = float(np.mean(last_actor_losses)) if last_actor_losses else float("nan")
            mean_c_loss = float(np.mean(last_critic_losses)) if last_critic_losses else float("nan")

            print(
                " || ".join(
                    [
                        f"Episode {ep_idx}",
                        f"len {ep_len}",
                        f"return {ep_ret:.3f}",
                        f"success {success}",
                        f"terminated {term_flag}",
                        f"truncated {trunc_flag}",
                        f"last_dist {ep_last_dist:.5f}",
                        f"min_dist {ep_min_dist:.5f}",
                        f"violations {ep_violations}",
                        f"buffer {len(buffer)}",
                        f"a_loss_avg {mean_a_loss:.3f}",
                        f"c_loss_avg {mean_c_loss:.3f}",
                        f"elapsed {elapsed/60:.2f} min",
                    ]
                )
            )

            # 回合结束，重置
            obs, _ = env.reset()
            ep_ret, ep_len = 0.0, 0
            ep_min_dist = float("inf")
            ep_last_dist = float("inf")
            ep_violations = 0
            ep_last_info = {}

        # 周期性评估与保存
        if step % args.eval_interval == 0 or step == args.total_steps:
            mean_r, std_r = evaluate(env, agent, episodes=args.eval_episodes)
            elapsed = time.perf_counter() - start_time
            print(f"[EVAL] Step {step}/{args.total_steps} | Return: {mean_r:.3f} ± {std_r:.3f} | Elapsed: {elapsed/60:.1f} min")

            # 保存 best/last
            agent.save(str(save_dir / "last.pt"))
            if mean_r > best_eval:
                best_eval = mean_r
                agent.save(str(save_dir / "best.pt"))

    env.close()


if __name__ == "__main__":
    main()