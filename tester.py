# Inspired from https://github.com/raillab/dqn
import random
import numpy as np
import gym

from dqn.agent import DQNAgent

from dqn.replay_buffer import ReplayBuffer
from dqn.wrappers import *
import torch
import argparse
from CustomFlapy import CustomFlapy
from CustomMonsterKong import CustomMonsterKong

if __name__ == "__main__":

    device = (torch.device("cuda" if torch.cuda.is_available() else "cpu"),)

    parser = argparse.ArgumentParser(description="DQN Atari")
    parser.add_argument(
        "--load-checkpoint-file",
        type=str,
        default="./flappybird_weights.pth",
        help="Where checkpoint file should be loaded from (usually results/checkpoint.pth)",
    )
    parser.add_argument(
        "--env",
        type=str,
        default="flappy",
        help="The envirenment to train on. Default is flappy. Options: flappy, monsterkong, waterworld",
    )

    parser.add_argument(
        "--force-fps",
        type=int,
        default=1,
        help="force fps",
    )
    args = parser.parse_args()
    # If you have a checkpoint file, spend less time exploring
    force_fps = False if args.force_fps == 1 else True
    if args.load_checkpoint_file:
        eps_start = 0.01
    else:
        eps_start = 1

    hyper_params = {
        "seed": 42,  # which seed to use
        "env": "PongNoFrameskip-v4",  # name of the game
        "replay-buffer-size": int(5e3),  # replay buffer size
        "learning-rate": 1e-4,  # learning rate for Adam optimizer
        "discount-factor": 0.99,  # discount factor
        "dqn_type": "neurips",
        # total number of steps to run the environment for
        "num-steps": int(1e6),
        "batch-size": 32,  # number of transitions to optimize at the same time
        "learning-starts": 10000,  # number of steps before learning starts
        "learning-freq": 1,  # number of iterations between every optimization step
        "use-double-dqn": True,  # use double deep Q-learning
        "target-update-freq": 1000,  # number of iterations between every target network update
        "eps-start": eps_start,  # e-greedy start threshold
        "eps-end": 0.01,  # e-greedy end threshold
        "eps-fraction": 0.1,  # fraction of num-steps
        "print-freq": 10,
    }

    np.random.seed(hyper_params["seed"])
    random.seed(hyper_params["seed"])

    assert "NoFrameskip" in hyper_params["env"], "Require environment with no frameskip"
    # env = gym.make(hyper_params["env"])
    env = None
    if args.env == "flappy":
        env = CustomFlapy(force_fps)
    elif args.env == "monsterkong":
        env = CustomMonsterKong(force_fps)
    assert env is not None, "Invalid environment"
    env.seed(hyper_params["seed"])

    env = NoopResetEnv(env, noop_max=30)
    env = MaxAndSkipEnv(env, skip=4)
    env = WarpFrame(env)
    env = PyTorchFrame(env)
    env = ClipRewardEnv(env)
    env = FrameStack(env, 4)
    # env = gym.wrappers.Monitor(
    #     env,
    #     "./video/",
    #     video_callable=lambda episode_id: episode_id % 50 == 0,
    #     force=True,
    # )

    replay_buffer = ReplayBuffer(hyper_params["replay-buffer-size"])

    agent = DQNAgent(
        env.observation_space,
        env.action_space,
        replay_buffer,
        use_double_dqn=hyper_params["use-double-dqn"],
        lr=hyper_params["learning-rate"],
        batch_size=hyper_params["batch-size"],
        gamma=hyper_params["discount-factor"],
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        dqn_type=hyper_params["dqn_type"],
    )
    print(agent.device)
    if args.load_checkpoint_file:
        print(f"Loading a policy - { args.load_checkpoint_file } ")
        agent.policy_network.load_state_dict(torch.load(args.load_checkpoint_file))

    eps_timesteps = hyper_params["eps-fraction"] * float(hyper_params["num-steps"])
    episode_rewards = [0.0]

    state = env.reset()
    reward_general = 0
    game = 0
    cum_reward = 0 
    while True:
        obs = env.reset()
        done = False
        print("reward: ", reward_general)
        reward_general = 0 
        while not done:
            action = agent.act(obs)
            obs, reward, done, _ = env.step(action)
            env.render()
            reward_general += reward
            if done:
                break
        game += 1
        cum_reward+=reward_general
        if game == 100:
            print("avg reward: ", cum_reward / 100)
            break