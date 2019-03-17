import argparse
import logging
from collections import deque

import numpy as np
import torch
from unityagents import UnityEnvironment

from ddpg_agent import Agent
from utils import plot_scores
from utils import save_checkpoint


def train(
    n_episodes,
    max_t,
    env_fp,
    no_graphics,
    seed,
    save_every_nth,
    buffer_size,
    batch_size,
    gamma,
    tau,
    lr_actor,
    lr_critic,
    weight_decay,
    log,
):
    log.info("#### Initializing environment...")
    # init environment
    env = UnityEnvironment(file_name=env_fp, no_graphics=no_graphics)

    # get the default brain
    brain_name = env.brain_names[0]
    brain = env.brains[brain_name]

    # reset the environment
    env_info = env.reset(train_mode=True)[brain_name]

    # number of agents
    num_agents = len(env_info.agents)
    log.info(f"Number of agents: {num_agents}")

    # size of each action
    action_size = brain.vector_action_space_size
    log.info(f"Size of each action: {action_size}")

    # examine the state space
    states = env_info.vector_observations
    state_size = states.shape[1]
    log.info(
        f"There are {states.shape[0]} agents. Each observes a state with length: {state_size}"
    )
    log.info(f"The state for the first agent looks like: {states[0]}")

    agent = Agent(
        num_agents=len(env_info.agents),
        state_size=state_size,
        action_size=action_size,
        buffer_size=buffer_size,
        batch_size=batch_size,
        gamma=gamma,
        tau=tau,
        lr_actor=lr_actor,
        lr_critic=lr_critic,
        weight_decay=weight_decay,
        random_seed=seed,
    )

    log.info("#### Training...")

    scores_deque = deque(maxlen=100)
    scores = []
    for i_episode in range(1, n_episodes + 1):
        brain_name = env.brain_names[0]
        env_info = env.reset(train_mode=True)[brain_name]
        states = env_info.vector_observations
        agent.reset()
        score = np.zeros((len(env_info.agents), 1))
        for t in range(max_t):
            actions = agent.act(states)
            env_info = env.step(actions)[brain_name]
            next_states = env_info.vector_observations
            rewards = env_info.rewards
            rewards = np.array(rewards).reshape((next_states.shape[0], 1))
            dones = env_info.local_done
            dones = np.array(dones).reshape((next_states.shape[0], 1))
            agent.step(states, actions, rewards, next_states, dones)
            score += rewards
            states = next_states
            if np.any(dones):
                break
        scores_deque.append(np.max(score))
        scores.append(np.max(score))
        print(
            "Episode {}\tAverage Score: {:.2f}\tScore: {:.2f}".format(
                i_episode, np.mean(scores_deque), scores[-1]
            ),
            end="\r",
        )

        if i_episode % 100 == 0:
            print(
                "\rEpisode {}\tAverage Score: {:.2f}".format(
                    i_episode, np.mean(scores_deque)
                )
            )
        if i_episode % save_every_nth == 0:
            save_checkpoint(
                state={
                    "episode": i_episode,
                    "actor_state_dict": agent.actor_local.state_dict(),
                    "critic_state_dict": agent.critic_local.state_dict(),
                    "scores_deque": scores_deque,
                    "scores": scores,
                },
                filename="checkpoint.pth",
            )
            plot_scores(
                scores=scores,
                title=f"Avg score over {len(env_info.agents)} agents",
                fname="avg_scores.png",
                savefig=True,
            )

        if np.mean(scores_deque) >= 0.5:
            torch.save(agent.actor_local.state_dict(), "checkpoint_actor.pth")
            torch.save(agent.critic_local.state_dict(), "checkpoint_critic.pth")
            print(
                "\nEnvironment solved in {:d} episodes!\tAverage Score: {:.2f}".format(
                    i_episode - 100, np.mean(scores_deque)
                )
            )
            break


def main():
    log = logging.getLogger(__name__)
    log.setLevel(logging.DEBUG)

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-env_fp",
        "--environment_filepath",
        help="Filepath to environment to load.",
        default="Tennis.app",
    )
    parser.add_argument(
        "-no_gr",
        "--no_graphics",
        help="Whether to display the environment.",
        action="store_true",
        default=True,
    )
    parser.add_argument(
        "-n_episodes",
        "--number_episodes",
        help="Number of episodes to train.",
        default=2500,
        type=int,
    )
    parser.add_argument(
        "-max_t",
        "--maximum_timesteps",
        help="Maximum number of timesteps within one episode.",
        default=1000,
        type=int,
    )
    parser.add_argument(
        "-seed", "--seed", help="Random seed for reproducibility.", default=2, type=int
    )
    parser.add_argument(
        "-save_every_nth",
        "--save_every_nth",
        help="Save checkpoint every nth episode",
        default=25,
        type=int,
    )
    parser.add_argument(
        "-bfsz",
        "--buffer_size",
        help="Buffer size of the replay memory.",
        default=int(1e5),
        type=int,
    )
    parser.add_argument(
        "-bsz",
        "--batch_size",
        help="Batch size of experience to sample from replay buffer.",
        default=256,
        type=int,
    )
    parser.add_argument(
        "-gamma",
        "--gamma",
        help="Discount factor for the cumulative rewards.",
        default=0.99,
        type=float,
    )
    parser.add_argument(
        "-tau",
        "--tau",
        help="Interpolation factor for soft update model parameters.",
        default=1e-3,
        type=float,
    )
    parser.add_argument(
        "-lr_actor",
        "--learning_rate_actor",
        help="Learning rate for SGD on actor's parameters.",
        default=1e-4,
        type=float,
    )
    parser.add_argument(
        "-lr_critic",
        "--learning_rate_critic",
        help="Learning rate for SGD on critic's parameters.",
        default=1e-3,
        type=float,
    )
    parser.add_argument(
        "-weight_decay",
        "--weight_decay",
        help="Weight decay (L2 penalty) for Adam optimizer.",
        default=0,
        type=float,
    )

    args = parser.parse_args()

    train(
        args.number_episodes,
        args.maximum_timesteps,
        args.environment_filepath,
        args.no_graphics,
        args.seed,
        args.save_every_nth,
        args.buffer_size,
        args.batch_size,
        args.gamma,
        args.tau,
        args.learning_rate_actor,
        args.learning_rate_critic,
        args.weight_decay,
        log,
    )


if __name__ == "__main__":
    main()
