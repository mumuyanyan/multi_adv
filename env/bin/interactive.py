#!/usr/bin/env python
import os,sys
sys.path.insert(1, os.path.join(sys.path[0], '..'))
import argparse

from env.multiagent.environment import MultiAgentEnv
from env.multiagent.policy import InteractivePolicy
import env.multiagent.scenarios as scenarios

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description=None)
    parser.add_argument('-s', '--scenario', default='simple_spread.py', help='Path of the scenario Python script.')
    args = parser.parse_args()


    scenario = scenarios.load(args.scenario).Scenario()

    world = scenario.make_world()

    env = MultiAgentEnv(world, scenario.reset_world, scenario.reward, scenario.observation, info_callback=None, shared_viewer = False)

    env.render()

    policies = [InteractivePolicy(env,i) for i in range(env.n)]

    obs_n = env.reset()
    while True:

        act_n = []
        for i, policy in enumerate(policies):
            act_n.append(policy.action(obs_n[i]))

        obs_n, reward_n, done_n, _ = env.step(act_n)
        print(act_n)

        env.render()



