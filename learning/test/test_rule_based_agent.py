#!/usr/bin/env python3
import os
dir_path = os.path.dirname(os.path.realpath(__file__))
lib_path = dir_path + '/../lib'
output_path = dir_path + '/output'
src_path = dir_path + '/../../src'
import sys
sys.path.insert(1, dir_path + '/../../environments')
sys.path.insert(1, lib_path)
import atexit
import argparse
import pickle
import numpy as np
import gymnasium as gym
from baseline_dynamic_agent import BaselineDynamicAgent
from active_nematic_discrete_env import ActiveNematicEnv
from param_utils import readParam

I = readParam('I')
J = readParam('J')
GEOM_NAME = readParam('GEOM_NAME')

SEND_PIPE_PATH = src_path + '/cpp/python_done'
RECV_PIPE_PATH = src_path + '/cpp/c_done'


def exit_handler():
    os.remove(SEND_PIPE_PATH)
    os.remove(RECV_PIPE_PATH)
    print('Removed named pipes')

atexit.register(exit_handler)

os.mkfifo(SEND_PIPE_PATH)
os.mkfifo(RECV_PIPE_PATH)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    args = parser.parse_args()

    env = ActiveNematicEnv(SEND_PIPE_PATH, RECV_PIPE_PATH)
    assert env.observation_space.shape is not None
    obs_size = env.observation_space.shape[0]
    assert isinstance(env.action_space, gym.spaces.Discrete)
    num_actions = int(env.action_space.n)
    env.initSim()

    agent = BaselineDynamicAgent()

    if GEOM_NAME == 'FREE':
        # local4
        #goal_frac_i = 0.7
        #goal_frac_j = 0.5

        # local8
        goal_frac_i = 0.35
        goal_frac_j = 0.55
    elif GEOM_NAME == 'T_JUNCTION':
        goal_frac_i = 19.5 / 21
        goal_frac_j = 0.5
    elif GEOM_NAME == 'CROSS_JUNCTION':
        #goal_frac_i = 0.75  # right channel
        #goal_frac_j = 0.5

        # intermediate goal for top or bottom channel
        goal_frac_i = 0.5
        goal_frac_j = 0.5
    else:
        raise ValueError(f'Unsupported geometry name {GEOM_NAME}')

    # Test
    num_steps = 0
    defect_positions = []
    total_reward = 0.0
    env.reset(options={'goal': [goal_frac_i, goal_frac_j]})
    state = {'defect_pos': env.getNearestDefect()}
    #env.loadAllowedPrimitivesList(filepath=src_path + '/cpp/primitives/allowed_primitives_list.dat.local4')  # use only for FREE local4
    env.loadAllowedPrimitivesList(filepath=src_path + '/cpp/primitives/allowed_primitives_list.dat.local8')  # use for everything else
    defect_positions.append(env.getNearestDefect())

    while True:
        if GEOM_NAME == 'FREE':
            # local4
            #goal_frac_i = 0.7
            #goal_frac_j = 0.5

            # local8
            goal_frac_i = 0.35
            goal_frac_j = 0.55
        elif GEOM_NAME == 'T_JUNCTION':
            if state['defect_pos'][0] <= I * 18 / 21:  # if in horizontal channel, aim for center of intersection
                goal_frac_i = 19.5 / 21
                goal_frac_j = 0.5
            else:  # if in vertical channel, aim for true goal
                goal_frac_i = 19.5 / 21
                #goal_frac_j = 0.75  # top channel
                goal_frac_j = 0.25  # bottom channel
        elif GEOM_NAME == 'CROSS_JUNCTION':
            #goal_frac_i = 0.75  # right channel
            #goal_frac_j = 0.5

            if state['defect_pos'][0] <= I * 9 / 21:  # if in left horizontal channel, aim for center of intersection
                goal_frac_i = 0.5
                goal_frac_j = 0.5
            else:
                goal_frac_i = 0.5
                goal_frac_j = 0.75  # top channel
                #goal_frac_j = 0.25  # bottom channel
        else:
            raise ValueError(f'Unsupported geometry name {GEOM_NAME}')

        env.setGoal(goal=[goal_frac_i, goal_frac_j])
        state['goal_pos'] = [goal_frac_i * I, goal_frac_j * J]
        action = agent.predict(state)
        _, reward, is_done, is_trunc, _ = env.step(action)
        total_reward += reward
        nearest_defect_pos = env.getNearestDefect()
        state['defect_pos'] = nearest_defect_pos
        defect_positions.append(nearest_defect_pos)

        if is_done or is_trunc:
            env.endSim()
            break

        num_steps += 1

    # Save position data
    with open(output_path + '/defect_positions_dynamic.pp', 'wb') as f:
        pickle.dump(defect_positions, f)

    print("Total reward: %.2f" % total_reward)
    env.close()
