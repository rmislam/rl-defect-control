#!/usr/bin/env python3
import os
dir_path = os.path.dirname(os.path.realpath(__file__))
output_path = dir_path + '/output'
src_path = dir_path + '/../../src'
import sys
sys.path.insert(1, dir_path + '/../../environments')
import atexit
import argparse
import pickle
import numpy as np
from active_nematic_multibinary_env import ActiveNematicEnv
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
    env.initSim()

    # Test
    num_steps = 0
    defect_positions = []
    total_reward = 0.0

    if GEOM_NAME == 'FREE':
        # local4
        goal_frac_i = 0.7
        goal_frac_j = 0.5

        # local8
        #goal_frac_i = 0.35
        #goal_frac_j = 0.55
    elif GEOM_NAME == 'T_JUNCTION':
        # Up
        goal_frac_i = 390 / 420
        goal_frac_j = 0.75

        # Down
        #goal_frac_i = 390 / 420
        #goal_frac_j = 0.25

    elif GEOM_NAME == 'CROSS_JUNCTION':
        # Up
        goal_frac_i = 0.5
        goal_frac_j = 0.75

        # Down
        #goal_frac_i = 0.5
        #goal_frac_j = 0.25

        # Straight
        #goal_frac_i = 0.75
        #goal_frac_j = 0.5
    else:
        raise ValueError(f'Unsupported geometry name {GEOM_NAME}')

    state, _ = env.reset(options={'goal': [goal_frac_i, goal_frac_j]})

    if GEOM_NAME == 'FREE':
        # local4
        env.loadAllowedPrimitivesList(filepath=src_path + '/cpp/primitives/allowed_primitives_list.dat.local4')

        # local8
        #env.loadAllowedPrimitivesList(filepath=src_path + '/cpp/primitives/allowed_primitives_list.dat.local8')
    elif GEOM_NAME == 'T_JUNCTION':
        # Up
        env.loadAllowedPrimitivesList(filepath=src_path + '/cpp/primitives/allowed_primitives_list.dat.t_junc_top')

        # Down
        #env.loadAllowedPrimitivesList(filepath=src_path + '/cpp/primitives/allowed_primitives_list.dat.t_junc_bot')        
    elif GEOM_NAME == 'CROSS_JUNCTION':
        # Up
        env.loadAllowedPrimitivesList(filepath=src_path + '/cpp/primitives/allowed_primitives_list.dat.left_top')

        # Down
        #env.loadAllowedPrimitivesList(filepath=src_path + '/cpp/primitives/allowed_primitives_list.dat.left_bot')

        # Straight
        #env.loadAllowedPrimitivesList(filepath=src_path + '/cpp/primitives/allowed_primitives_list.dat.left_right')
    else:
        print(f'Unsupported GEOM_NAME: {GEOM_NAME}')
        exit()

    defect_positions.append(env.getNearestDefect())

    while True:
        action = np.ones(env.action_space.n)  # enable all actions every timestep to produce static pattern
        state, reward, is_done, is_trunc, _ = env.step(action)
        total_reward += reward
        defect_positions.append(env.getNearestDefect())

        if is_done:  # TODO: See if this keeps the pattern on for the whole episode
            env.endSim()
            break

        num_steps += 1

    # Save position data
    with open(output_path + '/defect_positions_static.pp', 'wb') as f:
        pickle.dump(defect_positions, f)

    print("Total reward: %.2f" % total_reward)
    env.close()
