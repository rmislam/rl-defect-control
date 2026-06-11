#!/usr/bin/env python3
import os
dir_path = os.path.dirname(os.path.realpath(__file__))
src_path = dir_path + '/../../src'
output_path = dir_path + '/output'
import sys
sys.path.insert(1, dir_path + '/../../environments')
import atexit
import argparse
import pickle
from stable_baselines3 import PPO
from active_nematic_rotated_multibinary_env import ActiveNematicEnv
from param_utils import readParam

I = readParam('I')
J = readParam('J')

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
    env.setQuadrant('BL')
    env.setRotation(0)

    model_straight = PPO.load(dir_path + '/../saved_models/ppo-active-nematic-multibinary-CROSS_JUNCTION-straight-single-defect/ppo-active-nematic-multibinary-CROSS_JUNCTION_280064_steps.zip', env=env)
    model_up = PPO.load(dir_path + '/../saved_models/ppo-active-nematic-multibinary-CROSS_JUNCTION-up/ppo-active-nematic-multibinary-CROSS_JUNCTION_192000_steps.zip', env=env)
    model = model_straight

    # Test
    num_steps = 0
    defect_positions = []
    goal_positions = []
    total_reward = 0.0
    goal_frac_i = 0.75
    goal_frac_j = 0.5
    state, _ = env.reset(options={'goal': [goal_frac_i, goal_frac_j]})
    env.loadAllowedPrimitivesList(filepath=src_path + '/cpp/primitives/allowed_primitives_list.dat.left_right')
    defect_positions.append(env.getNearestDefect())
    goal_positions.append([goal_frac_i * I, goal_frac_j * J])

    while True:
        action, _ = model.predict(state, deterministic=True)
        state, reward, is_done, is_trunc, _ = env.step(action)
        total_reward += reward
        defect_positions.append(env.getNearestDefect())
        goal_positions.append([goal_frac_i * I, goal_frac_j * J])

        if is_done or is_trunc:
            env.endSim()
            break

        num_steps += 1

        if num_steps == 7:
            env.setQuadrant('BR')  # be sure to set quadrant before doing anything else
            env.setRotation(0)
            goal_frac_i = 0.5
            goal_frac_j = 0.75
            env.setGoal(goal=[goal_frac_i, goal_frac_j])
            state = env.getState()
            model = model_up
            env.loadAllowedPrimitivesList(filepath=src_path + '/cpp/primitives/allowed_primitives_list.dat.left_top')
        elif num_steps == 14:
            env.setQuadrant('TR')  # be sure to set quadrant before doing anything else
            env.setRotation(-1)
            goal_frac_i = 0.5
            goal_frac_j = 0.75
            env.setGoal(goal=[goal_frac_i, goal_frac_j])
            state = env.getState()
            model = model_up
            env.loadAllowedPrimitivesList(filepath=src_path + '/cpp/primitives/allowed_primitives_list.dat.left_top')
        # elif num_steps == 24:
        #     env.setQuadrant('TL')  # be sure to set quadrant before doing anything else
        #     env.setRotation(-2)
        #     goal_frac_i = 0.5
        #     goal_frac_j = 330 / 420
        #     env.setGoal(goal=[goal_frac_i, goal_frac_j])
        #     state = env.getState()
        #     model = model_up
        #     env.loadAllowedPrimitivesList(filepath=src_path + '/cpp/primitives/allowed_primitives_list.dat.left_top')

    with open(output_path + '/defect_positions_rl.pp', 'wb') as f:
        pickle.dump(defect_positions, f)

    with open(output_path + '/goal_positions.pp', 'wb') as f:
        pickle.dump(goal_positions, f)

    print("Total reward: %.2f" % total_reward)
    env.close()
