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
from active_nematic_multibinary_env import ActiveNematicEnv
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

    model = PPO.load(dir_path + '/../saved_models/ppo-active-nematic-multibinary-T_JUNCTION-up/ppo-active-nematic-multibinary-T_JUNCTION_209920_steps.zip', env=env)

    # Test
    num_steps = 0
    defect_positions = []
    total_reward = 0.0
    goal_frac_i = 390 / 420
    goal_frac_j = 0.75
    state, _ = env.reset(options={'goal': [goal_frac_i, goal_frac_j]})
    env.loadAllowedPrimitivesList(filepath=src_path + '/cpp/primitives/allowed_primitives_list.dat.t_junc_top')
    defect_positions.append(env.getNearestDefect())

    while True:
        action, _ = model.predict(state, deterministic=True)
        state, reward, is_done, is_trunc, _ = env.step(action)
        total_reward += reward
        defect_positions.append(env.getNearestDefect())

        if is_done or is_trunc:
            env.endSim()
            break

        num_steps += 1

    # Save position data
    with open(output_path + '/defect_positions_rl.pp', 'wb') as f:
        pickle.dump(defect_positions, f)

    # Save goal
    goal_position = [goal_frac_i * I, goal_frac_j * J]

    with open(output_path + '/goal_position.pp', 'wb') as f:
        pickle.dump(goal_position, f)

    print("Total reward: %.2f" % total_reward)
    env.close()
