import argparse
import math
import os
import random
import re
import time

import numpy as np
import torch
from ruamel.yaml import YAML, RoundTripDumper, dump
from tqdm import tqdm

import modules
from raisim_gym_torch.env.bin import joint_pos
from raisim_gym_torch.env.RaisimGymVecEnv import RaisimGymVecEnv as VecEnv


def natural_sort(l):
    ### https://stackoverflow.com/a/4836734
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    alphanum_key = lambda key: [convert(c) for c in re.split("([0-9]+)", key)]
    return sorted(l, key=alphanum_key)


# configuration
parser = argparse.ArgumentParser()
parser.add_argument("-w", "--weight", help="trained weight path", type=str, default="")
args = parser.parse_args()

# directories
task_name = "joint_pos"
home_path = os.path.realpath(os.path.dirname(os.path.realpath(__file__)) + "/../../")
task_path = home_path + "/gym_envs/" + task_name

# config
cfg = YAML().load(open(task_path + "/cfg.yaml", "r"))

# create environment from the configuration file
cfg["environment"]["render"] = True
# cfg["environment"]["server"]["port"] = 8080

env = VecEnv(
    joint_pos.RaisimGymEnv(
        home_path + "/resources", dump(cfg["environment"], Dumper=RoundTripDumper)
    ),
    cfg["environment"],
)

# Set seed
env.seed(cfg["seed"])
random.seed(cfg["seed"])
torch.manual_seed(cfg["seed"])
np.random.seed(cfg["seed"])

actor_critic_module = modules.get_actor_critic_module_from_config(
    cfg, env, joint_pos.NormalSampler, device="cpu"
)

# shortcuts
ob_dim = env.num_obs
act_dim = env.num_acts

weight_path = args.weight

if weight_path == "" or not weight_path:
    data_path = home_path + "/data/" + task_name
    sub_dirs = [f.path for f in os.scandir(data_path) if f.is_dir()]
    sub_dirs.sort()

    policy_paths = None

    if len(sub_dirs) > 0:
        policy_paths = [
            f.path
            for f in os.scandir(sub_dirs[-1])
            if f.is_file() and f.path.endswith(".pt")
        ]

        if len(policy_paths) > 0:
            policy_paths = natural_sort(policy_paths)
        else:
            policy_paths = None

    if policy_paths is None:
        print(
            "Can't find trained weight, please provide a trained weight with --weight switch\n"
        )
        weight_path = None
    else:
        weight_path = policy_paths[-1]

if weight_path:
    iteration_number = weight_path.rsplit("/", 1)[1].split("_", 1)[1].rsplit(".", 1)[0]
    weight_dir = weight_path.rsplit("/", 1)[0] + "/"

if weight_path is None:
    exit()
else:
    print("Loaded weight from {}\n".format(weight_path))
    start = time.time()

    env.reset()
    actor_critic_module.reset()

    reward_ll_sum = 0
    done_sum = 0
    average_dones = 0.0
    n_steps = math.floor(
        cfg["environment"]["max_time"] / cfg["environment"]["control_dt"]
    )
    total_steps = n_steps * 1
    start_step_id = 0

    print("Visualizing and evaluating the policy: ", weight_path)
    actor_critic_module.load_parameters(weight_path)

    env.load_scaling(weight_dir, int(iteration_number))
    env.turn_on_visualization()

    max_steps = 1000

    obs_mean, obs_var = env.mean, env.var
    num_envs = cfg["environment"]["num_envs"]
    expert_data = {
        "observations": np.zeros((num_envs, max_steps, 36)),
        "actions": np.zeros((num_envs, max_steps, 12)),
        "terminals": np.zeros((num_envs, max_steps, 1)),
        "vel_cmds": np.zeros((num_envs, max_steps, 3)),
    }
    tot_terminals = 0
    # walk
    # action_mean = np.array(
    #     [-0.14, 0.81, -0.96, 0.14, 0.81, -0.96, -0.14, -0.81, 0.96, 0.14, -0.81, 0.96]
    # )
    # action_mean = np.array(
    #     [-0.30, 1.90, -2.27, 0.30, 1.90, -2.27, -0.30, -1.90, 2.27, 0.30, -1.90, 2.27]
    # )
    # mid
    action_mean = np.array(
        [-0.17, 1.25, -1.56, 0.17, 1.25, -1.56, -0.17, -1.25, 1.56, 0.17, -1.25, 1.56]
    )
    # low crawl
    # action_mean = np.array(
    #     [-0.06, 1.41, -1.88, 0.06, 1.41, -1.88, -0.06, -1.41, 1.88, 0.06, -1.41, 1.88]
    # )

    action_ll = np.zeros((num_envs, 12), dtype=np.float32)
    torques = np.zeros((num_envs, 12), dtype=np.float32)
    for step in tqdm(range(max_steps)):
        with torch.no_grad():
            # time.sleep(cfg["environment"]["control_dt"])
            obs = env.observe(False)
            obs_unnorm = (obs * np.sqrt(obs_var) + obs_mean)[:, :36]
            base_pos = env.get_base_position()
            orientation = env.get_base_orientation()

            prev_action_ll = actor_critic_module.generate_action(
                torch.from_numpy(obs).cpu()
            )
            reward_ll, dones = env.step(action_ll)

            actor_critic_module.update_dones(dones)
            reward_ll_sum = reward_ll_sum + reward_ll[0]

            # expert_data["observations"][:, step, :2] = base_pos[:, :2]
            # expert_data["observations"][:, step, 2:6] = orientation
            expert_data["observations"][:, step, :36] = obs_unnorm[:, :36]
            expert_data["actions"][:, step, :] = (
                0.5 * prev_action_ll.cpu().detach().numpy() + action_mean
            )
            expert_data["terminals"][:, step, :] = dones.reshape(-1, 1)
            expert_data["vel_cmds"][:, step] = obs_unnorm[:, 33:36]

            if dones.any():
                tot_terminals += sum(dones)

            action_ll = prev_action_ll.cpu().detach().numpy()
            torques = env.get_torques()

    env.turn_off_visualization()
    env.reset()

    name = "crawl"
    np.save("expert_data/" + name + ".npy", expert_data)
    print(expert_data["observations"].shape)
    print("Number of terminals: ", tot_terminals)
    print("Expert data saved as " + name + ".npy")

    print("Finished at the maximum visualization steps")
