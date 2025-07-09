#!/bin/bash

# Train PPO
CUDA_VISIBLE_DEVICES=1 python baselines/ppo/ppo_pufferlib.py baselines/ppo/config/ppo_base_puffer_nosmart.yaml
