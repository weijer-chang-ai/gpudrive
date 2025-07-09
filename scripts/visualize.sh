#! /bin/bash

# Visualize rollouts
# python examples/experimental/viz_rollouts.py
# python examples/experimental/show_agent_behaviors.py
CHECKPOINT=/workspace/gpudrive/runs/PPO____S_75__07_09_01_12_32_090/model_PPO____S_75__07_09_01_12_32_090_001200.pt
CONFIG=/workspace/gpudrive/wandb/run-20250709_011241-PPO____S_75__07_09_01_12_32_090/files/config
DATA_PATH=/workspace/data/gpu_drive/validation

python examples/visualize.py \
    --checkpoint $CHECKPOINT \
    --config $CONFIG \
    --data_path $DATA_PATH \
    --num_envs 15 \