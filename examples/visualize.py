#!/usr/bin/env python3
"""
Local Checkpoint Visualization Script
Based on examples/tutorials/04_use_pretrained_sim_agent.ipynb

Usage:
    python visualize_checkpoint.py --checkpoint /path/to/model.pt --config config_name --data_path /path/to/data
"""

import torch
import dataclasses
import mediapy
import argparse
from pathlib import Path
import logging

from gpudrive.networks.late_fusion import NeuralNet
from gpudrive.env.config import EnvConfig
from gpudrive.env.env_torch import GPUDriveTorchEnv
from gpudrive.visualize.utils import img_from_fig
from gpudrive.env.dataset import SceneDataLoader
from gpudrive.utils.config import load_config

logging.basicConfig(level=logging.INFO)


def load_local_model(checkpoint_path: str, device: str) -> NeuralNet:
    """Load model from local checkpoint (replaces HuggingFace loading)."""
    logging.info(f"Loading model from {checkpoint_path}")
    
    # Load checkpoint
    saved_cpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    # Extract model info
    model_arch = saved_cpt["model_arch"]
    action_dim = saved_cpt["action_dim"]
    
    # Create model (same as notebook but from local checkpoint)
    model = NeuralNet(
        input_dim=model_arch["input_dim"],
        action_dim=action_dim,
        hidden_dim=model_arch["hidden_dim"],
    ).to(device)
    
    # Load weights
    model.load_state_dict(saved_cpt["parameters"])
    
    logging.info(f"Model loaded successfully - Action dim: {action_dim}, Input dim: {model_arch['input_dim']}")
    return model.eval()


def main(checkpoint_path: str, config_name: str, data_path: str, 
         num_envs: int = 2, dataset_size: int = 100, device: str = "cpu",
         zoom_radius: int = 70, deterministic: bool = False, 
         output_dir: str = "videos"):
    
    # === Load config (same as notebook) ===
    config = load_config(config_name)
    config = config.environment.value # we need to do this somehow due to Box object
    print("Loaded config:")
    print(config)
    
    max_agents = config.max_controlled_agents
    
    # === Load model from local checkpoint (replaces HuggingFace) ===
    sim_agent = load_local_model(checkpoint_path, device)
    
    print(f"Agent action dim: {sim_agent.action_dim}")
    print(f"Agent obs dim: {sim_agent.obs_dim}")
    
    # === Make environment (same as notebook) ===
    # Create data loader
    train_loader = SceneDataLoader(
        root=data_path,
        batch_size=num_envs,
        dataset_size=dataset_size,
        sample_with_replacement=False,
    )
    
    # Set params (same as notebook)
    env_config = dataclasses.replace(
        EnvConfig(),
        ego_state=config.ego_state,
        road_map_obs=config.road_map_obs,
        partner_obs=config.partner_obs,
        reward_type=config.reward_type,
        norm_obs=config.norm_obs,
        dynamics_model=config.dynamics_model,
        collision_behavior=config.collision_behavior,
        dist_to_goal_threshold=config.dist_to_goal_threshold,
        polyline_reduction_threshold=config.polyline_reduction_threshold,
        remove_non_vehicles=config.remove_non_vehicles,
        lidar_obs=config.lidar_obs,
        disable_classic_obs=config.lidar_obs,
        obs_radius=config.obs_radius,
        steer_actions=torch.round(
            torch.linspace(-torch.pi, torch.pi, config.action_space_steer_disc), decimals=3  
        ),
        accel_actions=torch.round(
            torch.linspace(-4.0, 4.0, config.action_space_accel_disc), decimals=3
        ),
    )
    
    # Make env (same as notebook)
    env = GPUDriveTorchEnv(
        config=env_config,
        data_loader=train_loader,
        max_cont_agents=config.max_controlled_agents,
        device=device,
    )
    
    print(f"Environment data batch: {env.data_batch}")
    
    # === Rollout (same as notebook) ===
    next_obs = env.reset()
    control_mask = env.cont_agent_mask
    
    print(f"Observation shape: {next_obs.shape}")
    
    frames = {f"env_{i}": [] for i in range(num_envs)}
    
    for time_step in range(env.episode_len):
        print(f"\rStep: {time_step}", end="", flush=True)
        
        # Predict actions
        action, _, _, _, _ = sim_agent(
            next_obs[control_mask], deterministic=deterministic
        )
        action_template = torch.zeros(
            (num_envs, max_agents), dtype=torch.int64, device=device
        )
        action_template[control_mask] = action.to(device)
        
        # Step
        env.step_dynamics(action_template)
        
        # Render    
        sim_states = env.vis.plot_simulator_state(
            env_indices=list(range(num_envs)),
            time_steps=[time_step]*num_envs,
            zoom_radius=zoom_radius,
        )
        
        for i in range(num_envs):
            frames[f"env_{i}"].append(img_from_fig(sim_states[i])) 
        
        next_obs = env.get_obs()
        reward = env.get_rewards() 
        done = env.get_dones()
        info = env.get_infos()
        
        if done.all():
            break
    
    print(f"\nRollout completed after {time_step + 1} steps")
    env.close()
    
    # === Save videos ===
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    checkpoint_name = Path(checkpoint_path).stem
    
    for env_id, env_frames in frames.items():
        if env_frames:  # Only save if we have frames
            video_path = output_path / f"{checkpoint_name}_{env_id}.gif"
            mediapy.write_video(str(video_path), env_frames, fps=15, codec='gif')
            logging.info(f"Saved video: {video_path}")
    
    logging.info(f"All videos saved to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize local checkpoint")
    parser.add_argument("--checkpoint", required=True, help="Path to model checkpoint (.pt file)")
    parser.add_argument("--config", required=True, help="Config name (without .yaml extension)")
    parser.add_argument("--data_path", required=True, help="Path to dataset")
    parser.add_argument("--num_envs", type=int, default=10, help="Number of parallel environments")
    parser.add_argument("--dataset_size", type=int, default=100, help="Number of scenes to sample")
    parser.add_argument("--device", default="cpu", help="Device to use (cpu/cuda)")
    parser.add_argument("--zoom_radius", type=int, default=70, help="Zoom radius for visualization")
    parser.add_argument("--deterministic", action="store_true", help="Use deterministic policy")
    parser.add_argument("--output_dir", default="videos", help="Output directory for videos")
    
    args = parser.parse_args()
    
    main(
        checkpoint_path=args.checkpoint,
        config_name=args.config,
        data_path=args.data_path,
        num_envs=args.num_envs,
        dataset_size=args.dataset_size,
        device=args.device,
        zoom_radius=args.zoom_radius,
        deterministic=args.deterministic,
        output_dir=args.output_dir
    )