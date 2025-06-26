"""Simple usage examples with Hydra configs."""

import hydra
from omegaconf import DictConfig
from gpudrive.utils.hydra_utils import create_env_from_cfg

@hydra.main(version_base=None, config_path="../gpudrive/configs", config_name="config")
def main(cfg: DictConfig) -> None:
    """Simple main function."""
    
    # Create environment
    env = create_env_from_cfg(cfg)
    
    print("Environment created successfully!")
    print(f"Using device: {env.device}")
    print(f"SMART enabled: {env.use_smart_reward}")
    print(f"VBD enabled: {env.use_vbd}")
    
    # Simple rollout
    obs = env.reset()
    print(f"Observation shape: {obs.shape}")
    
    for step in range(10):
        # Get expert actions
        expert_actions, _, _, _ = env.get_expert_actions()
        env.step_dynamics(expert_actions[:, :, step, :])
        
        # Get observations and rewards
        obs = env.get_obs()
        if env.use_smart_reward:
            reward = env.get_smart_rewards()
        else:
            reward = env.get_rewards()
        
        done = env.get_dones()
        
        print(f"Step {step}: Reward mean: {reward.mean().item():.4f}")
        
        if done.any():
            break
    
    env.close()
    print("Done!")

if __name__ == "__main__":
    main() 