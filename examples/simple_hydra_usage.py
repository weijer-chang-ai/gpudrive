"""Simple usage examples with Hydra configs."""

import hydra
from omegaconf import DictConfig
from gpudrive.utils.hydra_utils import create_env_from_cfg


@hydra.main(version_base=None, config_path="../gpudrive/configs", config_name="config")
def main(cfg: DictConfig) -> None:
    """Simple main function."""
    
    print("=== GPUDrive Hydra Config Example ===")
    print(f"Environment: {cfg.env._target_}")
    print(f"Device: {cfg.device}")
    print(f"Dynamics model: {cfg.env.dynamics_model}")
    
    # Check if integrations are loaded and enabled
    smart_enabled = hasattr(cfg.integrations, 'smart')
    vbd_enabled = hasattr(cfg.integrations, 'vbd')
    
    print(f"SMART enabled: {smart_enabled}")
    print(f"VBD enabled: {vbd_enabled}")
    
    # Create environment
    env = create_env_from_cfg(cfg)
    
    print(f"\nEnvironment created successfully!")
    print(f"Using device: {env.device}")
    print(f"SMART rewards: {env.use_smart_reward}")
    print(f"VBD enabled: {env.use_vbd}")
    print(f"Max agents: {env.max_cont_agents}")
    
    # Simple rollout
    obs = env.reset()
    print(f"Observation shape: {obs.shape}")
    
    print("\nRunning simulation...")
    for step in range(10):
        # Get expert actions
        expert_actions, _, _, _ = env.get_expert_actions()
        env.step_dynamics(expert_actions[:, :, step, :])
        
        # Get observations and rewards
        obs = env.get_obs()
        if env.use_smart_reward:
            reward = env.get_smart_rewards()
            print(f"Step {step}: SMART reward mean: {reward.mean().item():.4f}")
        else:
            reward = env.get_rewards()
            print(f"Step {step}: Standard reward mean: {reward.mean().item():.4f}")
        
        done = env.get_dones()
        
        if done.any():
            print(f"Episode ended at step {step}")
            break
    
    env.close()
    print("Done!")


if __name__ == "__main__":
    main() 