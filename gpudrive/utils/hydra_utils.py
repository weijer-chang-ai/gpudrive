"""Simple utilities for working with Hydra configs in GPUDrive."""

from omegaconf import DictConfig
from gpudrive.env.env_torch import GPUDriveTorchEnv
from gpudrive.env.dataset import SceneDataLoader
from gpudrive.env.config import EnvConfig, RenderConfig
from hydra.utils import instantiate
import torch
from typing import Optional

def create_env_from_cfg(cfg: DictConfig) -> GPUDriveTorchEnv:
    """Simple function to create GPUDrive environment from Hydra config."""
    
    # Set device
    device = cfg.device if torch.cuda.is_available() else "cpu"
    
    # Create data loader
    data_loader = SceneDataLoader(
        root=cfg.data.root,
        batch_size=cfg.data.batch_size,
        dataset_size=cfg.data.dataset_size,
        sample_with_replacement=cfg.data.sample_with_replacement,
        shuffle=cfg.data.shuffle,
    )
    
    # Instantiate env_config - this creates the proper EnvConfig dataclass
    env_config = instantiate(cfg.env)
    
    # Handle integration overrides by updating the instantiated EnvConfig
    smart_cfg = None
    smart_pkl_root = None
    use_smart_reward = False
    
    # Check if SMART integration exists and is enabled, #TODO modify this to use hydra for flexible
    if hasattr(cfg.integrations, 'smart'):
        # Update the EnvConfig fields
        env_config.use_smart_reward = True
        env_config.smart_pkl_root = cfg.integrations.smart.pkl_root
        env_config.smart_model_path = cfg.integrations.smart.model_ckpt
        
        # Prepare SMART config for the environment
        smart_cfg = cfg.integrations.smart
        smart_pkl_root = cfg.integrations.smart.pkl_root
        use_smart_reward = True
        print("✓ SMART integration enabled")
    
    # Check if VBD integration exists and is enabled  
    if hasattr(cfg.integrations, 'vbd'):
        # Update the EnvConfig fields
        env_config.use_vbd = True
        env_config.vbd_model_path = cfg.integrations.vbd.model_path
        env_config.vbd_trajectory_weight = cfg.integrations.vbd.trajectory_weight
        env_config.vbd_in_obs = cfg.integrations.vbd.in_obs
        env_config.init_steps = max(env_config.init_steps, cfg.integrations.vbd.min_init_steps)
        print("✓ VBD integration enabled")
    
    # Instantiate render config - this creates the proper RenderConfig dataclass
    render_config = instantiate(cfg.render)
    
    # Create environment with properly structured configs
    return GPUDriveTorchEnv(
        config=env_config,  # This is now the proper EnvConfig dataclass
        data_loader=data_loader,
        max_cont_agents=cfg.max_cont_agents,
        device=device,
        render_config=render_config,  # This is now the proper RenderConfig dataclass
        backend=cfg.backend,
        smart_pkl_root=smart_pkl_root,
        use_smart_reward=use_smart_reward,
        smart_cfg=smart_cfg,
    ) 