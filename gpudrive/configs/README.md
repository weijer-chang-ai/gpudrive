# GPUDrive Hydra Configuration

This directory contains the Hydra configuration system for GPUDrive. 

## Structure

```
configs/
├── config.yaml                 # Main configuration (no integrations)
├── base_config.yaml           # Base config for inheritance
├── env/                       # Environment configurations
│   ├── classic.yaml
│   └── delta_local.yaml
├── render/                    # Rendering configurations
│   ├── matplotlib.yaml
│   └── madrona_rgb.yaml
├── scene/                     # Scene loading configurations
│   └── default.yaml
├── integrations/              # Integration-specific configurations
│   ├── none.yaml              # No integrations
│   ├── smart.yaml             # SMART configuration
│   ├── vbd.yaml               # VBD configuration
│   └── smart_model.yaml       # SMART model architecture
└── examples/                  # Example configurations
    ├── smart_only.yaml
    ├── vbd_only.yaml
    └── smart_and_vbd.yaml
```

## Usage

### Basic Usage (No Integrations)
```bash
python examples/simple_hydra_usage.py
```

### Enable SMART Only
```bash
python examples/simple_hydra_usage.py +integrations/smart=smart
```

### Enable VBD Only
```bash
python examples/simple_hydra_usage.py +integrations/vbd=vbd
```

### Enable Both Integrations
```bash
python examples/simple_hydra_usage.py +integrations/smart=smart +integrations/vbd=vbd
```

### Use Example Configs
```bash
# SMART only with optimized settings
python examples/simple_hydra_usage.py --config-name=examples/smart_only

# VBD only with optimized settings  
python examples/simple_hydra_usage.py --config-name=examples/vbd_only

# Both integrations with optimized settings
python examples/simple_hydra_usage.py --config-name=examples/smart_and_vbd
```

### Override Individual Settings
```bash
# Change environment dynamics
python examples/simple_hydra_usage.py env=delta_local

# Change device
python examples/simple_hydra_usage.py device=cpu

# Override integration settings
python examples/simple_hydra_usage.py +integrations/smart=smart integrations.smart.weight=0.2

# Multiple overrides
python examples/simple_hydra_usage.py env=delta_local device=cpu data.batch_size=4
```

## How It Works

### Integration Control
- **No integrations**: Default behavior, only base environment
- **Enable integration**: Add `+integrations/<name>=<config>` to command line or defaults
- **Integration files**: Only contain configuration parameters, not enable/disable flags
- **Main config**: Controls which integrations are loaded through defaults list

### Key Principles
1. Each integration config file only contains settings for that integration
2. Enabling/disabling is controlled by including/excluding in defaults or command line
3. Integration configs don't have `enabled` fields - if loaded, they're enabled
4. Independent integrations can be mixed and matched freely

### Creating New Integrations
1. Create `integrations/my_integration.yaml` with configuration parameters
2. Update `hydra_utils.py` to check for `hasattr(cfg.integrations, 'my_integration')`
3. Add integration logic in the environment creation
4. Create example configs showing how to use it

## Examples in Code

```python
import hydra
from gpudrive.utils.hydra_utils import create_env_from_cfg

@hydra.main(config_path="gpudrive/configs", config_name="config")
def main(cfg):
    env = create_env_from_cfg(cfg)
    # Environment automatically configured based on loaded integrations
``` 