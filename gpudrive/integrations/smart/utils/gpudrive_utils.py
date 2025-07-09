# utils/rollout.py
import torch
from torch_geometric.data import HeteroData
from torch import Tensor

@torch.no_grad()
def update_rollout_batch(
    batch    : HeteroData,
    rollout    : Tensor,          # (A, T, 3)  – x, y, heading
    dt       : float = 0.1,     # timestep between frames   (s)
    replace_t: int = 11,        # timestep to start replacing from
) -> None:
    """
    Vector-ised, in-place replacement of the agent trajectory in `batch`.

    The function assumes *agent ordering* in `rollout` matches
    `batch['agent']` (same IDs).  If not, reorder first.

    Parameters
    ----------
    batch : HeteroData
        Graph batch from your dataloader (map, ptr, edges already set).
    rollout : Tensor (A,T,3)
        Channels: 0=x, 1=y, 2=heading (rad). May contain NaNs.
    dt    : float
        Time resolution used to derive velocity with Δx / dt.
    replace_t : int
        Timestep to start replacing from. Data before this timestep is preserved.
    """

    dev  = batch["agent"]["position"].device
    rollout = rollout.to(dev)                       # (A,T,3)

    # ------------------------------------------------------------------ #
    # 1.  Split + sanitise NaNs                                           #
    # ------------------------------------------------------------------ #
    pos_xy  = rollout[..., :2]                    # (A,T,2)
    heading = rollout[...,  2]                    # (A,T)

    nan_msk = torch.isnan(pos_xy[..., 0]) | torch.isnan(pos_xy[..., 1])
    pos_xy  = pos_xy.masked_fill(nan_msk.unsqueeze(-1), 0.0)
    heading = heading.masked_fill(nan_msk, 0.0)          # keep tensor aligned

    valid_mask = ~nan_msk                                    # (A,T) bool

    # ------------------------------------------------------------------ #
    # 2.  Velocity: forward diff, v[0]=0                                 #
    # ------------------------------------------------------------------ #
    vel = torch.zeros_like(pos_xy)                           # (A,T,2)
    if pos_xy.size(1) > 1:
        vel[:, 1:] = (pos_xy[:, 1:] - pos_xy[:, :-1]) / dt
    vel = vel.masked_fill(~valid_mask.unsqueeze(-1), 0.0)

    # ------------------------------------------------------------------ #
    # 3.  Preserve original data before replace_t, then replace from replace_t onwards #
    # ------------------------------------------------------------------ #
    
    # Get original data from batch
    orig_pos = batch["agent"]["position"][:, :, :2]  # (A,T,2) - remove z coordinate
    orig_heading = batch["agent"]["heading"]         # (A,T)
    orig_vel = batch["agent"]["velocity"]            # (A,T,2)
    orig_valid = batch["agent"]["valid_mask"]        # (A,T)
    
    # Create new tensors that preserve original data before replace_t
    new_pos_xy = orig_pos.clone()
    new_heading = orig_heading.clone()
    new_vel = orig_vel.clone()
    new_valid = orig_valid.clone()
    
    # Replace from replace_t onwards
    if replace_t < pos_xy.size(1):
        new_pos_xy[:, replace_t:] = pos_xy[:, replace_t:]
        new_heading[:, replace_t:] = heading[:, replace_t:]
        new_vel[:, replace_t:] = vel[:, replace_t:]
        new_valid[:, replace_t:] = valid_mask[:, replace_t:]

    # ------------------------------------------------------------------ #
    # 4.  Assign to graph (pad a zero-z coordinate)                      #
    # ------------------------------------------------------------------ #
    batch["agent"]["position"]   = torch.cat(
        [new_pos_xy, new_pos_xy.new_zeros(*new_pos_xy.shape[:-1], 1)], dim=-1
    )                                                             # (A,T,3)
    batch["agent"]["heading"]    = new_heading                    # (A,T)
    batch["agent"]["velocity"]   = new_vel                        # (A,T,2)
    batch["agent"]["valid_mask"] = new_valid                      # (A,T)
