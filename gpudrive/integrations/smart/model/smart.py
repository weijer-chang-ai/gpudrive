# Not a contribution
# Changes made by NVIDIA CORPORATION & AFFILIATES enabling <CAT-K> or otherwise documented as
# NVIDIA-proprietary are not a contribution and subject to the following terms and conditions:
# SPDX-FileCopyrightText: Copyright (c) <year> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

import math
from pathlib import Path

import hydra
import torch
from lightning import LightningModule
from torch.optim.lr_scheduler import LambdaLR

from gpudrive.integrations.smart.metrics import (
    CrossEntropy,
    TokenCls,
    # WOSACMetrics,
    # WOSACSubmission,
    minADE,
)
from gpudrive.integrations.smart.modules.smart_decoder import SMARTDecoder
from gpudrive.integrations.smart.tokens.token_processor import TokenProcessor
from gpudrive.integrations.smart.utils.finetune import set_model_for_finetuning
# Note: These imports from src.utils may need to be handled separately as they're not in the current integration
# from src.utils.vis_waymo import VisWaymo
# from src.utils.wosac_utils import get_scenario_id_int_tensor, get_scenario_rollouts

# import torch.nn.functional as F

class SMART(LightningModule):

    def __init__(self, model_config) -> None:
        super(SMART, self).__init__()
        self.save_hyperparameters()
        self.lr = model_config.lr
        self.lr_warmup_steps = model_config.lr_warmup_steps
        self.lr_total_steps = model_config.lr_total_steps
        self.lr_min_ratio = model_config.lr_min_ratio
        self.num_historical_steps = model_config.decoder.num_historical_steps
        self.log_epoch = -1
        self.val_open_loop = model_config.val_open_loop
        self.val_closed_loop = model_config.val_closed_loop
        self.token_processor = TokenProcessor(**model_config.token_processor)

        self.encoder = SMARTDecoder(
            **model_config.decoder, n_token_agent=self.token_processor.n_token_agent
        )
        set_model_for_finetuning(self.encoder, model_config.finetune)

        self.minADE = minADE()
        self.TokenCls = TokenCls(max_guesses=5)
        # self.wosac_metrics = WOSACMetrics("val_closed")
        # self.wosac_submission = WOSACSubmission(**model_config.wosac_submission)
        self.training_loss = CrossEntropy(model_config.training_loss.use_gt_raw,
         model_config.training_loss.gt_thresh_scale_length,
          model_config.training_loss.label_smoothing,
          rollout_as_gt = True
          )
        
        self.n_rollout_closed_val = model_config.n_rollout_closed_val
        self.n_vis_batch = model_config.n_vis_batch
        self.n_vis_scenario = model_config.n_vis_scenario
        self.n_vis_rollout = model_config.n_vis_rollout
        self.n_batch_wosac_metric = model_config.n_batch_wosac_metric

        # self.video_dir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
        # self.video_dir = Path(self.video_dir) / "videos"

        self.training_rollout_sampling = model_config.training_rollout_sampling
        self.validation_rollout_sampling = model_config.validation_rollout_sampling

    def likelihood(self, data_batch, world_states, dt = 0.1, replace_t = 11):
        """
        Parameters
        ----------
        data_batch['agent'].id      (N,)  int64  – agent-ids present in the graph
        data_batch['agent'].batch   (N,)  int64  – world index 0..W-1 of each node
        world_states                [W, A, T, F]  – last feature F-1 is the agent-id

        Returns
        -------
        like_world   Tensor [W, A, *]   (zeros where agent absent)
        """
        dev         = world_states.device
        W, A, T, F  = world_states.shape                 # F includes agent-id

        # ------------------------------------------------------------------ #
        # 1.  world-slot lookup table  (agent-id ⇒ slot index within world)
        # ------------------------------------------------------------------ #
        agent_slotid = world_states[:, :, 0, -1].long()  # [W, A] – id table (t=0)

        agent        = data_batch['agent']
        node_id      = agent.id.to(dev).long()           # (N,)
        node_world   = agent.batch.to(dev).long()        # (N,)

        id_in_world  = agent_slotid[node_world]          # (N, A)
        match        = id_in_world.eq(node_id[:, None])  # (N, A) bool
        found        = match.any(1)                      # (N,) which nodes exist
        slot_idx     = torch.where(
            found,
            match.float().argmax(1),                     # 0..A-1 for valid nodes
            torch.full_like(node_id, -1))                # -1 for missing nodes

        # ------------------------------------------------------------------ #
        # 2.  create full [N, T, F-1] tensor (zeros by default)
        # ------------------------------------------------------------------ #
        node_states = world_states.new_zeros(len(node_id), T, F-1)  # (N, T, F-1)

        if found.any():                                   # copy only real ones
            v         = torch.nonzero(found, as_tuple=False).squeeze(1)  # (M,)
            v_world   = node_world[v]
            v_slot    = slot_idx[v]
            node_states[v] = world_states[v_world, v_slot, :, :-1]       # (M,T,F-1)

        # ------------------------------------------------------------------ #
        # 3.  run the encoder – it sees the *full* batch and can mask itself
        # ------------------------------------------------------------------ #
        ##deubg

        ### replace data_batch with our rollout

        with torch.no_grad():
            #TODO how to make this part memory efficient?
            gt_pos = data_batch["agent"].position.clone()
            tok_map, tok_agent = self.token_processor(data_batch)
            results = self.encoder(tok_map, tok_agent)  
            orgiinal_logits = results["next_token_logits"]
            original_gt_idx = tok_agent["gt_idx"][:,2:]
            original_valid = results["next_token_valid"]  # Add this line
            
            from gpudrive.integrations.smart.utils.gpudrive_utils import update_rollout_batch
            update_rollout_batch(data_batch, node_states, dt, replace_t)
            tok_map, tok_agent = self.token_processor(data_batch)
            results = self.encoder(tok_map, tok_agent)        # (N, …)
            
            ## curreently the world_staes in 10Hz but likelihood 2Hz
            like_node_dist = results["next_token_logits"] #N, 16, vocab_size,
            next_token_valid = results["next_token_valid"] #N, 16
            
            LARGE_NEG = -1e9
            # Mask invalid tokens by setting their logits to -inf
            like_node_dist[~next_token_valid] = LARGE_NEG  # Set invalid positions to -inf
            
            ## based on logits obtain the likelihod of the executued index 
            gt_idx = tok_agent["gt_idx"][:,2:] #N, 16, 2
            log_p = torch.nn.functional.log_softmax(like_node_dist, dim=2)
            ## clamp 
            log_p   = torch.clamp(log_p, min=-20) #clamp the reward

            likelihood_node = log_p.gather(dim=2, index=gt_idx.unsqueeze(-1)).squeeze(-1) #N, 16, 2
            
            
            # Also mask the original logits for fair comparison
            orgiinal_logits[~original_valid] = LARGE_NEG
            
            ##TODO: may need to calculate exploration term
        
        if os.getenv("DEBUG", "FALSE") == "TRUE":
            like_prob = torch.nn.functional.softmax(like_node_dist, dim=2)
            like_prob_gt = torch.nn.functional.softmax(orgiinal_logits, dim=2)
            like_prob_gather = like_prob.gather(dim=2, index=gt_idx.unsqueeze(-1)).squeeze(-1)
            like_prob_gather_original = like_prob_gt.gather(dim=2, index=original_gt_idx.unsqueeze(-1)).squeeze(-1)
            ## should expand 
            like_prob_gather = like_prob_gather.repeat(1,5)
            like_prob_gather_original = like_prob_gather_original.repeat(1,5) #get 80
            #append 11 to the start of each row
            like_prob_gather = torch.cat([torch.ones_like(like_prob_gather[:,0:11]), like_prob_gather], dim=1)
            like_prob_gather_original = torch.cat([torch.ones_like(like_prob_gather_original[:,0:11]), like_prob_gather_original], dim=1)
            data_batch["agent"].position = gt_pos
            # viz_scenes(data_batch, node_states, like_prob_gather, like_prob_gather_original)
            viz_scenes_compare(data_batch, node_states, like_prob_gather, like_prob_gather_original)


        T_target   = world_states.size(2) - replace_t                # e.g. 80
        if likelihood_node.size(1) < T_target:
            assert T_target % likelihood_node.size(1) == 0, "T_target must be divisible by likelihood_node.size(1)"
            dup_factor = T_target // likelihood_node.size(1)
            likelihood_node = likelihood_node.repeat(1, dup_factor) / dup_factor #N, 16, vocab_size

        # ------------------------------------------------------------------ #
        # 4.  scatter back to padded [W, A, …]   (zeros stay for gaps)
        # ------------------------------------------------------------------ #
        like_world = torch.zeros_like(world_states[...,0])

        if found.any():
            like_world[node_world[found], slot_idx[found],replace_t:] = likelihood_node[found]


        return like_world


    
    def training_step(self, data, batch_idx):
        tokenized_map, tokenized_agent = self.token_processor(data)
        if self.training_rollout_sampling.num_k <= 0:
            pred = self.encoder(tokenized_map, tokenized_agent)
        else:
            pred = self.encoder.inference(
                tokenized_map,
                tokenized_agent,
                sampling_scheme=self.training_rollout_sampling,
            )

        loss = self.training_loss(
            **pred,
            token_agent_shape=tokenized_agent["token_agent_shape"],  # [n_agent, 2]
            token_traj=tokenized_agent["token_traj"],  # [n_agent, n_token, 4, 2]
            train_mask=data["agent"]["train_mask"],  # [n_agent]
            current_epoch=self.current_epoch,
        )
        self.log("train/loss", loss, on_step=True, batch_size=1)

        return loss

    def validation_step(self, data, batch_idx):
        tokenized_map, tokenized_agent = self.token_processor(data)

        # ! open-loop vlidation
        if self.val_open_loop:
            pred = self.encoder(tokenized_map, tokenized_agent)
            loss = self.training_loss(
                **pred,
                token_agent_shape=tokenized_agent["token_agent_shape"],  # [n_agent, 2]
                token_traj=tokenized_agent["token_traj"],  # [n_agent, n_token, 4, 2]
            )

            self.TokenCls.update(
                # action that goes from [(10->15), ..., (85->90)]
                pred=pred["next_token_logits"],  # [n_agent, 16, n_token]
                pred_valid=pred["next_token_valid"],  # [n_agent, 16]
                target=tokenized_agent["gt_idx"][:, 2:],
                target_valid=tokenized_agent["valid_mask"][:, 2:],
            )
            self.log(
                "val_open/acc",
                self.TokenCls,
                on_epoch=True,
                sync_dist=True,
                batch_size=1,
            )
            self.log("val_open/loss", loss, on_epoch=True, sync_dist=True, batch_size=1)

        # ! closed-loop vlidation
        if self.val_closed_loop:
            pred_traj, pred_z, pred_head = [], [], []
            for _ in range(self.n_rollout_closed_val):
                pred = self.encoder.inference(
                    tokenized_map, tokenized_agent, self.validation_rollout_sampling
                )
                pred_traj.append(pred["pred_traj_10hz"])
                pred_z.append(pred["pred_z_10hz"])
                pred_head.append(pred["pred_head_10hz"])

            pred_traj = torch.stack(pred_traj, dim=1)  # [n_ag, n_rollout, n_step, 2]
            pred_z = torch.stack(pred_z, dim=1)  # [n_ag, n_rollout, n_step]
            pred_head = torch.stack(pred_head, dim=1)  # [n_ag, n_rollout, n_step]

            # ! WOSAC
            scenario_rollouts = None
            if self.wosac_submission.is_active:  # ! save WOSAC submission
                self.wosac_submission.update(
                    scenario_id=data["scenario_id"],
                    agent_id=data["agent"]["id"],
                    agent_batch=data["agent"]["batch"],
                    pred_traj=pred_traj,
                    pred_z=pred_z,
                    pred_head=pred_head,
                    global_rank=self.global_rank,
                )
                _gpu_dict_sync = self.wosac_submission.compute()
                if self.global_rank == 0:
                    for k in _gpu_dict_sync.keys():  # single gpu fix
                        if type(_gpu_dict_sync[k]) is list:
                            _gpu_dict_sync[k] = _gpu_dict_sync[k][0]
                    scenario_rollouts = get_scenario_rollouts(**_gpu_dict_sync)
                    self.wosac_submission.aggregate_rollouts(scenario_rollouts)
                self.wosac_submission.reset()

            else:  # ! compute metrics, disable if save WOSAC submission
                self.minADE.update(
                    pred=pred_traj,
                    target=data["agent"]["position"][
                        :, self.num_historical_steps :, : pred_traj.shape[-1]
                    ],
                    target_valid=data["agent"]["valid_mask"][
                        :, self.num_historical_steps :
                    ],
                )

                # WOSAC metrics
                if batch_idx < self.n_batch_wosac_metric:
                    device = pred_traj.device
                    scenario_rollouts = get_scenario_rollouts(
                        scenario_id=get_scenario_id_int_tensor(
                            data["scenario_id"], device
                        ),
                        agent_id=data["agent"]["id"],
                        agent_batch=data["agent"]["batch"],
                        pred_traj=pred_traj,
                        pred_z=pred_z,
                        pred_head=pred_head,
                    )
                    self.wosac_metrics.update(data["tfrecord_path"], scenario_rollouts)

            # ! visualization
            if self.global_rank == 0 and batch_idx < self.n_vis_batch:
                if scenario_rollouts is not None:
                    for _i_sc in range(self.n_vis_scenario):
                        _vis = VisWaymo(
                            scenario_path=data["tfrecord_path"][_i_sc],
                            save_dir=self.video_dir
                            / f"batch_{batch_idx:02d}-scenario_{_i_sc:02d}",
                        )
                        _vis.save_video_scenario_rollout(
                            scenario_rollouts[_i_sc], self.n_vis_rollout
                        )
                        for _path in _vis.video_paths:
                            self.logger.log_video(
                                "/".join(_path.split("/")[-3:]), [_path]
                            )

    def on_validation_epoch_end(self):
        if self.val_closed_loop:
            if not self.wosac_submission.is_active:
                epoch_wosac_metrics = self.wosac_metrics.compute()
                epoch_wosac_metrics["val_closed/ADE"] = self.minADE.compute()
                if self.global_rank == 0:
                    epoch_wosac_metrics["epoch"] = (
                        self.log_epoch if self.log_epoch >= 0 else self.current_epoch
                    )
                    self.logger.log_metrics(epoch_wosac_metrics)

                self.wosac_metrics.reset()
                self.minADE.reset()

            if self.global_rank == 0:
                if self.wosac_submission.is_active:
                    self.wosac_submission.save_sub_file()

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr)

        def lr_lambda(current_step):
            current_step = self.current_epoch + 1
            if current_step < self.lr_warmup_steps:
                return (
                    self.lr_min_ratio
                    + (1 - self.lr_min_ratio) * current_step / self.lr_warmup_steps
                )
            return self.lr_min_ratio + 0.5 * (1 - self.lr_min_ratio) * (
                1.0
                + math.cos(
                    math.pi
                    * min(
                        1.0,
                        (current_step - self.lr_warmup_steps)
                        / (self.lr_total_steps - self.lr_warmup_steps),
                    )
                )
            )

        lr_scheduler = LambdaLR(optimizer, lr_lambda=lr_lambda)
        return [optimizer], [lr_scheduler]

    def test_step(self, data, batch_idx):
        tokenized_map, tokenized_agent = self.token_processor(data)

        # ! only closed-loop vlidation
        pred_traj, pred_z, pred_head = [], [], []
        for _ in range(self.n_rollout_closed_val):
            pred = self.encoder.inference(
                tokenized_map, tokenized_agent, self.validation_rollout_sampling
            )
            pred_traj.append(pred["pred_traj_10hz"])
            pred_z.append(pred["pred_z_10hz"])
            pred_head.append(pred["pred_head_10hz"])

        pred_traj = torch.stack(pred_traj, dim=1)  # [n_ag, n_rollout, n_step, 2]
        pred_z = torch.stack(pred_z, dim=1)  # [n_ag, n_rollout, n_step]
        pred_head = torch.stack(pred_head, dim=1)  # [n_ag, n_rollout, n_step]

        # ! WOSAC submission save
        self.wosac_submission.update(
            scenario_id=data["scenario_id"],
            agent_id=data["agent"]["id"],
            agent_batch=data["agent"]["batch"],
            pred_traj=pred_traj,
            pred_z=pred_z,
            pred_head=pred_head,
            global_rank=self.global_rank,
        )
        _gpu_dict_sync = self.wosac_submission.compute()
        if self.global_rank == 0:
            for k in _gpu_dict_sync.keys():  # single gpu fix
                if type(_gpu_dict_sync[k]) is list:
                    _gpu_dict_sync[k] = _gpu_dict_sync[k][0]
            scenario_rollouts = get_scenario_rollouts(**_gpu_dict_sync)
            self.wosac_submission.aggregate_rollouts(scenario_rollouts)
        self.wosac_submission.reset()

    def on_test_epoch_end(self):
        if self.global_rank == 0:
            self.wosac_submission.save_sub_file()

import os, matplotlib.pyplot as plt, numpy as np, torch
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable

def plot_colormap_line(ax, xy: np.ndarray, lik: np.ndarray, cmap=plt.cm.plasma, lw=2, linestyle='-'):
    """
    Plot a trajectory as a colored line, where each segment's color encodes its likelihood.

    Parameters
    ----------
    ax      : matplotlib Axes
    xy      : (T,2) array of trajectory points
    lik     : (T,) array of per-step likelihoods in [0,1]
    cmap    : colormap to use
    lw      : line width
    linestyle: style of the line segments
    """
    # Build line segments: shape (T-1, 2, 2)
    segments = np.stack([xy[:-1], xy[1:]], axis=1)
    # Normalize likelihoods to [0,1]
    norm = Normalize(vmin=0.0, vmax=1.0)
    colors = cmap(norm(lik[:-1]))
    # Create collection
    lc = LineCollection(segments, colors=colors, linewidths=lw, linestyle=linestyle)
    ax.add_collection(lc)
    # Return mappable for colorbar
    sm = ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array(lik)
    return sm


def viz_scenes(data_batch, node_states=None, likelihood_prob=None, 
               gt_likelihood_prob=None, save_dir="videos", max_scenes=50):
    """
    Write one PNG per scene in `data_batch`, coloring trajectories by per-step likelihood.
    """
    os.makedirs(save_dir, exist_ok=True)

    ptr        = data_batch["pt_token"].ptr.cpu()
    map_traj   = data_batch["map_save"].traj_pos.cpu().numpy()
    map_xy     = map_traj.reshape(-1, 2)
    ptr_exp    = ptr * 3
    W          = ptr.numel() - 1
    scenes     = range(min(W, max_scenes))

    use_graph = "position" in data_batch["agent"]
    if use_graph:
        ag_pos   = data_batch["agent"].position.cpu().numpy()
        ag_batch = data_batch["agent"].batch.cpu().numpy()
        ag_id    = data_batch["agent"].id.cpu().numpy()

    use_ns = node_states is not None
    if use_ns:
        ns_xy    = node_states.cpu().numpy()[..., :2]
        ns_batch = data_batch['agent'].batch.cpu().numpy()
        ns_id    = data_batch['agent'].id.cpu().numpy()

    use_pred = likelihood_prob is not None
    if use_pred:
        pred_lik = likelihood_prob.cpu().numpy()
        if pred_lik.ndim > 2:
            pred_lik = pred_lik.mean(axis=-1)

    use_gt = gt_likelihood_prob is not None
    if use_gt:
        gt_lik = gt_likelihood_prob.cpu().numpy()
        if gt_lik.ndim > 2:
            gt_lik = gt_lik.mean(axis=-1)

    for s in scenes:
        fig, ax = plt.subplots(figsize=(12, 6))
        # plot map
        xy = map_xy[ptr_exp[s]:ptr_exp[s+1]]
        if xy.size:
            non_zero = ~((xy[:,0]==0)&(xy[:,1]==0))
            ax.scatter(xy[non_zero,0], xy[non_zero,1], c="lightgray", s=0.5, zorder=1)

        # ground truth trajectories
        if use_graph:
            for i, track in enumerate(ag_pos[ag_batch==s]):
                color_idx = ag_id[ag_batch==s][i] % 20
                xy_track = track[:, :2]
                valid = ~(np.all(xy_track==0, axis=1))
                if valid.any():
                    if use_gt:
                        liks = gt_lik[ag_batch==s][i]
                        sm = plot_colormap_line(ax, xy_track[valid], liks[valid], lw=3, linestyle='-')
                    else:
                        ax.plot(xy_track[valid,0], xy_track[valid,1], c=plt.cm.tab20(color_idx), \
                                lw=3, linestyle='-', alpha=0.7)

        # predicted trajectories
        if use_ns:
            for i, traj in enumerate(ns_xy[ns_batch==s]):
                color_idx = ns_id[ns_batch==s][i] % 20
                valid = ~(np.all(traj==0, axis=1))
                if valid.any():
                    if use_pred:
                        liks = pred_lik[ns_batch==s][i]
                        sm_pred = plot_colormap_line(ax, traj[valid], liks[valid], lw=2, linestyle='--')
                    else:
                        ax.plot(traj[valid,0], traj[valid,1], c=plt.cm.tab20(color_idx), \
                                lw=2, linestyle='--', alpha=0.8)

        # colorbar
        if use_pred or use_gt:
            sm_final = sm_pred if use_pred else sm
            cbar = plt.colorbar(sm_final, ax=ax, fraction=0.046, pad=0.04)
            cbar.set_label('Step Likelihood', rotation=270, labelpad=20)

        ax.set_aspect('equal')
        ax.set_title(f"Scene {s}")
        out = os.path.join(save_dir, f"scene_{s:03d}.png")
        fig.savefig(out, dpi=120, bbox_inches='tight')
        plt.close(fig)
        print(f"✓ saved {out}")

    print(f"Done – wrote {len(list(scenes))} scene images to '{save_dir}'")
import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable

def plot_colormap_line(ax, xy: np.ndarray, lik: np.ndarray, cmap=plt.cm.plasma, lw=2, linestyle='-'):
    """
    Plot a trajectory as a colored line where each segment's color encodes its likelihood.
    Returns a ScalarMappable for a shared colorbar.
    """
    segments = np.stack([xy[:-1], xy[1:]], axis=1)
    norm = Normalize(vmin=0.0, vmax=1.0)
    colors = cmap(norm(lik[:-1]))
    lc = LineCollection(segments, colors=colors, linewidths=lw, linestyle=linestyle)
    ax.add_collection(lc)
    sm = ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array(lik)
    return sm




def viz_scenes_compare(data_batch, node_states=None, likelihood_prob=None,
                       gt_likelihood_prob=None, save_dir="videos", max_scenes=50):
    """Visualise each scene with predicted vs. ground‑truth trajectories side by side."""
    os.makedirs(save_dir, exist_ok=True)
    ##TODO, should also count non_vehicles to invalid

    ptr       = data_batch["pt_token"].ptr.cpu()
    map_xy    = data_batch["map_save"].traj_pos.cpu().numpy().reshape(-1, 2)
    ptr_exp   = ptr * 3
    scenes    = range(min(ptr.numel() - 1, max_scenes))

    ag_batch  = data_batch['agent'].batch.cpu().numpy()
    ag_valid  = data_batch['agent']['valid_mask'].cpu().numpy().astype(bool)  # (N_ag, T)

    # predicted
    use_pred = node_states is not None and likelihood_prob is not None
    if use_pred:
        ns_xy   = node_states.cpu().numpy()[..., :2]
        pred_lik_all = likelihood_prob.cpu().numpy()
        if pred_lik_all.ndim > 2:
            pred_lik_all = pred_lik_all.mean(axis=-1)

    # ground truth
    use_gt = 'position' in data_batch['agent'] and gt_likelihood_prob is not None
    if use_gt:
        ag_pos = data_batch['agent'].position.cpu().numpy()[..., :2]
        gt_lik_all = gt_likelihood_prob.cpu().numpy()
        if gt_lik_all.ndim > 2:
            gt_lik_all = gt_lik_all.mean(axis=-1)

    for s in scenes:
        fig, (ax_pred, ax_gt) = plt.subplots(1, 2, figsize=(14, 6), sharex=True, sharey=True)
        fig.suptitle(f"Scene {s}")

        # map background
        xy_map = map_xy[ptr_exp[s]:ptr_exp[s+1]]
        mask_map = ~((xy_map[:, 0] == 0) & (xy_map[:, 1] == 0))
        for ax in (ax_pred, ax_gt):
            if mask_map.any():
                ax.scatter(xy_map[mask_map, 0], xy_map[mask_map, 1], c='lightgray', s=0.5, zorder=1)

        coords = []
        # predicted
        pred_liks_scene = []
        if use_pred:
            for idx in np.where(ag_batch == s)[0]:
                traj, liks, valid = ns_xy[idx], pred_lik_all[idx], ag_valid[idx]
                mask = valid & ~(np.all(traj == 0, axis=1))
                if mask.any():
                    plot_colormap_line(ax_pred, traj[mask], liks[mask], lw=2, linestyle='--')
                    coords.append(traj[mask]); pred_liks_scene.append(liks[mask])
            m_pred = np.nanmean(np.concatenate(pred_liks_scene)) if pred_liks_scene else np.nan
            ax_pred.set_title(f"Predicted\nmean lik {m_pred:.3f}" if pred_liks_scene else "Predicted")
        else:
            ax_pred.set_visible(False)

        # ground truth
        gt_liks_scene = []
        if use_gt:
            for idx in np.where(ag_batch == s)[0]:
                traj, liks, valid = ag_pos[idx], gt_lik_all[idx], ag_valid[idx]
                mask = valid & ~(np.all(traj == 0, axis=1))
                if mask.any():
                    plot_colormap_line(ax_gt, traj[mask], liks[mask], lw=3, linestyle='-')
                    coords.append(traj[mask]); gt_liks_scene.append(liks[mask])
            m_gt = np.nanmean(np.concatenate(gt_liks_scene)) if gt_liks_scene else np.nan
            ax_gt.set_title(f"Ground Truth\nmean lik {m_gt:.3f}" if gt_liks_scene else "Ground Truth")
        else:
            ax_gt.set_visible(False)

        # axis limits
        if coords:
            pts = np.vstack(coords)
            xmean, ymean = np.nanmean(pts[:, 0]), np.nanmean(pts[:, 1])
            xmin, xmax = xmean - 50, xmean + 50
            ymin, ymax = ymean - 50, ymean + 50
            for ax in (ax_pred, ax_gt):
                ax.set_xlim(xmin, xmax); ax.set_ylim(ymin, ymax); ax.set_aspect('equal')

        # shared colorbar
        sm = ScalarMappable(norm=Normalize(0, 1), cmap=plt.cm.plasma); sm.set_array([])
        fig.colorbar(sm, ax=(ax_pred, ax_gt), fraction=0.046, pad=0.04).set_label('Step Likelihood', rotation=270, labelpad=20)

        out = os.path.join(save_dir, f'scene_compare_{s:03d}.png')
        fig.savefig(out, dpi=120, bbox_inches='tight'); plt.close(fig)
        print(f"✓ saved {out}")

    print(f"Done – wrote {len(list(scenes))} scene comparison figures to '{save_dir}'")
