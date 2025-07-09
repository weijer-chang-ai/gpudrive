"""
This implementation is adapted from the demo in PufferLib by Joseph Suarez,
which in turn is adapted from Costa Huang's CleanRL PPO + LSTM implementation.
Links
- PufferLib: https://github.com/PufferAI/PufferLib/blob/dev/demo.py
- Cleanrl: https://github.com/vwxyzjn/cleanrl/blob/master/cleanrl/ppo.py
"""

from pdb import set_trace as T
import numpy as np
import os
import random
import psutil
import time

from threading import Thread
from collections import defaultdict, deque

import torch

import pufferlib
import pufferlib.utils
import pufferlib.pytorch

torch.set_float32_matmul_precision("high")

# Fast Cython GAE implementation
import pyximport

pyximport.install(setup_args={"include_dirs": np.get_include()})

from c_gae import compute_gae
from gpudrive.integrations.puffer.logging import print_dashboard, abbreviate


def create(config, vecenv, policy, optimizer=None, wandb=None):
    seed_everything(config.seed, config.torch_deterministic)
    profile = Profile()
    losses = make_losses()

    utilization = Utilization()
    msg = f"Model Size: {abbreviate(count_params(policy))} parameters"
    if vecenv.use_vbd:
        msg += f" | Using VBD"
    print_dashboard(
        config.env, utilization, 0, 0, profile, losses, {}, msg, clear=True
    )

    vecenv.async_reset(config.seed)
    obs_shape = vecenv.single_observation_space.shape
    obs_dtype = vecenv.single_observation_space.dtype
    atn_shape = vecenv.single_action_space.shape
    total_agents = vecenv.num_agents

    # Log initial data coverage
    vecenv.wandb_obj = wandb
    vecenv.log_data_coverage()

    lstm = policy.lstm if hasattr(policy, "lstm") else None

    # Get action dimension for reference distribution storage
    if hasattr(vecenv.single_action_space, 'n'):
        action_dim = vecenv.single_action_space.n  # Discrete action space
    elif hasattr(vecenv.single_action_space, 'shape'):
        action_dim = vecenv.single_action_space.shape[0]  # Continuous action space
    else:
        action_dim = None

    # Rollout buffer
    experience = Experience(
        config.batch_size,
        config.bptt_horizon,
        config.minibatch_size,
        obs_shape,
        obs_dtype,
        atn_shape,
        config.cpu_offload,
        config.device,
        lstm,
        total_agents,
        controlled_agent_mask=vecenv.controlled_agent_mask,
        use_reference_loss=config.use_reference_loss,
        use_smart_likelihood=config.use_smart_likelihood,
        action_dim=action_dim,  # NEW: Pass action dimension
    )

    uncompiled_policy = policy

    if config.compile:
        policy = torch.compile(policy, mode=config.compile_mode)

    optimizer = torch.optim.Adam(
        policy.parameters(), lr=float(config.learning_rate), eps=1e-5
    )

    return pufferlib.namespace(
        config=config,
        vecenv=vecenv,
        policy=policy,
        uncompiled_policy=uncompiled_policy,
        optimizer=optimizer,
        experience=experience,
        profile=profile,
        losses=losses,
        wandb=wandb,
        global_step=0,
        global_step_pad=0,
        resample_buffer=0,
        resample_counter=0,
        epoch=0,
        stats={},
        infos=defaultdict(list),
        msg=msg,
        last_log_time=0,
        utilization=utilization,
    )


@pufferlib.utils.profile
def evaluate(data):

    # Sample new batch of scenarios before start of rollout
    if (
        data.config.resample_scenes
        and data.resample_buffer >= data.config.resample_interval
        and data.config.resample_dataset_size > data.vecenv.num_worlds
    ):
        print(f"Resampling scenarios at global step {data.global_step}")
        data.vecenv.resample_scenario_batch()
        data.experience.update_controlled_agent_mask(data.vecenv.controlled_agent_mask)
        data.resample_buffer = 0

    data.vecenv.clear_render_storage()

    config, profile, experience = data.config, data.profile, data.experience

    with profile.eval_misc:
        policy = data.policy
        lstm_h, lstm_c = experience.lstm_h, experience.lstm_c

    # Rollout loop
    while not experience.full:

        with profile.env:
            # Receive data from current timestep
            (
                obs,
                reward,
                terminal,
                truncated,
                info,
                env_id,
                scene_idx,
                timestep,
                mask,
            ) = data.vecenv.recv()

        with profile.eval_misc:
            total_alive = mask.sum().item()

            data.global_step += total_alive
            data.global_step_pad += data.vecenv.total_agents
            data.resample_buffer += total_alive

            data.vecenv.global_step = data.global_step
            data.vecenv.iters += 1

            obs_device = obs.to(config.device)

        with profile.eval_forward, torch.no_grad():
            if lstm_h is not None:
                h = lstm_h[:, env_id]
                c = lstm_c[:, env_id]
                actions, logprob, _, value, _, (h, c) = policy(obs_device, (h, c))
                lstm_h[:, env_id] = h
                lstm_c[:, env_id] = c
            else:
                actions, logprob, _, value, _ = policy(obs_device)

            if config.device == "cuda":
                torch.cuda.synchronize()

        with profile.env:
            # Step the environment and reset if done
            data.vecenv.send(actions)

        with profile.eval_misc:
            value = value.flatten()
            obs_device = obs_device if config.cpu_offload else obs_device

            # Use the terminal observation value to better estimate the reward
            # done_but_truncated = truncated & terminal
            # if done_but_truncated.any():
            #     terminal_obs = data.vecenv.last_obs[done_but_truncated]

            #     # Get terminal (truncated) observation value
            #     with torch.no_grad():
            #         _, _, _, terminal_value, _ = policy(terminal_obs)

            #     # Add discounted value to reward
            #     reward[done_but_truncated] += config.gamma * terminal_value.squeeze(-1)

            # Add to rollout buffer
            experience.store(
                obs_device,
                value,
                actions,
                logprob,
                reward,
                terminal,
                env_id,
                mask,
                scene_idx,
                timestep,
            )
            ## if lielhood is not None, assign likelihood reward
            pending_likelihood = data.vecenv.get_and_clear_pending_smart_likelihood()
            if pending_likelihood is not None:
                experience.assign_smart_likelihood_to_experiences(pending_likelihood)
            # Add metrics for logging
            for i in info:
                for k, v in pufferlib.utils.unroll_nested_dict(i):
                    data.infos[k].append(v)

        # Reference distribution computation moved to train() function

    with profile.eval_misc:
        data.stats = {}

        # Store the average across K done worlds across last N rollouts
        # ensure we are logging an unbiased estimate of the performance
        if sum(data.infos["num_completed_episodes"]) > data.config.log_window:
            for k, v in data.infos.items():
                try:
                    if "num_completed_episodes" in k:
                        data.stats[k] = np.sum(v)
                    else:
                        data.stats[k] = np.mean(v)

                    # Log variance for goal and collision metrics
                    if "goal" in k:
                        data.stats[f"std_{k}"] = np.std(v)
                except:
                    continue

            # Reset info dict
            data.infos = defaultdict(list)

    return data.stats, data.infos


@pufferlib.utils.profile
def train(data):
    config, profile, experience = data.config, data.profile, data.experience
    data.losses = make_losses()
    losses = data.losses

    with profile.train_misc:
        idxs = experience.sort_training_data()
        dones_np = experience.dones_np[idxs]
        values_np = experience.values_np[idxs]
        
        # Store original rewards before modification
        original_rewards_np = experience.rewards_np[idxs]
        rewards_np = original_rewards_np.copy()
        
        if experience.use_smart_likelihood:
            likelihood_np = experience.smart_likelihood_np[idxs]
            rewards_np = rewards_np + config.likelihood_weight * likelihood_np #plus or minus?
        else:
            likelihood_np = np.zeros_like(rewards_np)

        advantages_np = compute_gae(
            dones_np, values_np, rewards_np, config.gamma, config.gae_lambda
        ) #contiguous so each agents,scene advantages can be accumulated until
        
        # Compute reference distributions using executed actions
        if experience.use_reference_loss:
            # Extract executed actions in sorted order
            sorted_actions = experience.actions[idxs]  # Shape: (batch_size,)

            ##make the cube before we feed to reference distribution, do we want to do it here? Or everytime scene_idx finished

            reference_log_probs = data.vecenv.env.compute_reference_distribution(sorted_actions)
            
            # Store FULL reference distribution in experience buffer (2D)
            experience.reference_log_probs_np[:len(idxs)] = reference_log_probs.cpu().numpy()
        
        experience.flatten_batch(advantages_np)

    # Optimizing the policy and value network
    num_update_iters = config.update_epochs * experience.num_minibatches
    for epoch in range(config.update_epochs):
        lstm_state = None
        for mb in range(experience.num_minibatches):
            with profile.train_misc:
                obs = experience.b_obs[mb]
                obs = obs.to(config.device)
                atn = experience.b_actions[mb]
                log_probs = experience.b_logprobs[mb]
                val = experience.b_values[mb]
                adv = experience.b_advantages[mb]
                ret = experience.b_returns[mb] #add likielihood reward

            with profile.train_forward:
                if experience.lstm_h is not None:
                    _, newlogprob, entropy, newvalue, lstm_state = data.policy(
                        obs, state=lstm_state, action=atn
                    )
                    lstm_state = (
                        lstm_state[0].detach(),
                        lstm_state[1].detach(),
                    )
                else:
                    _, newlogprob, entropy, newvalue, new_logits = data.policy(
                        obs.reshape(
                            -1, *data.vecenv.single_observation_space.shape
                        ),
                        action=atn,
                    )

                if config.device == "cuda":
                    torch.cuda.synchronize()

            with profile.train_misc:
                logratio = newlogprob - log_probs.reshape(-1)
                ratio = logratio.exp()

                with torch.no_grad():
                    # calculate approx_kl http://joschu.net/blog/kl-approx.html
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfrac = (
                        ((ratio - 1.0).abs() > config.clip_coef).float().mean()
                    )

                adv = adv.reshape(-1)
                if config.norm_adv:
                    adv = (adv - adv.mean()) / (adv.std() + 1e-8)

                # Policy loss
                pg_loss1 = -adv * ratio
                pg_loss2 = -adv * torch.clamp(
                    ratio, 1 - config.clip_coef, 1 + config.clip_coef
                )
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # Value loss
                newvalue = newvalue.view(-1)
                if config.clip_vloss:
                    v_loss_unclipped = (newvalue - ret) ** 2
                    v_clipped = val + torch.clamp(
                        newvalue - val,
                        -config.vf_clip_coef,
                        config.vf_clip_coef,
                    )
                    v_loss_clipped = (v_clipped - ret) ** 2
                    v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                    v_loss = 0.5 * v_loss_max.mean()
                else:
                    v_loss = 0.5 * ((newvalue - ret) ** 2).mean()

                entropy_loss = entropy.mean()
                reference_loss = torch.tensor(0.0, device=config.device)
                if experience.use_reference_loss:
                    ref_log_probs = experience.b_reference_log_probs[mb]

                    logp_new  = torch.log_softmax(new_logits, dim=-1)
                    logp_old  = torch.log_softmax(ref_log_probs, dim=-1)
                    p_new     = logp_new.exp()
                    analytic_kl = (p_new * (logp_new - logp_old)).sum(-1).mean()
                    reference_loss = analytic_kl        
                    # ref_analytic_kl = (ref_log_probs_exp * (ref_log_probs - new_logits)).mean()

                    # calculate approx_kl http://joschu.net/blog/kl-approx.html
                    # ref_old_approx_kl = (-ref_logratio).mean()
                    # ref_approx_kl = ((ref_ratio - 1) - ref_logratio).mean()

                    # Get reference log probabilities for current minibatch
                    #TODO: design how to caluclate it 
                    # KL divergence: D_KL(policy || reference) = sum(p * log(p/q))
                    # = sum(policy_log_probs - reference_log_probs) * policy_probs
                    # Simplified: just penalize deviation from reference
                    # # Get full reference distribution for current minibatch
                    # ref_log_probs_full = experience.b_reference_log_probs[mb]  # Shape: (minibatch_size, action_dim)
                    
                    # # Now you have the full reference distribution available for sophisticated loss calculations
                    # # Example: You could compute KL divergence, extract specific actions, etc.
                    
                    # # Simple example - extract log probs for executed actions:
                    # atn_flat = atn.flatten()  # Shape: (minibatch_size,)
                    # ref_log_probs = ref_log_probs_full.gather(1, atn_flat.unsqueeze(1)).squeeze(1)
                    # reference_loss = -ref_log_probs.mean()  # Maximize reference likelihood
                    
                    # # Or compute full KL divergence with current policy distribution:
                    # # current_policy_logits = newlogprob  # Get full policy distribution
                    # # kl_loss = F.kl_div(current_policy_logits, ref_log_probs_full.exp(), reduction='batchmean')
                    # # reference_loss = kl_loss

                loss = (
                    pg_loss
                    - config.ent_coef * entropy_loss
                    + v_loss * config.vf_coef
                    + reference_loss * config.reference_loss_weight
                )

            with profile.learn:
                data.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    data.policy.parameters(), config.max_grad_norm
                )
                data.optimizer.step()
                if config.device == "cuda":
                    torch.cuda.synchronize()

            with profile.train_misc:
                losses.policy_loss += pg_loss.item() / num_update_iters
                losses.value_loss += v_loss.item() / num_update_iters
                losses.entropy += entropy_loss.item() / num_update_iters
                losses.old_approx_kl += old_approx_kl.item() / num_update_iters
                losses.approx_kl += approx_kl.item() / num_update_iters
                losses.clipfrac += clipfrac.item() / num_update_iters
                if experience.use_reference_loss:
                    losses.reference_loss += reference_loss.item() / num_update_iters
                
                # Track reward components
                losses.original_rewards += np.mean(original_rewards_np) / num_update_iters
                losses.likelihood += np.mean(likelihood_np) / num_update_iters

        if config.target_kl is not None:
            if approx_kl > config.target_kl:
                break

    with profile.train_misc:
        if config.anneal_lr:
            frac = 1.0 - data.global_step / config.total_timesteps
            lrnow = float(frac) * float(config.learning_rate)
            data.optimizer.param_groups[0]["lr"] = lrnow

        y_pred = experience.values_np
        y_true = experience.returns_np
        var_y = np.var(y_true)
        explained_var = (
            np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y
        )
        losses.explained_variance = explained_var
        data.epoch += 1

        done_training = data.global_step >= config.total_timesteps

        # Logging
        if profile.update(data) or done_training:
            print_dashboard(
                config.env,
                data.utilization,
                data.global_step,
                data.epoch,
                profile,
                data.losses,
                data.stats,
                data.msg,
            )

            # fmt: off
            if (
                data.wandb is not None
                and data.global_step > 0
                and time.perf_counter() - data.last_log_time > 3.0
            ):

                data.last_log_time = time.perf_counter()
                data.wandb.log(
                    {
                        "performance/controlled_agent_sps": profile.controlled_agent_sps,
                        "performance/controlled_agent_sps_env": profile.controlled_agent_sps_env,
                        "performance/pad_agent_sps": profile.pad_agent_sps,
                        "performance/pad_agent_sps_env": profile.pad_agent_sps_env,
                        "global_step": data.global_step,
                        "performance/epoch": data.epoch,
                        "performance/uptime": profile.uptime,
                        "train/learning_rate": data.optimizer.param_groups[0]["lr"],
                        **{f"metrics/{k}": v for k, v in data.stats.items()},
                        **{f"train/{k}": v for k, v in data.losses.items()},
                    }
                )

            if bool(data.stats):
                data.wandb.log({
                    **{f"metrics/{k}": v for k, v in data.stats.items()},
                })

            # fmt: on

        if data.epoch % config.checkpoint_interval == 0 or done_training:
            save_checkpoint(data)
            data.msg = f"Checkpoint saved at update {data.epoch}"


def close(data):
    data.vecenv.close()
    data.utilization.stop()
    config = data.config
    if data.wandb is not None:
        artifact_name = f"{config.exp_id}_model"
        artifact = data.wandb.Artifact(artifact_name, type="model")
        model_path = save_checkpoint(data)
        artifact.add_file(model_path)
        data.wandb.run.log_artifact(artifact)
        data.wandb.finish()


class Profile:
    controlled_agent_sps: ... = 0
    controlled_agent_sps_env: ... = 0
    pad_agent_sps: ... = 0
    pad_agent_sps_env: ... = 0
    uptime: ... = 0
    remaining: ... = 0
    eval_time: ... = 0
    env_time: ... = 0
    eval_forward_time: ... = 0
    eval_misc_time: ... = 0
    train_time: ... = 0
    train_forward_time: ... = 0
    learn_time: ... = 0
    train_misc_time: ... = 0

    def __init__(self):
        self.start = time.perf_counter()
        self.env = pufferlib.utils.Profiler()
        self.eval_forward = pufferlib.utils.Profiler()
        self.eval_misc = pufferlib.utils.Profiler()
        self.train_forward = pufferlib.utils.Profiler()
        self.learn = pufferlib.utils.Profiler()
        self.train_misc = pufferlib.utils.Profiler()
        self.prev_steps = 0
        self.prev_steps_pad = 0
        self.prev_env_elapsed = 0

    def __iter__(self):
        yield "controlled_agent_sps", self.controlled_agent_sps
        yield "controlled_agent_sps_env", self.controlled_agent_sps_env
        yield "pad_agent_sps", self.pad_agent_sps
        yield "pad_agent_sps_env", self.pad_agent_sps_env
        yield "uptime", self.uptime
        yield "remaining", self.remaining
        yield "eval_time", self.eval_time
        yield "env_time", self.env_time
        yield "eval_forward_time", self.eval_forward_time
        yield "eval_misc_time", self.eval_misc_time
        yield "train_time", self.train_time
        yield "train_forward_time", self.train_forward_time
        yield "learn_time", self.learn_time
        yield "train_misc_time", self.train_misc_time

    @property
    def epoch_time(self):
        return self.train_time + self.eval_time

    def update(self, data, interval_s=1):
        global_step = data.global_step
        global_step_pad = data.global_step_pad
        if global_step == 0:
            return True

        uptime = time.perf_counter() - self.start
        if uptime - self.uptime < interval_s:
            return False

        # SPS = delta global step / delta time (s)
        self.controlled_agent_sps = (global_step - self.prev_steps) / (
            uptime - self.uptime
        )
        self.controlled_agent_sps_env = (global_step - self.prev_steps) / (
            self.env.elapsed - self.prev_env_elapsed
        )

        self.pad_agent_sps = (global_step_pad - self.prev_steps_pad) / (
            uptime - self.uptime
        )
        self.pad_agent_sps_env = (global_step_pad - self.prev_steps_pad) / (
            self.env.elapsed - self.prev_env_elapsed
        )

        self.prev_steps = global_step
        self.prev_steps_pad = global_step_pad
        self.prev_env_elapsed = self.env.elapsed
        self.uptime = uptime

        self.remaining = (
            data.config.total_timesteps - global_step
        ) / self.controlled_agent_sps
        self.eval_time = data._timers["evaluate"].elapsed
        self.eval_forward_time = self.eval_forward.elapsed
        self.env_time = self.env.elapsed
        self.eval_misc_time = self.eval_misc.elapsed
        self.train_time = data._timers["train"].elapsed
        self.train_forward_time = self.train_forward.elapsed
        self.learn_time = self.learn.elapsed
        self.train_misc_time = self.train_misc.elapsed
        return True


def make_losses():
    return pufferlib.namespace(
        policy_loss=0,
        value_loss=0,
        entropy=0,
        old_approx_kl=0,
        approx_kl=0,
        clipfrac=0,
        explained_variance=0,
        reference_loss=0,
        original_rewards=0,  # NEW: Track original environment rewards
        likelihood=0,       # NEW: Track raw likelihood values
    )


class Experience:
    """Flat tensor storage (buffer) and array views for faster indexing."""

    def __init__(
        self,
        batch_size,
        bptt_horizon,
        minibatch_size,
        obs_shape,
        obs_dtype,
        atn_shape,
        cpu_offload=False,
        device="cuda",
        lstm=None,
        lstm_total_agents=0,
        controlled_agent_mask=None,
        use_reference_loss=False,  # NEW: Enable reference distribution loss
        use_smart_likelihood=False,
        action_dim=None,  # NEW: Add action dimension parameter
    ):
        if minibatch_size is None:
            minibatch_size = batch_size

        obs_dtype = pufferlib.pytorch.numpy_to_torch_dtype_dict[obs_dtype]
        pin = device == "cuda" and cpu_offload
        self.obs = torch.zeros(
            batch_size,
            *obs_shape,
            dtype=obs_dtype,
            pin_memory=pin,
            device=device if not pin else "cpu",
        )
        self.actions = torch.zeros(
            batch_size, *atn_shape, dtype=int, pin_memory=pin
        )
        self.logprobs = torch.zeros(batch_size, pin_memory=pin)
        self.rewards = torch.zeros(batch_size, pin_memory=pin)
        self.dones = torch.zeros(batch_size, pin_memory=pin)
        self.truncateds = torch.zeros(batch_size, pin_memory=pin)
        self.values = torch.zeros(batch_size, pin_memory=pin)
        
        # NEW: Storage for reference distribution loss (2D: batch_size x action_dim)
        self.use_reference_loss = use_reference_loss
        if use_reference_loss:
            if action_dim is None:
                raise ValueError("action_dim must be provided when use_reference_loss=True")
            # Store full reference distribution: (batch_size, action_dim)
            self.reference_log_probs = torch.zeros(batch_size, action_dim, pin_memory=pin)
            self.reference_log_probs_np = np.asarray(self.reference_log_probs)

        # +++ Add storage for SMART likelihood
        self.use_smart_likelihood = use_smart_likelihood
        self.smart_likelihood = torch.zeros(batch_size, pin_memory=pin)
        self.smart_likelihood_np = np.asarray(self.smart_likelihood)

        # self.obs_np = np.asarray(self.obs)
        self.actions_np = np.asarray(self.actions)
        self.logprobs_np = np.asarray(self.logprobs)
        self.rewards_np = np.asarray(self.rewards)
        self.dones_np = np.asarray(self.dones)
        self.truncateds_np = np.asarray(self.truncateds)
        self.values_np = np.asarray(self.values)

        self.lstm_h = self.lstm_c = None
        if lstm is not None:
            assert lstm_total_agents > 0
            shape = (lstm.num_layers, lstm_total_agents, lstm.hidden_size)
            self.lstm_h = torch.zeros(shape).to(device)
            self.lstm_c = torch.zeros(shape).to(device)

        num_minibatches = batch_size / minibatch_size
        self.num_minibatches = int(num_minibatches)
        if self.num_minibatches != num_minibatches:
            raise ValueError("batch_size must be divisible by minibatch_size")

        minibatch_rows = minibatch_size / bptt_horizon
        self.minibatch_rows = int(minibatch_rows)
        if self.minibatch_rows != minibatch_rows:
            raise ValueError(
                "minibatch_size must be divisible by bptt_horizon"
            )

        self.batch_size = batch_size
        self.bptt_horizon = bptt_horizon
        self.minibatch_size = minibatch_size
        self.device = device
        self.sort_keys = []
        self.ptr = 0
        self.smart_ptr = 0
        self.step = 0

        # NEW: Pre-compute mapping from flattened index to (world_idx, agent_idx)
        if controlled_agent_mask is not None:
            self.flat_to_world_agent = self._build_flat_to_world_agent_map(
                controlled_agent_mask
            )
        else:
            self.flat_to_world_agent = None

    @property
    def full(self):
        return self.ptr >= self.batch_size

    def store(self, obs, value, action, logprob, reward, done, env_id, mask, scene_idx, timestep):
        # Mask learner and ensure indices do not exceed batch size
        ptr = self.ptr
        indices = torch.where(mask)[0].cpu().numpy()[: self.batch_size - ptr]
        end = ptr + len(indices)

        # Store observations and other data
        self.obs[ptr:end] = obs.to(self.obs.device)[indices]
        self.values_np[ptr:end] = value.cpu().numpy()[indices]
        self.actions_np[ptr:end] = action.cpu().numpy()[indices]
        self.logprobs_np[ptr:end] = logprob.cpu().numpy()[indices]
        self.rewards_np[ptr:end] = reward.cpu().numpy()[indices]
        self.dones_np[ptr:end] = done.cpu().numpy()[indices]
        
        # +++ Initialize likelihood as zeros (will be assigned post-hoc)
        self.smart_likelihood_np[ptr:end] = 0.0
        
        # Store (world, agent, timestep) info for each experience
        if self.flat_to_world_agent is not None:
            # Batch lookup: shape (len(indices), 2)
            world_agent_pairs = self.flat_to_world_agent[indices]
            world_idx   = world_agent_pairs[:, 0]
            agent_idx   = world_agent_pairs[:, 1]
            scene_arr   = scene_idx[world_idx].cpu().numpy()                    # (n,)
            time_arr    = timestep[world_idx, agent_idx].cpu().numpy().astype(np.int16)

            self.sort_keys.extend(zip(scene_arr, world_idx, agent_idx, time_arr))
        else:
            # Fallback to old behavior
            self.sort_keys.extend([(env_id[i], self.step) for i in indices])
        
        self.ptr = end
        self.step += 1

    def sort_training_data(self):
        idxs = np.asarray(
            sorted(range(len(self.sort_keys)), key=self.sort_keys.__getitem__)
        )
        self.b_idxs_obs = (
            torch.as_tensor(
                idxs.reshape(
                    self.minibatch_rows,
                    self.num_minibatches,
                    self.bptt_horizon,
                ).transpose(1, 0, -1)
            )
            .to(self.obs.device)
            .long()
        )
        self.b_idxs = self.b_idxs_obs.to(self.device)
        self.b_idxs_flat = self.b_idxs.reshape(
            self.num_minibatches, self.minibatch_size
        )
        self.sort_keys = []
        self.ptr = 0
        self.smart_ptr = 0
        self.step = 0
        return idxs

    def flatten_batch(self, advantages_np):
        advantages = torch.from_numpy(advantages_np).to(self.device)
        b_idxs, b_flat = self.b_idxs, self.b_idxs_flat
        self.b_actions = self.actions.to(self.device, non_blocking=True)
        self.b_logprobs = self.logprobs.to(self.device, non_blocking=True)
        self.b_dones = self.dones.to(self.device, non_blocking=True)
        self.b_values = self.values.to(self.device, non_blocking=True)
        self.b_advantages = (
            advantages.reshape(
                self.minibatch_rows, self.num_minibatches, self.bptt_horizon
            )
            .transpose(0, 1)
            .reshape(self.num_minibatches, self.minibatch_size)
        )
        self.returns_np = advantages_np + self.values_np
        self.b_obs = self.obs[self.b_idxs_obs]
        self.b_actions = self.b_actions[b_idxs].contiguous()
        self.b_logprobs = self.b_logprobs[b_idxs]
        self.b_dones = self.b_dones[b_idxs]
        self.b_values = self.b_values[b_flat]
        self.b_returns = self.b_advantages + self.b_values
        
        # Add reference log probs flattening here (2D tensor handling)
        if self.use_reference_loss:
            self.b_reference_log_probs = self.reference_log_probs.to(self.device, non_blocking=True)
            # For 2D tensor: select indices for both dimensions (batch, action_dim)
            self.b_reference_log_probs = self.b_reference_log_probs[b_flat]  # Shape: (num_minibatches, minibatch_size, action_dim)

        # +++ Add SMART likelihood flattening
        self.b_smart_likelihood = self.smart_likelihood.to(self.device, non_blocking=True)
        self.b_smart_likelihood = self.b_smart_likelihood[b_flat]  # Shape: (num_minibatches, minibatch_size)

    def _build_flat_to_world_agent_map(self, controlled_agent_mask):
        """Pre-compute mapping from flattened controlled agent index to (world_idx, agent_idx)."""
        # Convert to numpy if needed
        mask_np = (
            controlled_agent_mask.cpu().numpy() 
            if isinstance(controlled_agent_mask, torch.Tensor) 
            else controlled_agent_mask
        )
        
        # Vectorized approach: find all (world, agent) pairs where mask is True
        world_idxs, agent_idxs = np.nonzero(mask_np)
        
        # Store as 2D array for fast batch lookup: shape (N, 2)
        # flat_to_world_agent[i] = [world_idx, agent_idx] for flat index i
        flat_to_world_agent = np.stack([world_idxs, agent_idxs], axis=1)
        
        return flat_to_world_agent

    def update_controlled_agent_mask(self, controlled_agent_mask):
        """Update the mapping when scenarios are resampled."""
        if controlled_agent_mask is not None:
            self.flat_to_world_agent = self._build_flat_to_world_agent_map(
                controlled_agent_mask
            )

    def assign_smart_likelihood_to_experiences(self, L):
        """
        Fill SMART likelihoods only for the buffer rows that do not have them yet:
        # will we have problems assinging the likelihood, if we sorted by scene,world,agent,timestep?
        given worlds,agents, timesteps --> assing the corrdponding likelihood to the agents
        [smart_ptr : ptr)
        """
     
        if L.numel() == 0:
            return

        start, end = self.smart_ptr, self.ptr
        if start >= end:                 # nothing pending
            return

        # --- 1. extract the pending (world, agent, timestep) triples ----------
        pending_keys = self.sort_keys[start:end]          # list of tuples
        triplets = np.asarray([k[1:] for k in pending_keys], dtype=np.int32)   # (N, 3) #we ignore the scene_idx
        w, a, t = triplets.T                              # each length N

        # --- 2. keep only indices that are valid for L ------------------------
        W, A, T = L.shape
        valid = (w >= 0) & (w < W) & (a >= 0) & (a < A) & (t >= 0) & (t < T)
        if not valid.any():
            self.smart_ptr = end      # mark as done even if nothing was valid
            return

        w, a, t = w[valid], a[valid], t[valid]
        buffer_pos = np.nonzero(valid)[0] + start         # where to write

        # --- 3. vectorised gather, no Python loop -----------------------------
        flat_idx = (w * (A * T) + a * T + t).astype(np.int64)   # (N_valid,)
        flat_L   = L.flatten().cpu().numpy()                    # (W*A*T,)

        self.smart_likelihood_np[buffer_pos] = flat_L[flat_idx]

        # --- 4. advance pointer ----------------------------------------------
        self.smart_ptr = end


class Utilization(Thread):
    def __init__(self, delay=1, maxlen=20):
        super().__init__()
        self.cpu_mem = deque(maxlen=maxlen)
        self.cpu_util = deque(maxlen=maxlen)
        self.gpu_util = deque(maxlen=maxlen)
        self.gpu_mem = deque(maxlen=maxlen)

        self.delay = delay
        self.stopped = False
        self.start()

    def run(self):
        while not self.stopped:
            self.cpu_util.append(psutil.cpu_percent())
            mem = psutil.virtual_memory()
            self.cpu_mem.append(mem.active / mem.total)
            self.gpu_util.append(torch.cuda.utilization())
            free, total = torch.cuda.mem_get_info()
            self.gpu_mem.append(free / total)
            time.sleep(self.delay)

    def stop(self):
        self.stopped = True


def save_checkpoint(data, save_checkpoint_to_wandb=True):

    config = data.config
    path = os.path.join(config.checkpoint_path, config.exp_id)

    if not os.path.exists(path):
        os.makedirs(path)

    model_name = f"model_{config.exp_id}_{data.epoch:06d}.pt"
    model_path = os.path.join(path, model_name)

    # Save training state
    state = {
        "parameters": data.uncompiled_policy.state_dict(),
        "optimizer_state_dict": data.optimizer.state_dict(),
        "global_step": data.global_step,
        "agent_step": data.global_step,
        "update": data.epoch,
        "model_name": model_name,
        "model_class": data.uncompiled_policy.__class__.__name__,
        "model_arch": config.network,
        "action_dim": data.uncompiled_policy.action_dim,
        "exp_id": config.exp_id,
        "num_params": config.network["num_parameters"],
    }

    torch.save(state, model_path)
    if save_checkpoint_to_wandb and data.wandb is not None:

        data.wandb.save(model_path)

        data.wandb.config.update(
            {
                "network_class": data.uncompiled_policy.__class__.__name__,
                "network_arch": config.network,
                "exp_id": config.exp_id,
            }
        )

        # Optionally log the optimizer state path
        data.wandb.save(model_path)

    return model_path


def count_params(policy):
    return sum(p.numel() for p in policy.parameters() if p.requires_grad)


def seed_everything(seed, torch_deterministic):
    random.seed(seed)
    np.random.seed(seed)
    if seed is not None:
        torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = torch_deterministic
