# Copyright 2025 Alibaba Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Ajet extension for verl DataParallelPPOActor.
Overrides `update_policy` to support `override_ppo_mini_batch_num` and add debug logging.
"""

import logging
import math
import os
from typing import Optional

import torch
import torch.distributed as dist
from torch import nn
from verl import DataProto
from verl.trainer.ppo.core_algos import (
    agg_loss, compute_self_distillation_loss,
    compute_self_distillation_with_rlvr_loss, get_policy_loss_fn, kl_penalty)
from verl.utils.device import get_device_id
from verl.utils.profiler import GPUMemoryLogger
from verl.utils.py_functional import append_to_dict
from verl.workers.actor.dp_actor import (DataParallelPPOActor,
                                         TrustRegionTeacher)

# ajet/backbone/verl/seqlen_balancing.py
from ajet.backbone.verl.seqlen_balancing import (prepare_dynamic_batch,
                                                 restore_dynamic_batch)
from ajet.tuner_lib.experimental.interchange_utils import http_push_verbose_log

__all__ = ["AjetDataParallelPPOActor"]

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


class AjetDataParallelPPOActor(DataParallelPPOActor):
    """DataParallelPPOActor with ajet-specific modifications:

    1. Supports `override_ppo_mini_batch_num` to control the number of optimizer steps per train-batch-step.
    2. Adds debug print for tensor shapes during training.
    3. Override `prepare_dynamic_batch`
    """

    @GPUMemoryLogger(role="dp actor", logger=logger)
    def compute_log_prob(self, data: DataProto, calculate_entropy: bool = False) -> dict[str, torch.Tensor]:
        """Compute the log probability of the responses given input_ids, attention_mask and position_ids

        Args:
            data (DataProto): a DataProto containing keys

                ``input_ids``: tensor of shape [batch_size, sequence_length]. torch.int64. Note that input_ids is the
                concatenation of prompt and response. Note that ``sequence_length = prompt_length + response_length``.

                ``attention_mask``: tensor of shape [batch_size, sequence_length]. torch.int64.

                ``position_ids``: tensor of shape [batch_size, sequence_length]. torch.int64.

                ``responses``:  tensor of shape [batch_size, response_length]. torch.int64.

        Returns:
            dict[str, torch.Tensor]: a dict containing keys
                - ``log_probs``: tensor of shape [batch_size, response_length]. torch.float32.
                - ``entropys``: tensor of shape [batch_size, response_length]. torch.float32.
                - ``sum_pi_squared``: tensor of shape [batch_size, response_length]. torch.float32.
        """
        calculate_sum_pi_squared = self.config.get("calculate_sum_pi_squared", False)
        self.actor_module.eval()

        micro_batch_size = data.meta_info["micro_batch_size"]
        temperature = data.meta_info["temperature"]  # temperature must be in the data.meta_info to avoid silent error
        use_dynamic_bsz = data.meta_info["use_dynamic_bsz"]
        pad_token_id = data.meta_info.get("pad_token_id", 0)
        has_multi_modal_inputs = "multi_modal_inputs" in data.non_tensor_batch.keys()

        select_keys = ["responses", "input_ids", "attention_mask", "position_ids"]
        non_tensor_select_keys = ["multi_modal_inputs"] if has_multi_modal_inputs else []
        if self.use_prefix_grouper:
            select_keys += [k for k in ["prompts", "response_mask"] if k in data.batch]
            if "uid" in data.non_tensor_batch:
                non_tensor_select_keys.append("uid")

        data = data.select(batch_keys=select_keys, non_tensor_batch_keys=non_tensor_select_keys)

        if use_dynamic_bsz:
            max_token_len = data.meta_info["max_token_len"] * self.ulysses_sequence_parallel_size
            micro_batches, batch_idx_list = prepare_dynamic_batch(data, max_token_len=max_token_len)
        else:
            micro_batches = data.split(micro_batch_size)

        log_probs_lst = []
        entropy_lst = []
        sum_pi_squared_lst = []
        # print(f"len(micro_batches) = {len(micro_batches)}")
        for micro_batch in micro_batches:
            micro_batch = micro_batch.to(get_device_id())
            model_inputs = {**micro_batch.batch, **micro_batch.non_tensor_batch, "pad_token_id": pad_token_id}
            with torch.no_grad():
                outputs = self._forward_micro_batch(
                    model_inputs, temperature=temperature, calculate_entropy=calculate_entropy
                )
            log_probs_lst.append(outputs["log_probs"])
            if calculate_entropy:
                entropy_lst.append(outputs["entropys"])
            if calculate_sum_pi_squared:
                sum_pi_squared_lst.append(outputs["sum_pi_squared"])

        log_probs = torch.concat(log_probs_lst, dim=0)
        if calculate_entropy:
            entropys = torch.concat(entropy_lst, dim=0)
        if calculate_sum_pi_squared:
            sum_pi_squared = torch.concat(sum_pi_squared_lst, dim=0)

        if use_dynamic_bsz:
            log_probs = restore_dynamic_batch(log_probs, batch_idx_list)
            if calculate_entropy:
                entropys = restore_dynamic_batch(entropys, batch_idx_list)
            if calculate_sum_pi_squared:
                sum_pi_squared = restore_dynamic_batch(sum_pi_squared, batch_idx_list)

        outputs = {"log_probs": log_probs}
        if calculate_entropy:
            outputs["entropys"] = entropys
        if calculate_sum_pi_squared:
            outputs["sum_pi_squared"] = sum_pi_squared
        return outputs

    @GPUMemoryLogger(role="dp actor", logger=logger)
    def update_policy(self, data: DataProto):
        # make sure we are in training mode
        self.actor_module.train()

        temperature = data.meta_info["temperature"]  # temperature must be in the data.meta_info to avoid silent error
        pad_token_id = data.meta_info.get("pad_token_id", 0)

        loss_mode = self.config.policy_loss.get("loss_mode", "vanilla")
        # vanilla -> verl.trainer.ppo.core_algos.compute_policy_loss_vanilla

        self_distillation_enabled = loss_mode == "sdpo"
        self_distillation_cfg = getattr(self.config, "self_distillation", None)
        teacher_regularization = "ema"
        teacher_update_rate = 0.0
        trust_region_teacher: Optional[nn.Module] = None
        if self_distillation_enabled:
            self_distillation_required_keys = [
                "teacher_input_ids",
                "teacher_attention_mask",
                "teacher_position_ids",
                "self_distillation_mask",
            ]
            missing = set(self_distillation_required_keys) - set(data.batch.keys())
            if missing:
                raise ValueError(f"SDPO is enabled but required teacher keys are missing: {sorted(missing)}")
            teacher_regularization = self.resolve_teacher_regularization(self_distillation_cfg)
            teacher_update_rate = self.resolve_teacher_update_rate(self_distillation_cfg)
            if teacher_regularization == "trust_region":
                if self.use_fused_kernels:
                    raise ValueError("SDPO trust-region teacher requires use_fused_kernels=False.")
                if self.teacher_module is None:
                    raise ValueError("Trust-region teacher requires a reference teacher_module.")
                if isinstance(self.teacher_module, TrustRegionTeacher):
                    trust_region_teacher = self.teacher_module
                else:
                    if self.teacher_module is self.actor_module:
                        raise ValueError("Trust-region teacher requires a separate reference teacher_module.")
                    trust_region_teacher = TrustRegionTeacher(
                        teacher_module=self.teacher_module,
                        student_module=self.actor_module,
                        mix_coef=teacher_update_rate,
                    )

        select_keys = [
            "responses",
            "response_mask",
            "input_ids",
            "attention_mask",
            "position_ids",
            "old_log_probs",
            "advantages",
        ]
        if self.use_prefix_grouper and "prompts" in data.batch.keys():
            select_keys.append("prompts")
        if self.config.use_kl_loss:
            select_keys.append("ref_log_prob")
        if self_distillation_enabled:
            select_keys.extend(self_distillation_required_keys)
        # Include pre-computed IS weights if present in batch
        # Weights are computed centrally in trainer and added to batch when algorithm.rollout_is=True
        if "rollout_is_weights" in data.batch.keys():
            select_keys.append("rollout_is_weights")
        # Include rollout_log_probs for computing rollout_corr metrics in bypass mode
        if "rollout_log_probs" in data.batch.keys():
            select_keys.append("rollout_log_probs")

        has_multi_modal_inputs = "multi_modal_inputs" in data.non_tensor_batch.keys()
        has_non_empty_multi_modal_inputs = self._has_non_empty_multi_modal_inputs(
            data.non_tensor_batch.get("multi_modal_inputs")
        )
        non_tensor_select_keys = []
        if has_multi_modal_inputs:
            non_tensor_select_keys.append("multi_modal_inputs")
        if self.use_prefix_grouper and "uid" in data.non_tensor_batch.keys():
            non_tensor_select_keys.append("uid")

        data = data.select(batch_keys=select_keys, non_tensor_batch_keys=non_tensor_select_keys)

        # Split to make minibatch iterator for updating the actor
        # See PPO paper for details. https://arxiv.org/abs/1707.06347
        # [AJET] Support override_ppo_mini_batch_num to control the number of optimizer steps
        if self.config.override_ppo_mini_batch_num > 0:
            mini_batch_split_size = math.ceil(data.batch.batch_size[0] / self.config.override_ppo_mini_batch_num)
        else:
            mini_batch_split_size = self.config.ppo_mini_batch_size

        mini_batches = data.split(mini_batch_split_size)

        on_policy = len(mini_batches) == 1 and self.config.ppo_epochs == 1

        metrics = {
            "actor/pg_loss": 0.0,
            "actor/kl_loss": 0.0,
        }
        did_update = False
        for _ in range(self.config.ppo_epochs):
            for batch_idx, mini_batch in enumerate(mini_batches):
                if self.config.use_dynamic_bsz:
                    max_token_len = self.config.ppo_max_token_len_per_gpu * self.ulysses_sequence_parallel_size
                    micro_batches, _ = prepare_dynamic_batch(mini_batch, max_token_len=max_token_len)
                else:
                    self.gradient_accumulation = (
                        self.config.ppo_mini_batch_size // self.config.ppo_micro_batch_size_per_gpu
                    )
                    micro_batches = mini_batch.split(self.config.ppo_micro_batch_size_per_gpu)

                self.actor_optimizer.zero_grad()

                num_micro_batches = len(micro_batches)
                for micro_batch_idx, micro_batch in enumerate(micro_batches, 1):
                    micro_batch = micro_batch.to(get_device_id())
                    micro_batch_metrics = {}
                    model_inputs = {**micro_batch.batch, **micro_batch.non_tensor_batch, "pad_token_id": pad_token_id}
                    response_mask = model_inputs["response_mask"]
                    old_log_prob = model_inputs["old_log_probs"]
                    advantages = model_inputs["advantages"]
                    # [AJET] Debug logging for tensor shapes
                    input_ids = model_inputs["input_ids"]
                    _shape_msg = f'[Update Policy] -> Micro batch shape, input_ids {input_ids.shape}, response {response_mask.shape} @{micro_batch_idx}/{num_micro_batches}'
                    print(_shape_msg)
                    if (not dist.is_available()) or (not dist.is_initialized()) or dist.get_rank() == 0:
                        http_push_verbose_log(_shape_msg, tag="update_policy")

                    entropy_coeff = self.config.entropy_coeff
                    loss_agg_mode = self.config.loss_agg_mode

                    calculate_entropy = self.config.calculate_entropy or (entropy_coeff != 0)
                    self_distillation_mask = (
                        model_inputs.get("self_distillation_mask") if self_distillation_enabled else None
                    )
                    if self_distillation_enabled and has_non_empty_multi_modal_inputs:
                        raise ValueError("SDPO does not support multi-modal inputs in actor.update_policy.")

                    if self.config.override_ppo_mini_batch_num > 0:
                        loss_scale_factor = response_mask.shape[0] / mini_batch_split_size
                    elif self.config.use_dynamic_bsz:
                        loss_scale_factor = response_mask.shape[0] / self.config.ppo_mini_batch_size
                    else:
                        loss_scale_factor = 1 / self.gradient_accumulation
                    loss_scale_factor *= self.config.loss_extra_scale_ratio  # [AJET] Extra scaling for loss if needed

                    # all return: (bsz, response_length)
                    compute_full_log_probs = False
                    if self_distillation_enabled and self_distillation_cfg.full_logit_distillation:
                        compute_full_log_probs = True
                    outputs = self._forward_micro_batch(
                        self.actor_module,
                        model_inputs,
                        temperature=temperature,
                        calculate_entropy=calculate_entropy,
                        compute_full_log_probs=compute_full_log_probs,
                        top_k_log_probs=self_distillation_cfg.distillation_topk if compute_full_log_probs else None
                    )
                    log_prob = outputs["log_probs"]
                    entropy = outputs["entropys"] if calculate_entropy else None

                    # for fully_async_policy
                    if hasattr(self.config, "use_rollout_log_probs") and self.config.use_rollout_log_probs:
                        old_log_prob = model_inputs["old_log_probs"]
                    else:
                        if on_policy:
                            old_log_prob = log_prob.detach()
                        else:
                            old_log_prob = model_inputs["old_log_probs"]

                    # Extract pre-computed rollout correction weights if present
                    # Weights are computed centrally in trainer and added when algorithm.rollout_is=True
                    rollout_is_weights = model_inputs.get("rollout_is_weights", None)

                    if self_distillation_enabled:
                        teacher_inputs = {
                            "responses": model_inputs["responses"],
                            "input_ids": model_inputs["teacher_input_ids"],
                            "attention_mask": model_inputs["teacher_attention_mask"],
                            "position_ids": model_inputs["teacher_position_ids"],
                            "topk_indices": model_inputs.get("topk_indices", None),
                        }
                        teacher_model = trust_region_teacher or self.teacher_module or self.actor_module
                        with torch.no_grad():
                            teacher_outputs = self._forward_micro_batch(
                                teacher_model, teacher_inputs,
                                temperature=temperature,
                                calculate_entropy=False,
                                compute_full_log_probs=compute_full_log_probs,
                            )
                        full_log_prob = data.get("full_log_probs", None)
                        teacher_log_prob, teacher_full_log_prob = teacher_outputs["log_probs"], teacher_outputs.get("full_log_probs", None)
                        if self_distillation_cfg.use_sdrlvr:
                            pg_loss, pg_metrics = compute_self_distillation_with_rlvr_loss(
                                old_log_prob=old_log_prob,
                                log_prob=log_prob,
                                advantages=advantages,
                                response_mask=response_mask,
                                full_log_prob=full_log_prob,
                                teacher_log_prob=teacher_log_prob,
                                teacher_full_log_prob=teacher_full_log_prob,
                                self_distillation_mask=self_distillation_mask,
                                loss_agg_mode=loss_agg_mode,
                                config=self.config,
                                rollout_is_weights=rollout_is_weights,
                            )
                        else:
                            pg_loss, pg_metrics = compute_self_distillation_loss(
                                old_log_prob=old_log_prob,
                                log_prob=log_prob,
                                advantages=advantages,
                                response_mask=response_mask,
                                full_log_prob=full_log_prob,
                                teacher_log_prob=teacher_log_prob,
                                teacher_full_log_prob=teacher_full_log_prob,
                                self_distillation_mask=self_distillation_mask,
                                loss_agg_mode=loss_agg_mode,
                                config=self.config,
                                rollout_is_weights=rollout_is_weights,
                            )
                        micro_batch_metrics.update(pg_metrics)
                    else:
                        # gpg -> verl.trainer.ppo.core_algos.compute_policy_loss_gpg
                        # clip_cov -> verl.trainer.ppo.core_algos.compute_policy_loss_clip_cov
                        policy_loss_fn = get_policy_loss_fn(loss_mode)

                        # Compute policy loss (any function is expected to return 2 values)
                        pg_loss, pg_metrics = policy_loss_fn(
                            old_log_prob=old_log_prob,
                            log_prob=log_prob,
                            advantages=advantages,
                            response_mask=response_mask,
                            loss_agg_mode=loss_agg_mode,
                            config=self.config,
                            rollout_is_weights=rollout_is_weights,
                        )
                        micro_batch_metrics.update(pg_metrics)

                    # Skip if using bypass_mode loss (metrics already computed in pg_metrics)
                    rollout_log_prob = model_inputs.get("rollout_log_probs", None)
                    if loss_mode != "bypass_mode" and rollout_log_prob is not None:
                        # Compute metrics using CURRENT policy π_θ vs π_rollout
                        # Tracks evolving off-policy gap as π_θ updates during mini-batch training
                        from verl.trainer.ppo.rollout_corr_helper import \
                            compute_rollout_corr_metrics_from_logprobs

                        rollout_corr_metrics = compute_rollout_corr_metrics_from_logprobs(
                            log_prob=log_prob,
                            rollout_log_prob=rollout_log_prob,
                            response_mask=response_mask,
                        )
                        micro_batch_metrics.update(rollout_corr_metrics)

                    policy_loss = pg_loss
                    if calculate_entropy and entropy is not None:
                        entropy_agg = agg_loss(loss_mat=entropy, loss_mask=response_mask, loss_agg_mode=loss_agg_mode)
                        micro_batch_metrics["actor/entropy"] = entropy_agg.detach().item()
                        if entropy_coeff != 0:
                            policy_loss -= entropy_agg * entropy_coeff

                    if self.config.use_kl_loss:
                        ref_log_prob = model_inputs["ref_log_prob"]
                        # compute kl loss
                        kld = kl_penalty(
                            logprob=log_prob, ref_logprob=ref_log_prob, kl_penalty=self.config.kl_loss_type
                        )
                        kl_loss = agg_loss(loss_mat=kld, loss_mask=response_mask, loss_agg_mode=loss_agg_mode)

                        policy_loss = policy_loss + kl_loss * self.config.kl_loss_coef
                        metrics["actor/kl_loss"] += kl_loss.detach().item() * loss_scale_factor
                        micro_batch_metrics["actor/kl_coef"] = self.config.kl_loss_coef

                    if self.config.use_dynamic_bsz:
                        # relative to the dynamic bsz
                        loss = policy_loss * loss_scale_factor
                    else:
                        loss = policy_loss * loss_scale_factor
                    if self.scaler is not None:
                        self.scaler.scale(loss).backward()
                    else:
                        loss.backward()

                    metrics["actor/pg_loss"] += pg_loss.detach().item() * loss_scale_factor
                    append_to_dict(metrics, micro_batch_metrics)

                print('-> optimizer_step !')
                grad_norm = self._optimizer_step()
                if torch.isfinite(grad_norm).item():
                    did_update = True
                mini_batch_metrics = {"actor/grad_norm": grad_norm.detach().item()}
                append_to_dict(metrics, mini_batch_metrics)
        self.actor_optimizer.zero_grad()
        if did_update:
            self._update_teacher()
        return metrics
