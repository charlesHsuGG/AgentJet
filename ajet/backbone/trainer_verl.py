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


import os
import uuid
from collections import defaultdict
from copy import deepcopy
from pprint import pprint
from typing import Any, List, Optional

import hydra
import numpy as np
import torch
from beast_logger import print_dict
from loguru import logger
from torch.utils.data import Dataset, Sampler
from tqdm import tqdm
from verl import DataProto
from verl.experimental.agent_loop.agent_loop import AsyncLLMServerManager
from verl.experimental.dataset.sampler import AbstractCurriculumSampler
from verl.single_controller.ray import RayWorkerGroup, ResourcePoolManager
from verl.trainer.ppo.core_algos import AdvantageEstimator, agg_loss
from verl.trainer.ppo.metric_utils import (compute_data_metrics,
                                           compute_throughout_metrics,
                                           compute_timing_metrics,
                                           compute_variance_proxy_metrics)
from verl.trainer.ppo.ray_trainer import (RayPPOTrainer, apply_kl_penalty,
                                          compute_advantage,
                                          compute_response_mask)
from verl.trainer.ppo.reward import extract_reward
from verl.trainer.ppo.utils import Role, WorkerType
from verl.utils.checkpoint.checkpoint_manager import should_save_ckpt_esi
from verl.utils.config import omega_conf_to_dataclass
from verl.utils.debug import marked_timer
from verl.utils.metric import reduce_metrics
from verl.utils.rollout_skip import RolloutSkip

from ajet.backbone.warm_up import warm_up_process
from ajet.context_tracker.single_agent_tracking import \
    SingleAgentContextTracker
from ajet.schema.task import Task
from ajet.task_reader import dict_to_ajet_task
from ajet.task_rollout.native_parallel_worker import VerlRolloutManager
from ajet.utils.metric_helper import (save_trajectory_as_json_file,
                                      update_metrics)


def compute_reward(data: DataProto, reward_fn) -> tuple[torch.Tensor, dict[str, Any]]:
    """
    Compute reward for a batch of data.
    Args:
        data: DataProto object containing the input data.
        reward_fn: Reward function to compute the reward.
    Returns:
        Tuple of reward tensor and extra info dictionary.
    """
    try:
        reward_result = reward_fn(data, return_dict=True)
        reward_tensor = reward_result["reward_tensor"]
        reward_extra_infos_dict = reward_result.get("reward_extra_info", {})
    except Exception as e:
        print(f"Error in reward_fn: {e}")
        reward_tensor = reward_fn(data)
        reward_extra_infos_dict = {}

    return reward_tensor, reward_extra_infos_dict


def parse_reward_from_dataproto(data: DataProto, return_dict=False) -> dict | torch.Tensor:
    """
    Compute reward for a batch of data.
    Args:
        data: DataProto object containing the input data.
        return_dict: Whether to return a dictionary or just the reward tensor.

    Returns:
        Tensor of shape (bs, response_len) if return_dict is False,
        or a dict with 'reward_tensor' and 'reward_extra_info'.
    """
    # Within DataFlow, world.execute() will pass a float score, which will be contained in the DataProto.non_tensor_batch('reward_scores')

    # Initialize reward tensor
    reward_tensor = torch.zeros_like(data.batch["responses"], dtype=torch.float32)  # (bs, reslen)
    reward_extra_info = defaultdict(list)

    # Batch-level processing
    prompt_ids_batch = data.batch["prompts"]  # (bs, prompt_len)
    prompt_lengths = prompt_ids_batch.shape[-1]

    # Get attention masks for all items
    attention_masks = data.batch["attention_mask"]  # (bs, total_len)
    response_lengths = attention_masks[:, prompt_lengths:].sum(dim=1)  # (bs, )

    # Get reward scores
    reward_scores_list = [item for item in data.non_tensor_batch["reward_scores"]]
    reward_scores = torch.tensor(
        reward_scores_list, device=reward_tensor.device, dtype=torch.float32
    )  # (bs, )

    # Use advanced indexing to assign rewards (placing reward at the last token position)
    reward_tensor[torch.arange(len(data)), response_lengths - 1] = reward_scores

    if return_dict:
        return {
            "reward_tensor": reward_tensor,
            "reward_extra_info": reward_extra_info,
        }
    return reward_tensor


def union_gen_batch_via_task_id(tasks, batch: DataProto, gen_batch_output: DataProto, discard_original_batch=False):
    """
    Union the gen_batch_output with the batch based on task_id.
    """
    if not discard_original_batch:
        map_task_id_to_index = {t.task_id: i for i, t in enumerate(tasks)}
        gen_task_task_ids = gen_batch_output.non_tensor_batch["task_ids"]
        indices = [map_task_id_to_index[tid] for tid in gen_task_task_ids]
        batch_extend = batch.select_idxs(indices)
        batch_final = batch_extend.union(gen_batch_output)
        return batch_final
    else:
        gen_batch_output.non_tensor_batch['uid'] = gen_batch_output.non_tensor_batch["task_ids"]
        task_id_counter = {}
        for i, tid in enumerate(gen_batch_output.non_tensor_batch["task_ids"]):
            if tid in task_id_counter:
                task_id_counter[tid] += 1
            else:
                task_id_counter[tid] = 1
            current_id = task_id_counter[tid]
            gen_batch_output.non_tensor_batch['rollout_ids'][i] = f"T{tid}R{current_id}"
        logger.info(f'task_id_counter: {task_id_counter}')
        return gen_batch_output


def import_or_export_data_proto(batch: DataProto, direction: str = "export", file: str = "./tmp.pkl") -> DataProto:
    """Import or export a DataProto batch to/from a pickle file.

    Args:
        batch: The DataProto batch object. Used when direction is "export";
               ignored (can be None) when direction is "import".
        direction: "import" to load a batch from file, "export" to save the batch to file.
        file: Path to the pickle file. Defaults to "./tmp.pkl".

    Returns:
        The DataProto batch — either the one just loaded (import) or the one just saved (export).

    Raises:
        ValueError: If direction is not "import" or "export".
        FileNotFoundError: If direction is "import" and the file does not exist.
    """
    import pickle
    if direction == "export":
        with open(file, "wb") as f:
            pickle.dump(batch, f)
        logger.info(f"[import_or_export_data_proto] Exported batch to {file}")
        return batch
    elif direction == "import":
        with open(file, "rb") as f:
            batch = pickle.load(f)
        logger.info(f"[import_or_export_data_proto] Imported batch from {file}")
        return batch
    else:
        raise ValueError(f"direction must be 'import' or 'export', got '{direction}'")


class AjetRayPPOTrainer(RayPPOTrainer):
    """Distributed PPO trainer using Ray for scalable reinforcement learning.
    Slightly modified from RayPPOTrainer in verl.
    """

    def __init__(
        self,
        config,
        tokenizer,
        role_worker_mapping: dict[Role, WorkerType],
        resource_pool_manager: ResourcePoolManager,
        ray_worker_group_cls: type[RayWorkerGroup] = RayWorkerGroup,
        processor=None,
        train_dataset: Optional[Dataset] = None,
        val_dataset: Optional[Dataset] = None,
        collate_fn=None,
        train_sampler: Optional[Sampler] = None,
        device_name=None,
    ):
        super().__init__(
            config, tokenizer, role_worker_mapping, resource_pool_manager, ray_worker_group_cls, processor, train_dataset, val_dataset, collate_fn, train_sampler, device_name
        )
        if self.config.algorithm.adv_estimator == "sdpo":
            self.config.algorithm.adv_estimator = "grpo"
            self.config.actor_rollout_ref.actor.policy_loss.loss_mode = 'sdpo'

        if self.config.algorithm.adv_estimator == "sdrlvr":
            self.config.algorithm.adv_estimator = "grpo"
            self.config.actor_rollout_ref.actor.policy_loss.loss_mode = 'sdpo'
            self.config.actor_rollout_ref.actor.self_distillation.use_sdrlvr = True

    # #######################################
    # init
    # #######################################
    def _validate_config(self):
        config = self.config
        # number of GPUs total
        n_gpus = config.trainer.n_gpus_per_node * config.trainer.nnodes
        minimal_bsz = n_gpus

        # 1. Check total batch size for data correctness
        real_train_batch_size = (
            config.ajet.data.train_batch_size * config.ajet.rollout.num_repeat
        )
        assert real_train_batch_size % minimal_bsz == 0, (
            f"real_train_batch_size ({real_train_batch_size}) must be divisible by minimal possible batch size "
            f"({minimal_bsz})"
        )

        # A helper function to check "micro_batch_size" vs "micro_batch_size_per_gpu"
        # We throw an error if the user sets both. The new convention is "..._micro_batch_size_per_gpu".
        def check_mutually_exclusive(mbs, mbs_per_gpu, name: str):
            """Validate mutually exclusive micro batch size configuration options.

            Ensures that users don't set both deprecated micro_batch_size and
            the new micro_batch_size_per_gpu parameters simultaneously.

            Args:
                mbs: Deprecated micro batch size parameter value.
                mbs_per_gpu: New micro batch size per GPU parameter value.
                name (str): Configuration section name for error messages.

            Raises:
                ValueError: If both parameters are set or neither is set.
            """
            settings = {
                "reward_model": "micro_batch_size",
                "actor_rollout_ref.ref": "log_prob_micro_batch_size",
                "actor_rollout_ref.rollout": "log_prob_micro_batch_size",
            }

            if name in settings:
                param = settings[name]
                param_per_gpu = f"{param}_per_gpu"

                if mbs is None and mbs_per_gpu is None:
                    raise ValueError(
                        f"[{name}] Please set at least one of '{name}.{param}' or '{name}.{param_per_gpu}'."
                    )

                if mbs is not None and mbs_per_gpu is not None:
                    raise ValueError(
                        f"[{name}] You have set both '{name}.{param}' AND '{name}.{param_per_gpu}'. Please remove "
                        f"'{name}.{param}' because only '*_{param_per_gpu}' is supported (the former is deprecated)."
                    )

        # Actor validation done in ActorConfig.__post_init__ and validate()
        try:
            actor_config = omega_conf_to_dataclass(config.actor_rollout_ref.actor)
            actor_config.validate(
                n_gpus,
                config.ajet.data.train_batch_size,
                config.actor_rollout_ref.model,
            )
        except hydra.errors.InstantiationException as e:
            raise ValueError("You are using an unsupported VERL version. Please read `documents/backbones.md`") from e
        if not config.actor_rollout_ref.actor.use_dynamic_bsz:
            if self.use_reference_policy:
                # reference: log_prob_micro_batch_size vs. log_prob_micro_batch_size_per_gpu
                check_mutually_exclusive(
                    config.actor_rollout_ref.ref.log_prob_micro_batch_size,
                    config.actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu,
                    "actor_rollout_ref.ref",
                )

            #  The rollout section also has log_prob_micro_batch_size vs. log_prob_micro_batch_size_per_gpu
            check_mutually_exclusive(
                config.ajet.rollout.log_prob_micro_batch_size,
                config.ajet.rollout.log_prob_micro_batch_size_per_gpu,
                "actor_rollout_ref.rollout",
            )

        if self.config.algorithm.use_kl_in_reward and config.actor_rollout_ref.actor.use_kl_loss:
            logger.warning("NOTICE: You have both enabled in-reward kl and kl loss.")

        # critic
        if self.use_critic:
            critic_config = omega_conf_to_dataclass(config.critic)
            critic_config.validate(n_gpus, config.ajet.data.train_batch_size)

        if config.data.get("val_batch_size", None) is not None:
            logger.warning(
                "WARNING: val_batch_size is deprecated. Validation datasets are sent to inference engines as a whole batch, "
                "which will schedule the memory themselves."
            )

        # check eval config
        if config.ajet.rollout.val_kwargs.do_sample:
            assert (
                config.ajet.rollout.temperature > 0
            ), "validation gen temperature should be greater than 0 when enabling do_sample"

        logger.success("[validate_config] All configuration checks passed successfully!")

    def init_workers(self):
        super().init_workers()

        assert hasattr(self.async_rollout_manager, "agent_loop_workers")
        assert len(self.async_rollout_manager.agent_loop_workers) == 1, "Please set `num_workers = 1` in `ajet/default_config/verl/verl_default.yaml`"

        servers = list(zip(self.async_rollout_manager.server_addresses, self.async_rollout_manager.server_handles, strict=True))
        real_async_rollout_manager: AsyncLLMServerManager = AsyncLLMServerManager(
            config=self.async_rollout_manager.config,
            servers=servers,
            load_balancer_handle=self.async_rollout_manager.global_load_balancer
        )

        self.parallel_env = VerlRolloutManager(
            config=self.config,
            async_rollout_manager=real_async_rollout_manager,
            max_parallel=self.config.ajet.rollout.max_env_worker,
            tokenizer=self.tokenizer,
        )

    def _update_interchange_server_status_flag(self, status: str):
        if self.config.ajet.enable_interchange_server:
            if self.config.ajet.enable_swarm_mode:
                from ajet.tuner_lib.experimental.interchange_utils import \
                    http_change_engine_status  # pylint: disable=import-outside-toplevel
                http_change_engine_status(self.config, status, global_step=self.global_steps)

    # #######################################
    # training loop
    # #######################################
    def fit(self):  # noqa: C901

        from omegaconf import OmegaConf
        from verl.utils.tracking import Tracking

        warm_up_process(self.config)

        verl_tracker = Tracking(
            project_name=self.config.trainer.project_name,
            experiment_name=self.config.trainer.experiment_name,
            default_backend=self.config.trainer.logger,
            config=OmegaConf.to_container(self.config, resolve=True),
        )

        self.global_steps = 0

        # load checkpoint and update weights before doing anything
        self._load_checkpoint()
        self.checkpoint_manager.update_weights(self.global_steps)
        self.checkpoint_manager.sleep_replicas()

        # [oc] swarm_mode is not compatible with `val_before_train` and `val_only`
        assert not (self.config.ajet.enable_swarm_mode and (self.config.ajet.trainer_common.val_before_train or self.config.ajet.trainer_common.val_only)), \
            "swarm_mode is not compatible with `val_before_train` and `val_only`"

        # perform validation before training
        # currently, we only support validation using the reward_function.
        if self.config.ajet.trainer_common.val_before_train and not self.config.ajet.enable_swarm_mode:
            val_metrics = self._validate()
            assert val_metrics, f"{val_metrics=}"
            pprint(f"Initial validation metrics: {val_metrics}")
            verl_tracker.log(data=val_metrics, step=self.global_steps)
            val_print_to_markdown_file_path = self.config.ajet.trainer_common.val_print_to_markdown_file_path
            if val_print_to_markdown_file_path:
                os.makedirs(os.path.dirname(val_print_to_markdown_file_path), exist_ok=True)
                with open(val_print_to_markdown_file_path, mode="a+", encoding="utf-8") as f:
                    f.write(str(val_metrics))
                    f.write('\n')
            if self.config.ajet.trainer_common.val_only:
                return

        if self.config.actor_rollout_ref.rollout.get("skip_rollout", False):
            rollout_skip = RolloutSkip(self.config, self.async_rollout_manager)
            rollout_skip.wrap_generate_sequences()

        # add tqdm
        progress_bar = tqdm(total=self.total_training_steps, initial=self.global_steps, desc="Training Progress")

        # we start from step 1
        self.global_steps += 1
        last_val_metrics = None
        self.max_steps_duration = 0

        prev_step_profile = False
        curr_step_profile = (
            self.global_steps in self.config.global_profiler.steps
            if self.config.global_profiler.steps is not None
            else False
        )
        next_step_profile = False

        for epoch in range(self.config.trainer.total_epochs):
            for batch_dict in self.train_dataloader:
                if hasattr(self.actor_rollout_wg, "async_calls_finalize_fn_exec"):
                    self.actor_rollout_wg.async_calls_finalize_fn_exec(blocking=False)

                metrics = {}
                timing_raw = {}

                with marked_timer("start_profile", timing_raw):
                    self._start_profiling(
                        not prev_step_profile and curr_step_profile
                        if self.config.global_profiler.profile_continuous_steps
                        else curr_step_profile
                    )

                batch_dict["index"] = torch.tensor(
                    [i for i in range(len(batch_dict["task_id"]))], dtype=torch.long,
                )

                batch: DataProto = DataProto.from_single_dict(batch_dict)

                # add uid to batch
                batch.non_tensor_batch["uid"] = np.array(
                    [str(uuid.uuid4()) for _ in range(len(batch.batch))], dtype=object
                )

                # # pop those keys for generation
                batch_keys_to_pop = ["index"]
                non_tensor_batch_keys_to_pop = [
                    "task_id",
                    "main_query",
                    "env_type",
                    "metadata",
                    "init_messages",
                ]
                gen_batch = batch.pop(
                    batch_keys=batch_keys_to_pop,
                    non_tensor_batch_keys=non_tensor_batch_keys_to_pop,
                )
                gen_batch = self._get_gen_batch(gen_batch)

                # pass global_steps to trace
                gen_batch.meta_info["global_steps"] = self.global_steps

                is_last_step = self.global_steps >= self.total_training_steps
                with marked_timer("step", timing_raw):
                    # generate a batch
                    logger.info("rollout step begin")
                    with marked_timer("gen", timing_raw, color="red"):
                        # assert self.async_rollout_mode
                        logger.info("wake up begin")
                        self.checkpoint_manager.update_weights(self.global_steps)
                        self._update_interchange_server_status_flag("ENGINE.ROLLING")
                        logger.info("wake up end")
                        if curr_step_profile:
                            self.async_rollout_manager.start_profile()
                        tasks: List[Task] = [
                            dict_to_ajet_task(dict(
                                task_id=gen_batch.non_tensor_batch["task_id"][i],
                                main_query=gen_batch.non_tensor_batch["main_query"][i],
                                env_type=gen_batch.non_tensor_batch["env_type"][i],
                                metadata=gen_batch.non_tensor_batch["metadata"][i],
                                init_messages=gen_batch.non_tensor_batch["init_messages"][i],
                            ))
                            for i in range(len(gen_batch))
                        ]
                        logger.info(
                            str(
                                [
                                    gen_batch.non_tensor_batch["task_id"][i]
                                    for i in range(len(gen_batch))
                                ]
                            )
                        )
                        logger.info("start fit rollout")
                        self.parallel_env.current_global_steps = self.global_steps
                        context_tracker_arr: List[SingleAgentContextTracker] = self.parallel_env.rollout(
                            tasks, mode="sample", epoch=f"train.{epoch}"
                        )

                        # from ajet import bp; bp("BATCH")

                        logger.info("end fit rollout")
                        gen_batch_output = self.parallel_env.to_dataproto(context_tracker_arr)
                        logger.info("end dataproto convertion")

                        success_rate = [
                            traj.reward_structure.success_rate for traj in context_tracker_arr
                        ]
                        madness_rate = [
                            traj.reward_structure.madness for traj in context_tracker_arr
                        ]
                        # reward = [traj.reward_structure.raw_reward for traj in context_tracker_arr]
                        llm_call_cnt = [traj.llm_call_cnt for traj in context_tracker_arr]
                        metrics.update(
                            {
                                "critic/llm_call_cnt": np.mean(llm_call_cnt),
                                "critic/madness_rate": np.mean(madness_rate),
                                "critic/success_rate": np.mean(success_rate),
                                "critic/real_success_rate": np.mean(
                                    context_tracker_arr[0].current_batch_success_rate
                                ),
                                "critic/real_reward": np.mean(
                                    context_tracker_arr[0].current_batch_reward
                                ),
                            }
                        )
                        save_trajectory_as_json_file(context_tracker_arr, self.global_steps, self.config, prefix="train")
                        update_metrics(context_tracker_arr, metrics, prefix="train_")
                        if self.config.ajet.execute_test:  # apply a test probe
                            from swanlab.data.run.main import get_run

                            from ajet.utils.testing_utils import \
                                _test_if_test_mode

                            run_info = get_run().public.json()  # type: ignore
                            data = {
                                "step": self.global_steps,
                                "reward_for_test_robot": metrics["critic/real_reward"],
                                "data_dashboard_url": run_info["cloud"]["experiment_url"],
                            }
                            _test_if_test_mode(key="reward_probe", value=data, config=self.config)

                        logger.info(
                            f"gen_batch_output.info batch.keys={gen_batch_output.batch.keys()}"
                        )
                        logger.info(
                            f"gen_batch_output.info non_tensor_batch.keys={gen_batch_output.non_tensor_batch.keys()}"
                        )
                        self._update_interchange_server_status_flag("ENGINE.WEIGHT_SYNCING")
                        self.checkpoint_manager.sleep_replicas()
                        if curr_step_profile:
                            self.async_rollout_manager.stop_profile()
                    logger.info("rollout step end")

                    batch.non_tensor_batch["uid"] = np.array(
                        [str(uuid.uuid4()) for _ in range(len(batch.batch))],
                        dtype=object,
                    )
                    discard_original_batch = self.config.ajet.enable_swarm_mode
                    batch = union_gen_batch_via_task_id(tasks, batch, gen_batch_output, discard_original_batch)

                    if self.config.algorithm.adv_estimator == AdvantageEstimator.REMAX:
                        with marked_timer("gen_max", timing_raw, color="purple"):
                            gen_baseline_batch = deepcopy(batch)
                            gen_baseline_batch.meta_info["do_sample"] = False
                            if curr_step_profile:
                                self.async_rollout_manager.start_profile()
                            gen_baseline_output = self.async_rollout_manager.generate_sequences(gen_baseline_batch)
                            self.checkpoint_manager.sleep_replicas()
                            if curr_step_profile:
                                self.async_rollout_manager.stop_profile()
                            batch = batch.union(gen_baseline_output)
                            # compute reward model score on batch
                            rm_scores = None
                            if self.use_rm and "rm_scores" not in batch.batch.keys():
                                batch_reward = self._compute_reward_colocate(batch)
                                batch = batch.union(batch_reward)

                            # Compute or extract reward for REMAX baseline
                            reward_baseline_tensor = batch.batch["rm_scores"].sum(dim=-1)

                            keys_to_pop = set(gen_baseline_output.batch.keys())
                            if rm_scores is not None:
                                keys_to_pop.update(rm_scores.batch.keys())
                            batch.pop(batch_keys=list(keys_to_pop))

                            batch.batch["reward_baselines"] = reward_baseline_tensor

                            del rm_scores, gen_baseline_batch, gen_baseline_output

                    batch.batch["response_mask"] = compute_response_mask(batch)

                    if "response_mask" not in batch.batch.keys():
                        batch.batch["response_mask"] = compute_response_mask(batch)
                    # Balance the number of valid tokens across DP ranks.
                    # NOTE: This usually changes the order of data in the `batch`,
                    # which won't affect the advantage calculation (since it's based on uid),
                    # but might affect the loss calculation (due to the change of mini-batching).
                    # TODO: Decouple the DP balancing and mini-batching.
                    if self.config.trainer.balance_batch:
                        self._balance_batch(batch, metrics=metrics)

                    # compute global_valid tokens
                    batch.meta_info["global_token_num"] = torch.sum(batch.batch["attention_mask"], dim=-1).tolist()

                    with marked_timer("reward", timing_raw, color="yellow"):
                        # compute reward model score
                        if self.use_rm and "rm_scores" not in batch.batch.keys():
                            batch_reward = self._compute_reward_colocate(batch)
                            batch = batch.union(batch_reward)

                            # extract reward_tensor and reward_extra_infos_dict for training
                            reward_tensor, reward_extra_infos_dict = extract_reward(batch)
                        else:
                            reward_tensor, reward_extra_infos_dict = compute_reward(batch, parse_reward_from_dataproto)

                        self_distillation_data = self._maybe_build_self_distillation_batch(
                            batch,
                            reward_tensor,
                            reward_extra_infos_dict,
                        )
                        if self_distillation_data is not None:
                            self_distillation_batch, self_distillation_metrics = self_distillation_data
                            batch = batch.union(self_distillation_batch)
                            metrics.update(self_distillation_metrics)

                    # recompute old_log_probs
                    # Operating Mode Selection:
                    # - Bypass mode: Sets old_log_probs = rollout_log_probs (2 policies: π_rollout, π_θ)
                    # - Decoupled mode: Recomputes old_log_probs as proximal anchor (3 policies: π_rollout, π_old, π_θ)
                    #   Note: π_old computed once per data batch, serves as stable reference during mini-batch updates
                    rollout_corr_config = self.config.algorithm.get("rollout_correction", None)
                    bypass_recomputing_logprobs = rollout_corr_config and rollout_corr_config.get("bypass_mode", False)
                    if bypass_recomputing_logprobs:  # Use `rollout_log_probs`
                        from verl.trainer.ppo.rollout_corr_helper import \
                            apply_bypass_mode  # pylint: disable=import-outside-toplevel
                        apply_bypass_mode(
                            batch=batch,
                            rollout_corr_config=rollout_corr_config,
                            policy_loss_config=self.config.actor_rollout_ref.actor.policy_loss,
                        )
                    else:  # Recompute old_log_probs
                        with marked_timer("old_log_prob", timing_raw, color="blue"):
                            old_log_prob, old_log_prob_mfu = self._compute_old_log_prob(batch)
                            entropys = old_log_prob.batch["entropys"]
                            response_masks = batch.batch["response_mask"]
                            actor_config = self.config.actor_rollout_ref.actor
                            entropy_agg = agg_loss(
                                loss_mat=entropys,
                                loss_mask=response_masks,
                                loss_agg_mode=actor_config.loss_agg_mode,
                                loss_scale_factor=actor_config.loss_scale_factor,
                            )
                            old_log_prob_metrics = {
                                "actor/entropy": entropy_agg.detach().item(),
                                "perf/mfu/actor_infer": old_log_prob_mfu,
                            }
                            metrics.update(old_log_prob_metrics)
                            old_log_prob.batch.pop("entropys")
                            if "routed_experts" in batch.batch and "routed_experts" in old_log_prob.batch:
                                raise ValueError(
                                    "Detected conflicting router replay configuration: "
                                    "router_replay.mode='R2' and enable_rollout_routing_replay=True "
                                    "cannot be enabled simultaneously. "
                                    "The enable_rollout_routing_replay option is only used in R3 mode; "
                                    "it should not be set when using R2 mode."
                                )
                            batch = batch.union(old_log_prob)
                            if "rollout_log_probs" in batch.batch.keys():
                                # TODO: we may want to add diff of probs too.
                                from verl.utils.debug.metrics import \
                                    calculate_debug_metrics  # pylint: disable=import-outside-toplevel

                                metrics.update(calculate_debug_metrics(batch))

                    assert "old_log_probs" in batch.batch, f'"old_log_prob" not in {batch.batch.keys()=}'

                    if self.use_reference_policy:
                        # compute reference log_prob
                        with marked_timer("ref", timing_raw, color="olive"):
                            if not self.ref_in_actor:
                                ref_log_prob = self.ref_policy_wg.compute_ref_log_prob(batch)
                            else:
                                ref_log_prob = self.actor_rollout_wg.compute_ref_log_prob(batch)
                            batch = batch.union(ref_log_prob)

                    # compute values
                    if self.use_critic:
                        with marked_timer("values", timing_raw, color="cyan"):
                            values = self.critic_wg.compute_values(batch)
                            batch = batch.union(values)

                    with marked_timer("adv", timing_raw, color="brown"):
                        # we combine with rule-based rm
                        reward_extra_infos_dict: dict[str, list]
                        batch.batch["token_level_scores"] = reward_tensor   # from compute_reward

                        if reward_extra_infos_dict:
                            batch.non_tensor_batch.update({k: np.array(v) for k, v in reward_extra_infos_dict.items()})
                        from ajet import bp; bp("KL")  # pylint: disable=no-name-in-module, import-outside-toplevel # noqa
                        # compute rewards. apply_kl_penalty if available
                        if self.config.algorithm.use_kl_in_reward:
                            batch, kl_metrics = apply_kl_penalty(
                                batch, kl_ctrl=self.kl_ctrl_in_reward, kl_penalty=self.config.algorithm.kl_penalty,
                            )
                            metrics.update(kl_metrics)
                        else:
                            batch.batch["token_level_rewards"] = batch.batch["token_level_scores"]

                        # Compute rollout correction: IS weights, rejection sampling, and metrics
                        # Only runs in decoupled mode (computes once per batch using stable π_old)
                        # In bypass mode, this is skipped - actor computes metrics from evolving π_θ vs π_rollout
                        if (
                            rollout_corr_config is not None and "rollout_log_probs" in batch.batch and not bypass_recomputing_logprobs  # Only in decoupled mode
                        ):
                            from verl.trainer.ppo.rollout_corr_helper import \
                                compute_rollout_correction_and_add_to_batch  # pylint: disable=import-outside-toplevel

                            # Compute IS weights, apply rejection sampling, compute metrics
                            batch, is_metrics = compute_rollout_correction_and_add_to_batch(batch, rollout_corr_config)
                            # IS and off-policy metrics already have rollout_corr/ prefix
                            metrics.update(is_metrics)

                        # compute advantages, executed on the driver process
                        norm_adv_by_std_in_grpo = self.config.algorithm.get(
                            "norm_adv_by_std_in_grpo", True
                        )  # GRPO adv normalization factor

                        batch = compute_advantage(
                            batch,
                            adv_estimator=self.config.algorithm.adv_estimator,
                            gamma=self.config.algorithm.gamma,
                            lam=self.config.algorithm.lam,
                            num_repeat=self.config.ajet.rollout.num_repeat,
                            norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
                            config=self.config.algorithm,
                        )

                    # update critic
                    if self.use_critic:
                        with marked_timer("update_critic", timing_raw, color="pink"):
                            critic_output = self._update_critic(batch)
                        critic_output_metrics = reduce_metrics(critic_output.meta_info["metrics"])
                        metrics.update(critic_output_metrics)

                    # implement critic warmup
                    if self.config.trainer.critic_warmup <= self.global_steps:
                        # update actor
                        with marked_timer("update_actor", timing_raw, color="red"):
                            actor_output = self._update_actor(batch)

                        # Check if the ESI (Elastic Server Instance)/training plan is close to expiration.
                        esi_close_to_expiration = should_save_ckpt_esi(
                            max_steps_duration=self.max_steps_duration,
                            redundant_time=self.config.trainer.esi_redundant_time,
                        )
                        # Check if the conditions for saving a checkpoint are met.
                        # The conditions include a mandatory condition (1) and
                        # one of the following optional conditions (2/3/4):
                        # 1. The save frequency is set to a positive value.
                        # 2. It's the last training step.
                        # 3. The current step number is a multiple of the save frequency.
                        # 4. The ESI(Elastic Server Instance)/training plan is close to expiration.
                        if self.config.trainer.save_freq > 0 and (
                            is_last_step
                            or self.global_steps % self.config.trainer.save_freq == 0
                            or esi_close_to_expiration
                        ):
                            if esi_close_to_expiration:
                                print("Force saving checkpoint: ESI instance expiration approaching.")
                            with marked_timer("save_checkpoint", timing_raw, color="green"):
                                self._save_checkpoint()

                        # update weights from trainer to rollout
                        with marked_timer("update_weights", timing_raw, color="red"):
                            self.checkpoint_manager.update_weights(self.global_steps)

                        actor_output_metrics = reduce_metrics(actor_output.meta_info["metrics"])
                        metrics.update(actor_output_metrics)

                    # Log rollout generations if enabled
                    rollout_data_dir = self.config.trainer.get("rollout_data_dir", None)
                    if rollout_data_dir:
                        self._log_rollout_data(batch, reward_extra_infos_dict, timing_raw, rollout_data_dir)

                # validate
                if (
                    self.config.trainer.test_freq > 0
                    and (is_last_step or self.global_steps % self.config.trainer.test_freq == 0)
                    and (not self.config.ajet.enable_swarm_mode)
                ):
                    with marked_timer("testing", timing_raw, color="green"):
                        val_metrics: dict = self._validate()
                        if is_last_step:
                            last_val_metrics = val_metrics
                    metrics.update(val_metrics)
                    val_print_to_markdown_file_path = self.config.ajet.trainer_common.val_print_to_markdown_file_path
                    if val_print_to_markdown_file_path:
                        os.makedirs(os.path.dirname(val_print_to_markdown_file_path), exist_ok=True)
                        with open(val_print_to_markdown_file_path, mode="a+", encoding="utf-8") as f:
                            f.write(str(val_metrics))
                            f.write('\n')

                with marked_timer("stop_profile", timing_raw):
                    next_step_profile = (
                        self.global_steps + 1 in self.config.global_profiler.steps
                        if self.config.global_profiler.steps is not None
                        else False
                    )
                    self._stop_profiling(
                        curr_step_profile and not next_step_profile
                        if self.config.global_profiler.profile_continuous_steps
                        else curr_step_profile
                    )
                    prev_step_profile = curr_step_profile
                    curr_step_profile = next_step_profile

                steps_duration = timing_raw["step"]
                self.max_steps_duration = max(self.max_steps_duration, steps_duration)

                # training metrics
                metrics.update(
                    {
                        "training/global_step": self.global_steps,
                        "training/epoch": epoch,
                    }
                )
                # collect metrics
                metrics.update(compute_data_metrics(batch=batch, use_critic=self.use_critic))
                # GDPO per-component reward metrics
                gdpo_reward_keys = self.config.algorithm.get("gdpo_reward_keys", None)
                if gdpo_reward_keys and self.config.algorithm.adv_estimator in ("gdpo", AdvantageEstimator.GDPO):
                    for key in gdpo_reward_keys:
                        if key in batch.non_tensor_batch:
                            vals = np.asarray(batch.non_tensor_batch[key], dtype=np.float32)
                            metrics[f"gdpo/{key}/mean"] = float(np.mean(vals))
                            metrics[f"gdpo/{key}/std"] = float(np.std(vals))
                            metrics[f"gdpo/{key}/max"] = float(np.max(vals))
                            metrics[f"gdpo/{key}/min"] = float(np.min(vals))
                metrics.update(compute_timing_metrics(batch=batch, timing_raw=timing_raw))
                # TODO: implement actual tflpo and theoretical tflpo
                n_gpus = self.resource_pool_manager.get_n_gpus()
                metrics.update(compute_throughout_metrics(batch=batch, timing_raw=timing_raw, n_gpus=n_gpus))
                # compute variance proxy metrics
                gradient_norm = metrics.get("actor/grad_norm", None)
                metrics.update(compute_variance_proxy_metrics(batch=batch, gradient_norm=gradient_norm))
                # Note: mismatch metrics (KL, PPL, etc.) are collected at line 1179 after advantage computation

                # this is experimental and may be changed/removed in the future in favor of a general-purpose one
                if isinstance(self.train_dataloader.sampler, AbstractCurriculumSampler):
                    self.train_dataloader.sampler.update(batch=batch)

                verl_tracker.log(data=metrics, step=self.global_steps)
                train_print_to_markdown_file_path = self.config.ajet.trainer_common.train_print_to_markdown_file_path
                if train_print_to_markdown_file_path:
                    os.makedirs(os.path.dirname(train_print_to_markdown_file_path), exist_ok=True)
                    with open(train_print_to_markdown_file_path, mode="a+", encoding="utf-8") as f:
                        f.write(str(metrics))
                        f.write('\n')
                progress_bar.update(1)
                self.global_steps += 1

                if (
                    hasattr(self.config.actor_rollout_ref.actor, "profiler")
                    and self.config.actor_rollout_ref.actor.profiler.tool == "torch_memory"
                ):
                    self.actor_rollout_wg.dump_memory_snapshot(
                        tag=f"post_update_step{self.global_steps}", sub_dir=f"step{self.global_steps}"
                    )

                if is_last_step:
                    if hasattr(self.actor_rollout_wg, "async_calls_finalize_fn_exec"):
                        self.actor_rollout_wg.async_calls_finalize_fn_exec(blocking=True)
                    pprint(f"Final validation metrics: {last_val_metrics}")
                    progress_bar.close()
                    return

    # #######################################
    # Validate
    # #######################################
    def _validate(self, merged: bool = False):
        data_source_lst = []
        reward_extra_infos_dict: dict[str, list] = defaultdict(list)

        # Lists to collect samples for the table
        sample_outputs = []
        sample_gts = []
        sample_scores = []
        sample_turns = []
        sample_uids = []

        for test_data in self.val_dataloader:
            test_data["index"] = torch.tensor([i for i in range(len(test_data["task_id"]))], dtype=torch.long)
            test_batch = DataProto.from_single_dict(test_data)

            # repeat test batch
            test_batch = test_batch.repeat(
                repeat_times=self.config.actor_rollout_ref.rollout.val_kwargs.n, interleave=True
            )

            ground_truths = [
                item.non_tensor_batch.get("reward_model", {}).get("ground_truth", None) for item in test_batch
            ]
            sample_gts.extend(ground_truths)

            batch_keys_to_pop = ["index"]
            non_tensor_batch_keys_to_pop = [
                "task_id", "main_query", "env_type", "metadata",
                "init_messages",
            ]
            if "extras" in test_batch.non_tensor_batch:
                non_tensor_batch_keys_to_pop.append("extras")
            test_gen_batch = test_batch.pop(
                batch_keys=batch_keys_to_pop, non_tensor_batch_keys=non_tensor_batch_keys_to_pop,
            )

            test_gen_batch = self._get_gen_batch(test_gen_batch)
            test_gen_batch.meta_info = {
                "eos_token_id": self.tokenizer.eos_token_id,
                "pad_token_id": self.tokenizer.pad_token_id,
                "recompute_log_prob": False,
                "do_sample": self.config.ajet.rollout.val_kwargs.do_sample,
                "validate": True,
                "global_steps": self.global_steps,
            }
            logger.info(f"test_gen_batch meta info: {test_gen_batch.meta_info}")

            self.checkpoint_manager.update_weights(self.global_steps)
            main_val_dataset = self.get_val_dataset()

            logger.info("Starting validate rollout")
            context_tracker_arr, tasks, val_metrics = self._rollout_val_dataset(
                target_dataset=main_val_dataset,
                target_dataset_name="main_val_dataset",
                mode="validate",
                epoch="test.1",
            )
            logger.info("Completed validate rollout")
            test_output_gen_batch = self.parallel_env.to_dataproto(context_tracker_arr)
            self.checkpoint_manager.sleep_replicas()

            # Store generated outputs
            output_ids = test_output_gen_batch.batch["responses"]
            output_texts = [
                self.tokenizer.decode(ids, skip_special_tokens=True) for ids in output_ids
            ]
            sample_outputs.extend(output_texts)

            test_batch.non_tensor_batch["uid"] = np.array(
                [str(uuid.uuid4()) for _ in range(len(test_batch.batch))],
                dtype=object,
            )
            tasks = tasks[: len(main_val_dataset)]
            discard_original_batch = self.config.ajet.enable_swarm_mode
            test_batch = union_gen_batch_via_task_id(tasks, test_batch, test_output_gen_batch, discard_original_batch)
            # test_batch = test_batch.union(test_output_gen_batch)
            test_batch.meta_info["validate"] = True

            # evaluate using reward_function
            result = parse_reward_from_dataproto(test_batch, return_dict=True)
            reward_tensor = result["reward_tensor"]
            scores = reward_tensor.sum(-1).cpu().tolist()
            sample_scores.extend(scores)
            sample_uids.extend(test_batch.non_tensor_batch["uid"])

            reward_extra_infos_dict["reward"].extend(scores)
            logger.info(
                f"len reward_extra_infos_dict['reward']: {len(reward_extra_infos_dict['reward'])}"
            )
            if "reward_extra_info" in result:
                for key, lst in result["reward_extra_info"].items():
                    reward_extra_infos_dict[key].extend(lst)
                    logger.info(
                        f"len reward_extra_infos_dict['{key}']: {len(reward_extra_infos_dict[key])}"
                    )

            # collect num_turns of each prompt
            if "__num_turns__" in test_batch.non_tensor_batch:
                sample_turns.append(test_batch.non_tensor_batch["__num_turns__"])

            data_source_lst.append(
                test_batch.non_tensor_batch.get("data_source", ["unknown"] * reward_tensor.shape[0])
            )
            break  # hack to escape the loop after one batch

        # self._maybe_log_val_generations(inputs=sample_inputs, outputs=sample_outputs, scores=sample_scores)

        # # dump generations
        # val_data_dir = self.config.trainer.get("validation_data_dir", None)
        # if val_data_dir:
        #     self._dump_generations(
        #         inputs=sample_inputs,
        #         outputs=sample_outputs,
        #         gts=sample_gts,
        #         scores=sample_scores,
        #         reward_extra_infos_dict=reward_extra_infos_dict,
        #         dump_path=val_data_dir,
        #     )

        for key_info, lst in reward_extra_infos_dict.items():
            assert len(lst) == 0 or len(lst) == len(sample_scores), f"{key_info}: {len(lst)=}, {len(sample_scores)=}"

        if merged:
            print("_merge_validation_results validate result will be merged")
            return {
                "data_sources": data_source_lst,
                "sample_uids": sample_uids,
                "sample_turns": sample_turns,
                "reward_extra_infos_dict": reward_extra_infos_dict,
            }
        data_sources = np.concatenate(data_source_lst, axis=0)
        return {**self._val_metrics_update(data_sources, sample_uids, reward_extra_infos_dict, sample_turns), **val_metrics}

    def _rollout_val_dataset(self, target_dataset, target_dataset_name, mode, epoch):
        """
        Evaluate a dataset by running rollouts and computing task completion metrics.

        Args:
            target_dataset: The dataset to evaluate
            target_dataset_name: Name for logging purposes
            mode: Evaluation mode ("sample" or "validate")
            epoch: Current epoch for logging

        Returns:
            Tuple of (ctx_trackers, tasks) containing trajectory results and task definitions
        """
        pass_n = self.config.ajet.trainer_common.val_pass_n

        tasks = []
        for _ in range(pass_n):
            tasks += [task for task in target_dataset]

        ctx_trackers = self.parallel_env.rollout(
            tasks=tasks, mode=mode, epoch=epoch
        )  # "sample" or "validate"
        task_results = {}
        for ctx_tracker in ctx_trackers:
            reward = ctx_tracker.reward_structure.raw_reward
            task_id = ctx_tracker.task_id
            if task_id not in task_results:
                task_results[task_id] = {}
                task_results[task_id]["reward_arr"] = []
                task_results[task_id]["tag_arr"] = []
            if reward >= 1:
                ctx_tracker.tag = "success"
            elif reward == 0:
                ctx_tracker.tag = "failure"
            else:
                ctx_tracker.tag = "half_success"
            task_results[task_id]["tag_arr"] += [ctx_tracker.tag]
            task_results[task_id]["reward_arr"] += [ctx_tracker.reward_structure.raw_reward]
            task_results[task_id]["scenario"] = task_id.split("_")[0]

        repeated_success_tasks = 0
        num_all_success_tasks = 0  # number of tasks that is successful among all n attempts
        num_pass_n_tasks = 0  # number of tasks that is successful at least once among n attempts
        for task_id, task_outcomes in task_results.items():
            # Calculate num_all_success_tasks  # The number of tasks where all were successful in n experiments
            # Calculate num_pass_n_tasks       # The number of tasks where at least one was successful in n experiments
            assert len(task_outcomes["tag_arr"]) == pass_n, f"expect {pass_n} attempts, but got {len(task_outcomes['tag_arr'])} attempts for task_id={task_id}."
            if all(tag == "success" for tag in task_outcomes["tag_arr"]):
                num_all_success_tasks += 1
            if any(tag == "success" for tag in task_outcomes["tag_arr"]):
                num_pass_n_tasks += 1
            repeated_success_tasks += task_outcomes["tag_arr"].count("success")

        # record logs
        for ctx_tracker in ctx_trackers:
            ctx_tracker.generate_log()

        rewards = [ctx_tracker.reward_structure.raw_reward for ctx_tracker in ctx_trackers]
        num_tasks = len(task_results)
        assert num_tasks == len(ctx_trackers) // pass_n

        val_metrics = {
            "global_steps": self.global_steps,
            "pass_n": pass_n,
            "total_tasks": len(task_results),
            "num_all_success_tasks": num_all_success_tasks,
            f"num_pass_n_tasks(pass@{pass_n})": num_pass_n_tasks,
            "task_pass_rate@1": repeated_success_tasks / (num_tasks * pass_n),
            f"task_pass_rate@{pass_n}": num_pass_n_tasks / num_tasks,
            f"task_pass_rate@{pass_n}-all-pass": num_all_success_tasks / num_tasks,
            "mean_reward": sum(rewards) / len(rewards) if rewards else 0,
            "std_reward": np.std(rewards) if rewards else 0,
        }
        for k in [2, 4, 8, 16]:
            if pass_n > k:
                num_pass_k = 0
                for task_id, task_outcomes in task_results.items():
                    if any(tag == "success" for tag in task_outcomes["tag_arr"][:k]):
                        num_pass_k += 1
                val_metrics[f"task_pass_rate@{k}"] = num_pass_k / num_tasks

        save_trajectory_as_json_file(ctx_trackers, self.global_steps, self.config, prefix="eval")
        update_metrics(ctx_trackers, val_metrics, prefix="eval_")
        print_dict(
            val_metrics,
            narrow=True,
            header=target_dataset_name,
            mod="evaluation",
        )

        val_metrics.update({"target_dataset_name": target_dataset_name})

        return ctx_trackers, tasks, val_metrics

    def get_val_dataset(self):
        from ajet.task_reader import RouterTaskReader

        task_reader = RouterTaskReader(
            self.config.ajet.task_reader.type,
            self.config.ajet.task_reader,
        )
        tasks = task_reader.get_validation_tasks()

        # clip validation tasks if val_max_num_task_each_validation is set
        val_max_num_task = self.config.ajet.trainer_common.val_max_num_task_each_validation
        if val_max_num_task is not None and len(tasks) > val_max_num_task:
            original_size = len(tasks)
            clip_method = self.config.ajet.trainer_common.val_max_num_task_clip_method
            if clip_method == "fix_seed_random_n":
                rng = np.random.RandomState(seed=42)  # pylint: disable=no-member
                indices = rng.choice(len(tasks), val_max_num_task, replace=False)
                tasks = [tasks[i] for i in sorted(indices)]
            elif clip_method == "random_n":
                indices = np.random.choice(len(tasks), val_max_num_task, replace=False)
                tasks = [tasks[i] for i in sorted(indices)]
            elif clip_method == "first_n":
                tasks = tasks[:val_max_num_task]
            else:
                raise ValueError(f"Unknown val_max_num_task_clip_method: {clip_method}, expected 'fix_seed_random_n', 'random_n', or 'first_n'")
            logger.info(f"Clipped validation dataset from {original_size} to {val_max_num_task} tasks using '{clip_method}'")

        self.main_val_dataset = tasks
        return self.main_val_dataset
