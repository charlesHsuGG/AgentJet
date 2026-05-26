# Copyright 2024 Bytedance Ltd. and/or its affiliates
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
Note that we don't combine the main with ray_trainer as ray_trainer is used by other main.
"""

import atexit
import os
import socket

import hydra
import ray
from omegaconf import DictConfig, OmegaConf
from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy
from torch.utils.data import Dataset as TorchDataset
from verl.trainer import main_ppo
from verl.trainer.ppo.utils import need_critic, need_reference_policy
from verl.utils.config import validate_config
from verl.utils.dataset.rl_dataset import collate_fn
from verl.utils.device import is_cuda_available

# Create training and validation datasets.
from ajet.backbone.warm_up import warm_up_process
from ajet.task_reader import RouterTaskReader, task_to_standard_dataset
from ajet.utils.core_env_vars import get_runtime_env
from ajet.utils.launch_utils import set_loguru_default_color
from ajet.utils.process_dataset import create_rl_sampler

set_loguru_default_color()


@hydra.main(config_path="config", config_name="ppo_trainer", version_base=None)
def main(config: DictConfig) -> None:
    """Main entry point for PPO training with Hydra configuration management.

    Args:
        config: Hydra configuration dictionary containing training parameters.
    """
    run_ppo(config)


# Define a function to run the PPO-like training process
def run_ppo(config: DictConfig, task_runner_class=None) -> None:
    """Initialize Ray cluster and run distributed PPO training process.

    Args:
        config: Training configuration object containing all necessary parameters
                for distributed PPO training including Ray initialization settings,
                model paths, and training hyperparameters.
    """
    # Check if Ray is not initialized
    if not ray.is_initialized():
        # Initialize Ray with a local cluster configuration
        # Set environment variables in the runtime environment to control tokenizer parallelism,
        # NCCL debug level, VLLM logging level, and allow runtime LoRA updating
        # `num_cpus` specifies the number of CPU cores Ray can use, obtained from the configuration
        default_runtime_env = get_runtime_env(config)
        ray_init_kwargs = config.ray_kwargs.get("ray_init", {})
        runtime_env_kwargs = ray_init_kwargs.get("runtime_env", {})

        if config.transfer_queue.enable:
            # Add runtime environment variables for transfer queue
            runtime_env_vars = runtime_env_kwargs.get("env_vars", {})
            runtime_env_vars["TRANSFER_QUEUE_ENABLE"] = "1"
            runtime_env_kwargs["env_vars"] = runtime_env_vars

        allow_broadcast_env = ["HF_", "NVTE_", "CUDA_", "WANDB_", "TIKTOKEN_", "NCCL_", "VLLM_", "SGLANG_", "TORCH_"]
        default_env_vars = {
            key: value for key, value in os.environ.items()
            if any(key.startswith(prefix) for prefix in allow_broadcast_env)
        }
        if "env_vars" not in runtime_env_kwargs:
            runtime_env_kwargs["env_vars"] = {
                "TOKENIZERS_PARALLELISM": "true", "NCCL_DEBUG": "WARN",
                "VLLM_LOGGING_LEVEL": "WARN", **default_env_vars
            }
        else:
            runtime_env_kwargs["env_vars"].update({
                "TOKENIZERS_PARALLELISM": "true", "NCCL_DEBUG": "WARN",
                "VLLM_LOGGING_LEVEL": "WARN", **default_env_vars
            })

        runtime_env = OmegaConf.merge(default_runtime_env, runtime_env_kwargs)
        ray_init_kwargs = OmegaConf.create({**ray_init_kwargs, "runtime_env": runtime_env})
        print(f"ray init kwargs: {ray_init_kwargs}")
        ray.init(**OmegaConf.to_container(ray_init_kwargs))

    def on_shutdown():
        if ray.is_initialized():
            ray.shutdown()
        if config.ajet.enable_interchange_server:
            if config.ajet.enable_swarm_mode:
                from ajet.tuner_lib.experimental.interchange_utils import \
                    http_change_engine_status
                print("Changing engine status to OFFLINE before shutdown...")
                http_change_engine_status(config, "ENGINE.OFFLINE", global_step=0)

    atexit.register(on_shutdown)  # ray shutdown on exit

    if task_runner_class is None:
        nodes = ray.nodes()
        ray_head_node_name = os.environ.get("RAY_HEAD_NODE_NAME", None)
        try:
            target_node_id = next(node["NodeID"] for node in nodes if ray_head_node_name is not None and ray_head_node_name in node["NodeManagerHostname"])
            print(f"Scheduling main_task on node_id: {target_node_id}")
            task_runner_class = ray.remote(
                num_cpus=1, scheduling_strategy=NodeAffinitySchedulingStrategy(target_node_id, soft=False)
            )(TaskRunner)  # please make sure main_task is not scheduled on head
        except StopIteration:
            print(f"No node with {ray_head_node_name} in NodeManagerHostname found. The main task will be scheduled without node affinity.")
            task_runner_class = ray.remote(num_cpus=1)(TaskRunner)

    # Create a remote instance of the TaskRunner class, and
    # Execute the `run` method of the TaskRunner instance remotely and wait for it to complete
    if (
        is_cuda_available and config.global_profiler.tool == "nsys" and config.global_profiler.get("steps") is not None and len(config.global_profiler.get("steps", [])) > 0
    ):
        from verl.utils.import_utils import is_nvtx_available

        assert is_nvtx_available(), "nvtx is not available in CUDA platform. Please 'pip3 install nvtx'"
        nsight_options = OmegaConf.to_container(
            config.global_profiler.global_tool_config.nsys.controller_nsight_options
        )
        runner = task_runner_class.options(runtime_env={"nsight": nsight_options}).remote()
    else:
        runner = task_runner_class.remote()

    ray.get(runner.run.remote(config))

    # [Optional] get the path of the timeline trace file from the configuration, default to None
    # This file is used for performance analysis
    timeline_json_file = config.ray_kwargs.get("timeline_json_file", None)
    if timeline_json_file:
        ray.timeline(filename=timeline_json_file)


class TaskRunner(main_ppo.TaskRunner):
    """Ray remote class for executing distributed PPO training tasks.

    This class encapsulates the main training logic and runs as a Ray remote actor
    to enable distributed execution across multiple nodes and GPUs.
    """

    def run(self, config):
        """Execute the main PPO training workflow.

        This method sets up the distributed training environment, initializes
        workers, datasets, and reward functions, then starts the training process.

        Args:
            config: Training configuration object containing all parameters needed
                   for setting up and running the PPO training process.
        """
        # Print the initial configuration. `resolve=True` will evaluate symbolic values.
        from pprint import pprint

        from loguru import logger
        from verl.utils.fs import copy_to_local
        warm_up_process(config)

        logger.info(f"TaskRunner hostname: {socket.gethostname()}, PID: {os.getpid()}")
        pprint(OmegaConf.to_container(config, resolve=True))
        OmegaConf.resolve(config)

        actor_rollout_cls, ray_worker_group_cls = self.add_actor_rollout_worker(config)
        self.add_critic_worker(config)

        self.add_reward_model_resource_pool(config)

        # Add a reference policy worker if KL loss or KL reward is used.
        self.add_ref_policy_worker(config, actor_rollout_cls)

        # validate config
        validate_config(
            config=config,
            use_reference_policy=need_reference_policy(config),
            use_critic=need_critic(config),
        )

        # Download the checkpoint from HDFS to the local machine.
        # `use_shm` determines whether to use shared memory, which could lead to faster model loading if turned on
        local_path = copy_to_local(
            config.ajet.model.path, use_shm=config.actor_rollout_ref.model.get("use_shm", False)
        )

        # Instantiate the tokenizer and processor.
        from verl.utils import hf_processor, hf_tokenizer

        trust_remote_code = config.data.get("trust_remote_code", False)
        tokenizer = hf_tokenizer(local_path, trust_remote_code=trust_remote_code)
        # Used for multimodal LLM, could be None
        processor = hf_processor(local_path, trust_remote_code=trust_remote_code, use_fast=True)

        resource_pool_manager = self.init_resource_pool_mgr(config)

        task_reader = RouterTaskReader(config.ajet.task_reader.type, config.ajet.task_reader,)

        train_dataset: TorchDataset = task_to_standard_dataset(task_reader.generate_training_tasks)  # type: ignore
        val_dataset: TorchDataset = task_to_standard_dataset(task_reader.generate_validation_tasks)  # type: ignore
        train_sampler = create_rl_sampler(config.data, train_dataset)

        from ajet.backbone.trainer_verl import AjetRayPPOTrainer

        if config.ajet.enable_interchange_server:
            from ajet.tuner_lib.experimental.oai_model_server import \
                start_interchange_server
            start_interchange_server(config)

        # Initialize the PPO trainer.
        trainer = AjetRayPPOTrainer(
            config=config,
            tokenizer=tokenizer,
            processor=processor,
            role_worker_mapping=self.role_worker_mapping,
            resource_pool_manager=resource_pool_manager,
            ray_worker_group_cls=ray_worker_group_cls,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            collate_fn=collate_fn,
            train_sampler=train_sampler,
            device_name=config.trainer.device,
        )
        # Initialize the workers of the trainer.
        trainer.init_workers()
        # Start the training process.
        trainer.fit()


if __name__ == "__main__":
    main()  # pylint: disable=no-value-for-parameter
