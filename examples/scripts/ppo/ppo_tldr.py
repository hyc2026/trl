# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
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

import shutil

from accelerate import PartialState
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    HfArgumentParser,
    Trainer
)

from trl import ModelConfig, PPOConfig, PPOTrainer, ScriptArguments
from trl.trainer.utils import SIMPLE_CHAT_TEMPLATE
from typing import Optional


"""
python examples/scripts/ppo/ppo_tldr.py \
    --dataset_name trl-internal-testing/tldr-preference-sft-trl-style
    --dataset_test_split validation \
    --learning_rate 3e-6 \
    --output_dir models/minimal/ppo \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 64 \
    --total_episodes 30000 \
    --model_name_or_path EleutherAI/pythia-1b-deduped \
    --sft_model_path cleanrl/EleutherAI_pythia-1b-deduped__sft__tldr \
    --reward_model_path cleanrl/EleutherAI_pythia-1b-deduped__reward__tldr \
    --missing_eos_penalty 1.0 \
    --stop_token eos \
    --response_length 53

WANDB_DISABLED=true accelerate launch \
    --config_file examples/accelerate_configs/deepspeed_zero3.yaml \
    --num_processes 8 \
    --main_process_port 2501 \
    --machine_rank 0 \
    --main_process_ip 127.0.0.1 \
    examples/scripts/ppo/ppo_tldr.py \
    --dataset_name trl-internal-testing/tldr-preference-sft-trl-style \
    --dataset_test_split validation \
    --output_dir /mnt/bn/videonasi18n/heyc/paper_agent_demo/ckpts/ppo \
    --learning_rate 3e-6 \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 4 \
    --total_episodes 100 \
    --model_name_or_path /mnt/bn/videonasi18n/heyc/ckpts/Qwen2.5-7B-Instruct \
    --sft_model_path /mnt/bn/videonasi18n/heyc/ckpts/Qwen2.5-7B-Instruct \
    --reward_model_path /mnt/bn/videonasi18n/heyc/ckpts/Qwen2.5-7B-Instruct-rw \
    --local_rollout_forward_batch_size 4 \
    --missing_eos_penalty 1.0 \
    --attn_implementation "flash_attention_2" \
    --stop_token eos
"""

import wandb
import os
if int(os.environ.get('LOCAL_RANK', 0)) == 0:
    wandb.init(
        project="paper agent",
    )

class FixZero3CheckpointPPOTrainer(PPOTrainer):

    def save_model(self, output_dir: Optional[str] = None, _internal_call: bool = False):
        backup_model = self.model
        self.model = self.model.policy  # save only the policy

        Trainer.save_model(self, output_dir, _internal_call)

        self.model = backup_model

    def _save(self, output_dir: Optional[str] = None, state_dict=None):
        if self.is_deepspeed_enabled:
            state_dict = {name.removeprefix('policy.'): param for name, param in state_dict.items()
                          if name.startswith('policy.')}

        super()._save(output_dir, state_dict)

if __name__ == "__main__":
    parser = HfArgumentParser((ScriptArguments, PPOConfig, ModelConfig))
    script_args, training_args, model_config = parser.parse_args_into_dataclasses()
    # remove output_dir if exists
    shutil.rmtree(training_args.output_dir, ignore_errors=True)

    ################
    # Model & Tokenizer
    ################
    tokenizer = AutoTokenizer.from_pretrained(
        model_config.model_name_or_path,
        padding_side="left",
        trust_remote_code=model_config.trust_remote_code,
    )
    tokenizer.add_special_tokens({"pad_token": "[PAD]"})
    if tokenizer.chat_template is None:
        tokenizer.chat_template = SIMPLE_CHAT_TEMPLATE
    value_model = AutoModelForSequenceClassification.from_pretrained(
        training_args.reward_model_path, trust_remote_code=model_config.trust_remote_code, num_labels=1
    )
    reward_model = AutoModelForSequenceClassification.from_pretrained(
        training_args.reward_model_path, trust_remote_code=model_config.trust_remote_code, num_labels=1
    )
    ref_policy = AutoModelForCausalLM.from_pretrained(
        training_args.sft_model_path, trust_remote_code=model_config.trust_remote_code
    )
    policy = AutoModelForCausalLM.from_pretrained(
        training_args.sft_model_path, trust_remote_code=model_config.trust_remote_code
    )
    ################
    # Dataset
    ################
    dataset = load_dataset(script_args.dataset_name)
    train_dataset = dataset[script_args.dataset_train_split]
    eval_dataset = dataset[script_args.dataset_test_split]

    def prepare_dataset(dataset, tokenizer):
        """pre-tokenize the dataset before training; only collate during training"""

        def tokenize(element):
            input_ids = tokenizer.apply_chat_template(
                element["messages"][:1],
                padding=False,
                add_generation_prompt=True,
            )
            return {"input_ids": input_ids, "lengths": len(input_ids)}

        return dataset.map(
            tokenize,
            remove_columns=dataset.column_names,
            num_proc=training_args.dataset_num_proc,
        )

    # Compute that only on the main process for faster data processing.
    # see: https://github.com/huggingface/trl/pull/1255
    with PartialState().local_main_process_first():
        train_dataset = prepare_dataset(train_dataset, tokenizer)
        eval_dataset = prepare_dataset(eval_dataset, tokenizer)
        # filtering
        train_dataset = train_dataset.filter(lambda x: x["lengths"] <= 512, num_proc=training_args.dataset_num_proc)
        eval_dataset = eval_dataset.filter(lambda x: x["lengths"] <= 512, num_proc=training_args.dataset_num_proc)

    assert train_dataset[0]["input_ids"][-1] != tokenizer.eos_token_id, "The last token should not be an EOS token"
    ################
    # Training
    ################
    trainer = FixZero3CheckpointPPOTrainer(
        config=training_args,
        processing_class=tokenizer,
        policy=policy,
        ref_policy=ref_policy,
        reward_model=reward_model,
        value_model=value_model,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
    )
    trainer.train()

    # Save and push to hub
    trainer.save_model(training_args.output_dir)
    if training_args.push_to_hub:
        trainer.push_to_hub(dataset_name=script_args.dataset_name)

    trainer.generate_completions()
"""
ScriptArguments(
    dataset_name='trl-internal-testing/tldr-preference-sft-trl-style',
    dataset_train_split='train',
    dataset_test_split='validation',
    config=None,
    gradient_checkpointing_use_reentrant=False,
    ignore_bias_buffers=False
)
PPOConfig(
    _n_gpu=1,
    accelerator_config={
        'split_batches': False,
        'dispatch_batches': None,
        'even_batches': True,
        'use_seedable_sampler': True,
        'non_blocking': False,
        'gradient_accumulation_kwargs': None,
        'use_configured_state': False
    },
    adafactor=False,
    adam_beta1=0.9,
    adam_beta2=0.999,
    adam_epsilon=1e-08,
    auto_find_batch_size=False,
    batch_eval_metrics=False,
    batch_size=None,
    bf16=False,
    bf16_full_eval=False,
    cliprange=0.2,
    cliprange_value=0.2,
    data_seed=None,
    dataloader_drop_last=False,
    dataloader_num_workers=0,
    dataloader_persistent_workers=False,
    dataloader_pin_memory=True,
    dataloader_prefetch_factor=None,
    dataset_num_proc=None,
    ddp_backend=None,
    ddp_broadcast_buffers=None,
    ddp_bucket_cap_mb=None,
    ddp_find_unused_parameters=None,
    ddp_timeout=1800,
    debug=[],
    deepspeed=None,
    disable_tqdm=False,
    dispatch_batches=None,
    do_eval=False,
    do_predict=False,
    do_train=False,
    eval_accumulation_steps=None,
    eval_delay=0,
    eval_do_concat_batches=True,
    eval_on_start=False,
    eval_steps=None,
    eval_strategy=no,
    eval_use_gather_object=False,
    evaluation_strategy=None,
    exp_name=ppo_config,
    fp16=False,
    fp16_backend=auto,
    fp16_full_eval=False,
    fp16_opt_level=O1,
    fsdp=[],
    fsdp_config={
        'min_num_params': 0,
        'xla': False,
        'xla_fsdp_v2': False,
        'xla_fsdp_grad_ckpt': False
    },
    fsdp_min_num_params=0,
    fsdp_transformer_layer_cls_to_wrap=None,
    full_determinism=False,
    gamma=1.0,
    gradient_accumulation_steps=4,
    gradient_checkpointing=False,
    gradient_checkpointing_kwargs=None,
    greater_is_better=None,
    group_by_length=False,
    half_precision_backend=auto,
    hub_always_push=False,
    hub_model_id=None,
    hub_private_repo=False,
    hub_strategy=every_save,
    hub_token=<HUB_TOKEN>,
    ignore_data_skip=False,
    include_for_metrics=[],
    include_inputs_for_metrics=False,
    include_num_input_tokens_seen=False,
    include_tokens_per_second=False,
    jit_mode_eval=False,
    kl_coef=0.05,
    label_names=None,
    label_smoothing_factor=0.0,
    lam=0.95,
    learning_rate=3e-06,
    length_column_name=length,
    load_best_model_at_end=False,
    local_batch_size=None,
    local_mini_batch_size=None,
    local_rank=0,
    local_rollout_forward_batch_size=4,
    log_level=passive,
    log_level_replica=warning,
    log_on_each_node=True,
    logging_dir=/mnt/bn/videonasi18n/heyc/paper_agent_demo/ckpts/ppo/runs/Oct26_17-41-40_n124-174-132,
    logging_first_step=False,
    logging_nan_inf_filter=True,
    logging_steps=500,
    logging_strategy=steps,
    lr_scheduler_kwargs={},
    lr_scheduler_type=linear,
    max_grad_norm=1.0,
    max_steps=-1,
    metric_for_best_model=None,
    micro_batch_size=None,
    mini_batch_size=None,
    missing_eos_penalty=1.0,
    mp_parameters=,
    neftune_noise_alpha=None,
    no_cuda=False,
    num_mini_batches=1,
    num_ppo_epochs=4,
    num_sample_generations=10,
    num_total_batches=None,
    num_train_epochs=3.0,
    optim=adamw_torch,
    optim_args=None,
    optim_target_modules=None,
    output_dir=/mnt/bn/videonasi18n/heyc/paper_agent_demo/ckpts/ppo,
    overwrite_output_dir=False,
    past_index=-1,
    per_device_eval_batch_size=8,
    per_device_train_batch_size=4,
    prediction_loss_only=False,
    push_to_hub=False,
    push_to_hub_model_id=None,
    push_to_hub_organization=None,
    push_to_hub_token=<PUSH_TO_HUB_TOKEN>,
    ray_scope=last,
    remove_unused_columns=True,
    report_to=[],
    response_length=53,
    restore_callback_states_from_checkpoint=False,
    resume_from_checkpoint=None,
    reward_model_path=/mnt/bn/videonasi18n/heyc/ckpts/Qwen2.5-7B-Instruct-rw,
    run_name=/mnt/bn/videonasi18n/heyc/paper_agent_demo/ckpts/ppo,
    save_on_each_node=False,
    save_only_model=False,
    save_safetensors=True,
    save_steps=500,
    save_strategy=steps,
    save_total_limit=None,
    seed=42,
    sft_model_path=/mnt/bn/videonasi18n/heyc/ckpts/Qwen2.5-7B-Instruct,
    skip_memory_metrics=True,
    split_batches=None,
    stop_token=<STOP_TOKEN>,
    stop_token_id=None,
    temperature=0.7,
    tf32=None,
    torch_compile=False,
    torch_compile_backend=None,
    torch_compile_mode=None,
    torch_empty_cache_steps=None,
    torchdynamo=None,
    total_episodes=100,
    tpu_metrics_debug=False,
    tpu_num_cores=None,
    use_cpu=False,
    use_ipex=False,
    use_legacy_prediction_loop=False,
    use_liger_kernel=False,
    use_mps_device=False,
    vf_coef=0.1,
    warmup_ratio=0.0,
    warmup_steps=0,
    weight_decay=0.0,
    whiten_rewards=False,
    world_size=None,
)
ModelConfig(
    model_name_or_path='/mnt/bn/videonasi18n/heyc/ckpts/Qwen2.5-7B-Instruct',
    model_revision='main',
    torch_dtype=None,
    trust_remote_code=False,
    attn_implementation='flash_attention_2',
    use_peft=False,
    lora_r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    lora_target_modules=None,
    lora_modules_to_save=None,
    lora_task_type='CAUSAL_LM',
    use_rslora=False,
    load_in_8bit=False,
    load_in_4bit=False,
    bnb_4bit_quant_type='nf4',
    use_bnb_nested_quant=False
)
"""