# Copyright 2023 The HuggingFace Inc. team. All rights reserved.
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
# Full training
nohup accelerate launch --config_file examples/accelerate_configs/deepspeed_zero3.yaml --num_processes 8 \
    --main_process_port 2501 --machine_rank 0 --main_process_ip 127.0.0.1 \
    examples/scripts/sft.py \
    --model_name_or_path /mnt/bn/videonasi18n/heyc/ckpts/Qwen2.5-3B-Instruct \
    --dataset_name /mnt/bn/videonasi18n/heyc/paper_agent_demo/data/train_agent/new_sft/sft1.jsonl \
    --learning_rate 1.0e-5 \
    --num_train_epochs 1 \
    --bf16 True \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 1 \
    --gradient_checkpointing \
    --logging_steps 50 \
    --save_steps 2000 \
    --max_seq_length 1024 \
    --weight_decay 0.01 \
    --warmup_ratio 0.01 \
    --output_dir /mnt/hdfs/foundation/agent/heyc/sft1/ \
    --attn_implementation "flash_attention_2" & 

# LoRA
python examples/scripts/sft.py \
    --model_name_or_path Qwen/Qwen2-0.5B \
    --dataset_name trl-lib/Capybara \
    --learning_rate 2.0e-4 \
    --num_train_epochs 1 \
    --packing \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 8 \
    --gradient_checkpointing \
    --logging_steps 25 \
    --eval_strategy steps \
    --eval_steps 100 \
    --use_peft \
    --lora_r 32 \
    --lora_alpha 16 \
    --output_dir Qwen2-0.5B-SFT \
    --push_to_hub
"""

from datasets import load_dataset
from transformers import AutoTokenizer

from trl import (
    ModelConfig, # trl/trl/trainer/model_config.py
    ScriptArguments, # trl/trl/utils.py
    SFTConfig, # trl/trl/trainer/sft_config.py transformers/src/transformers/training_args.py
    SFTTrainer, # trl/trl/trainer/sft_trainer.py
    TrlParser,
    get_kbit_device_map,
    get_peft_config, # trl/trl/trainer/utils.py
    get_quantization_config,
    DataCollatorForCompletionOnlyLM
)

import wandb

import os
if int(os.environ.get('LOCAL_RANK', 0)) == 0:
    wandb.init(
        project="paper agent",
    )


if __name__ == "__main__":
    parser = TrlParser((ScriptArguments, SFTConfig, ModelConfig))
    script_args, training_args, model_config = parser.parse_args_and_config()

    ################
    # Model init kwargs & Tokenizer
    ################
    quantization_config = get_quantization_config(model_config)
    model_kwargs = dict(
        revision=model_config.model_revision,
        trust_remote_code=model_config.trust_remote_code,
        attn_implementation=model_config.attn_implementation,
        torch_dtype=model_config.torch_dtype,
        use_cache=False if training_args.gradient_checkpointing else True,
        device_map=get_kbit_device_map() if quantization_config is not None else None,
        quantization_config=quantization_config,
    )
    training_args.model_init_kwargs = model_kwargs
    tokenizer = AutoTokenizer.from_pretrained(
        model_config.model_name_or_path, trust_remote_code=model_config.trust_remote_code, use_fast=True
    )

    ################
    # Dataset
    ################
    dataset = load_dataset("json", data_files={"train": script_args.dataset_name})
    response_template = "<|im_start|>assistant\n"
    collator = DataCollatorForCompletionOnlyLM(response_template, tokenizer=tokenizer) # 只计算assistant部分的loss

    ################
    # Training
    ################
    trainer = SFTTrainer(
        model=model_config.model_name_or_path,
        args=training_args,
        train_dataset=dataset[script_args.dataset_train_split],
        # eval_dataset=dataset[script_args.dataset_test_split],
        processing_class=tokenizer,
        peft_config=get_peft_config(model_config),
        data_collator=collator,
        # gradient_checkpointing_kwargs={'use_reentrant':False},
    )

    trainer.train()

    # Save and push to hub
    trainer.save_model(training_args.output_dir)
    if training_args.push_to_hub:
        trainer.push_to_hub(dataset_name=script_args.dataset_name)

"""
Args
ScriptArguments(
    dataset_name='/mnt/bn/videonasi18n/heyc/paper_agent_demo/data/agent_small/train.jsonl',
    dataset_train_split='train',
    dataset_test_split='test',
    config=None,
    gradient_checkpointing_use_reentrant=False,
    ignore_bias_buffers=False,
)

SFTConfig(
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
    bf16=True,
    bf16_full_eval=False,
    chars_per_token=<CHARS_PER_TOKEN>,
    data_seed=None,
    dataloader_drop_last=False,
    dataloader_num_workers=0,
    dataloader_persistent_workers=False,
    dataloader_pin_memory=True,
    dataloader_prefetch_factor=None,
    dataset_batch_size=1000,
    dataset_kwargs=None,
    dataset_num_proc=None,
    dataset_text_field=text,
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
    eval_packing=None,
    eval_steps=None,
    eval_strategy=no,
    eval_use_gather_object=False,
    evaluation_strategy=None,
    fp16=False,
    fp16_backend=auto,
    fp16_full_eval=False,
    fp16_opt_level=O1,
    fsdp=[],
    fsdp_config={'min_num_params': 0,
        'xla': False,
        'xla_fsdp_v2': False,
        'xla_fsdp_grad_ckpt': False
    },
    fsdp_min_num_params=0,
    fsdp_transformer_layer_cls_to_wrap=None,
    full_determinism=False,
    gradient_accumulation_steps=1,
    gradient_checkpointing=True,
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
    label_names=None,
    label_smoothing_factor=0.0,
    learning_rate=1e-05,
    length_column_name=length,
    load_best_model_at_end=False,
    local_rank=0,
    log_level=passive,
    log_level_replica=warning,
    log_on_each_node=True,
    logging_dir=/mnt/bn/videonasi18n/heyc/paper_agent_demo/ckpts/sft/runs/Oct25_04-02-09_n124-254-017,
    logging_first_step=False,
    logging_nan_inf_filter=True,
    logging_steps=1,
    logging_strategy=steps,
    lr_scheduler_kwargs={},
    lr_scheduler_type=linear,
    max_grad_norm=1.0,
    max_seq_length=None,
    max_steps=-1,
    metric_for_best_model=None,
    model_init_kwargs=None,
    mp_parameters=,
    neftune_noise_alpha=None,
    no_cuda=False,
    num_of_sequences=1024,
    num_train_epochs=1.0,
    optim=adamw_torch,
    optim_args=None,
    optim_target_modules=None,
    output_dir=/mnt/bn/videonasi18n/heyc/paper_agent_demo/ckpts/sft,
    overwrite_output_dir=False,
    packing=True,
    past_index=-1,
    per_device_eval_batch_size=8,
    per_device_train_batch_size=16,
    prediction_loss_only=False,
    push_to_hub=False,
    push_to_hub_model_id=None,
    push_to_hub_organization=None,
    push_to_hub_token=<PUSH_TO_HUB_TOKEN>,
    ray_scope=last,
    remove_unused_columns=True,
    report_to=['wandb'],
    restore_callback_states_from_checkpoint=False,
    resume_from_checkpoint=None,
    run_name=/mnt/bn/videonasi18n/heyc/paper_agent_demo/ckpts/sft,
    save_on_each_node=False,
    save_only_model=False,
    save_safetensors=True,
    save_steps=500,
    save_strategy=steps,
    save_total_limit=None,
    seed=42,
    skip_memory_metrics=True,
    split_batches=None,
    tf32=None,
    torch_compile=False,
    torch_compile_backend=None,
    torch_compile_mode=None,
    torch_empty_cache_steps=None,
    torchdynamo=None,
    tpu_metrics_debug=False,
    tpu_num_cores=None,
    use_cpu=False,
    use_ipex=False,
    use_legacy_prediction_loop=False,
    use_liger=False,
    use_liger_kernel=False,
    use_mps_device=False,
    warmup_ratio=0.0,
    warmup_steps=0,
    weight_decay=0.0,
)

ModelConfig(
    model_name_or_path='/mnt/bn/videonasi18n/heyc/ckpts/Qwen2.5-7B-Instruct',
    model_revision='main',
    torch_dtype=None,
    trust_remote_code=False,
    attn_implementation=None,
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
    use_bnb_nested_quant=False,
)
"""