#!/usr/bin/env bash
set -x
shopt -s extglob
PYTHONPATH="." CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 ./benchmarker/cli/l5/train.py \
--model_name_or_path $T5_MODEL_PATH \
--relative_bias_args='[{"type": "1d"}, {"type": "horizontal"}, {"type": "vertical"}]' \
--dropout_rate 0.15 \
--label_smoothing 0 \
--model_type=t5  \
--data_dir $BINARIZATION_OUT/train \
--val_data_dir $BINARIZATION_OUT/dev \
--test_data_dir $BINARIZATION_OUT/test \
--val_check_interval 0.20 \
--gpus 8 \
--num_workers 16 \
--train_batch_size 1 \
--eval_batch_size 2 \
--accumulate_grad_batches 8 \
--max_epochs 5 \
--output_dir $TRAINING_OUT \
--max_target_length 256 \
--eval_max_gen_length 256 \
--warmup_steps 100 \
--learning_rate 2e-4  \
--lr_scheduler constant \
--val_metric loss \
--do_train \
--do_predict \
--additional_data_fields doc_id label_name \
--segment_levels tokens pages \
--optimizer adamw \
--weight_decay 1e-5 \
--adam_epsilon 1e-8 \
--gradient_checkpointing \
--trim_batches \
--accelerator=ddp \
--seed 3 \
--max_source_length 1450 \
--early_stopping_patience 5 \
;

