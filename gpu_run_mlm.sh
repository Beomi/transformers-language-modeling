# export NCCL_IB_DISABLE=1
export BS=32
export NCCL_DEBUG=INFO

deepspeed run_mlm.py \
--seed 42 \
--model_type bert \
--tokenizer_name beomi/KcELECTRA-base \
--train_file ./sampled_20190101_20200611_v2.txt \
--num_train_epochs 2 \
--per_device_train_batch_size $BS \
--per_device_eval_batch_size $BS \
--do_train \
--output_dir ./test-bert-zero2 \
--fp16 \
--logging_first_step \
--max_seq_length 300 \
--deepspeed ./ds_zero2_1gpu.json  \
