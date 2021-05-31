#/bin/bash

RANDOM_SEED=$1
QUALITY=$2
set -e

# Register the model as a source root
export PYTHONPATH="$(pwd):${PYTHONPATH}"

#MODEL_DIR="/tmp/resnet_imagenet_${RANDOM_SEED}"

DATA_DIR="/data/mlperf-datasets/imn/tf_records"
MODEL_DIR="/dsta/mlperf-result/image_classification/resnet_imagenet_${RANDOM_SEED}"
TRAIN_EPOCH=41	#	41
WARMUP_EPOCH=5	#	5

#python3 official/resnet/imagenet_main.py $RANDOM_SEED --data_dir /imn/imagenet/combined/  \
#  --model_dir $MODEL_DIR --train_epochs 10000 --stop_threshold $QUALITY --batch_size 64 \
#  --version 1 --resnet_size 50 --epochs_between_evals 4

python ./official/resnet-tf2/resnet_ctl_imagenet_main.py \
--base_learning_rate=8.5 \
--batch_size=1024 \
--clean \
--data_dir=${DATA_DIR} \
--datasets_num_private_threads=32 \
--dtype=fp32 \
--device_warmup_steps=1 \
--noenable_device_warmup \
--enable_eager \
--noenable_xla \
--epochs_between_evals=4 \
--noeval_dataset_cache \
--eval_offset_epochs=2 \
--eval_prefetch_batchs=192 \
--label_smoothing=0.1 \
--lars_epsilon=0 \
--log_steps=125 \
--lr_schedule=polynomial \
--model_dir=${MODEL_DIR} \
--momentum=0.9 \
--num_accumulation_steps=2 \
--num_classes=1000 \
--num_gpus=8 \
--optimizer=LARS \
--noreport_accuracy_metrics \
--single_l2_loss_op \
--noskip_eval \
--steps_per_loop=1252 \
--target_accuracy=0.759 \
--notf_data_experimental_slack \
--tf_gpu_thread_mode=gpu_private \
--notrace_warmup \
--train_epochs=${TRAIN_EPOCH} \
--notraining_dataset_cache \
--training_prefetch_batchs=128 \
--nouse_synthetic_data \
--warmup_epochs=${WARMUP_EPOCH} \
--weight_decay=0.0002
# To run on 8xV100s, instead run:
#python3 official/resnet/imagenet_main.py $RANDOM_SEED --data_dir /imn/imagenet/combined/ \
#   --model_dir $MODEL_DIR --train_epochs 10000 --stop_threshold $QUALITY --batch_size 1024 \
#   --version 1 --resnet_size 50 --dtype fp16 --num_gpus 8 \
#   --epochs_between_evals 4
