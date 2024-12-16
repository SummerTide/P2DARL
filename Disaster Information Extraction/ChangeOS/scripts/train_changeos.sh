export CUDA_VISIBLE_DEVICES=3,5
NUM_GPUS=2

config_path='changeos_r101_align'
model_dir='./logs/resnet101_changeos_align_A6000'

python -m torch.distributed.launch --nproc_per_node=${NUM_GPUS} --master_port $RANDOM train.py \
    --config_path=${config_path} \
    --model_dir=${model_dir} \
    learning_rate.params.max_iters 60000 \
    train.num_iters 60000 \
    data.train.params.batch_size 8
