export CUDA_VISIBLE_DEVICES=4
NUM_GPUS=1
export PYTHONPATH=$PYTHONPATH:`pwd`

ckpt_path='./logs/resnet101_changeos_Titan/model-60000.pth'
config_path='changeos_r101_mlp'
save_path='/data1/gjc23/Code/changeOS/output/changeos_r101_mlp/'

python eval.py \
    --ckpt_path=${ckpt_path} \
    --config_path=${config_path} \
    --save_path=${save_path}