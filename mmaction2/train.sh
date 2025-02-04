echo $1
CUDA_VISIBLE_DEVICES=0,1,4,5 mim train mmaction $1 --launcher pytorch --gpus 4
# mim train mmaction $1 --launcher pytorch --gpus 8
