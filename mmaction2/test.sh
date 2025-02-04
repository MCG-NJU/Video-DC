CUDA_VISIBLE_DEVICES=0,1,4,5 mim test mmaction $1 --checkpoint $2 --launcher pytorch --gpus 4
