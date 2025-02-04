# accelerate launch --num_processes=8 --main_process_port=29500 ./main.py \
CUDA_VISIBLE_DEVICES=1 python ./main.py \
--subset "ucf101" \
--arch-name "conv4" \
--factor 1 \
--num-crop 5 \
--ipc 0 \
--stud-name "conv4" \
--re-epochs 300 \
--scheduler 'cos' \
--mem \
--mix-type cutmix \
--wandb_name 'exp'