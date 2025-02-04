IPC=$1
GPU=$2
PORT=`expr 29500 + $GPU`
CUDA_VISIBLE_DEVICES=$GPU accelerate launch --num_processes=1 --main_process_port=$PORT ./main.py \
--subset "k400" \
--arch-name "conv4" \
--factor 1 \
--num-crop 10 \
--mipc 100 \
--ipc $IPC \
--stud-name "conv4" \
--re-epochs 300 \
--scheduler 'cos' \
--mem \
--mix-type 'cutmix' \
--wandb_name "k400-ipc${IPC}-rded"