IPC=$1
GPU=$2
PORT=`expr 29300 + $GPU`
CUDA_VISIBLE_DEVICES=$GPU accelerate launch --num_processes=1 --main_process_port=$PORT ./main.py \
--subset "ucf101" \
--arch-name "conv4" \
--factor 1 \
--num-crop 1 \
--mipc $IPC \
--ipc $IPC \
--stud-name "conv4" \
--re-epochs 300 \
--scheduler 'cos' \
--mem \
--mix-type 'cutmix' \
--wandb_name "soft-ipc${IPC}-random"