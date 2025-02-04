IPC=$1
GPU=$2
PORT=`expr 29500 + $GPU`
CUDA_VISIBLE_DEVICES=$GPU accelerate launch --num_processes=1 --main_process_port=$PORT ./main.py \
--subset "hmdb51" \
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
--wandb_name "1-hmdb51-ipc${IPC}-random"