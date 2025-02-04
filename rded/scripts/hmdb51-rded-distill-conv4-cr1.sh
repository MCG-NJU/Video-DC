IPC=$1
GPU=$2
PORT=`expr 29500 + $GPU`
CUDA_VISIBLE_DEVICES=$GPU accelerate launch --num_processes=1 --main_process_port=$PORT ./main.py \
--subset "hmdb51" \
--arch-name "conv4" \
--factor 1 \
--num-crop 5 \
--tau 8 \
--mipc 70 \
--ipc $IPC \
--stud-name "conv4" \
--re-epochs 300 \
--scheduler 'cos' \
--mem \
--mix-type 'cutmix' \
--wandb_name "hmdb51-ipc${IPC}-rded"