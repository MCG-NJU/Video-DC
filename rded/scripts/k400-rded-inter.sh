IPC=$1
GPU=$2
PORT=`expr 29600 + $GPU`
CUDA_VISIBLE_DEVICES=$GPU accelerate launch --num_processes=1 --main_process_port=$PORT ./main.py \
--subset "k400" \
--arch-name "conv4" \
--factor 1 \
--num-crop 10 \
--tau 8 \
--inter-mode 'sample' \
--mipc 70 \
--ipc $IPC \
--stud-name "conv4" \
--re-epochs 300 \
--scheduler 'cos' \
--mem \
--mix-type 'cutmix' \
--loss-type 'ce' \
--wandb_name "k400-16s-ipc${IPC}-rded"