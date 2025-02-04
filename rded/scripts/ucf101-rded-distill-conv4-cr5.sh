IPC=$1
GPU=$2
PORT=`expr 29500 + $GPU`
CUDA_VISIBLE_DEVICES=$GPU accelerate launch --num_processes=1 --main_process_port=$PORT ./main.py \
--subset "ucf101" \
--arch-name "conv4" \
--factor 1 \
--num-crop 5 \
--tau 8 \
--inter-mode 'none' \
--mipc 70 \
--ipc $IPC \
--stud-name "conv4" \
--re-epochs 300 \
--scheduler 'cos' \
--mem \
--wandb_name "ipc${IPC}-rded"