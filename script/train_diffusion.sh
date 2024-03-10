# CUDA_VISIBLE_DEVICES=0 python train_diffuser.py --dataset halfcheetah-medium-replay-v2 --save_file_name 5m_samples_5ep.npz --dp_epsilon=5.0 &
# CUDA_VISIBLE_DEVICES=1 python train_diffuser.py --dataset halfcheetah-medium-replay-v2 --save_file_name 5m_samples_10ep.npz --dp_epsilon=10.0 &
# CUDA_VISIBLE_DEVICES=2 python train_diffuser.py --dataset halfcheetah-medium-replay-v2 --save_file_name 5m_samples_20ep.npz --dp_epsilon=20.0 &
# CUDA_VISIBLE_DEVICES=3 python train_diffuser.py --dataset halfcheetah-medium-replay-v2 --save_file_name 5m_samples_50ep.npz --dp_epsilon=50.0 &

CUDA_VISIBLE_DEVICES=0 python train_diffuser.py --dataset hopper-medium-replay-v2 --save_file_name hopper_5m_samples_5ep.npz --dp_epsilon=5.0 &
CUDA_VISIBLE_DEVICES=1 python train_diffuser.py --dataset hopper-medium-replay-v2 --save_file_name hopper_5m_samples_10ep.npz --dp_epsilon=10.0 &
CUDA_VISIBLE_DEVICES=2 python train_diffuser.py --dataset hopper-medium-replay-v2 --save_file_name hopper_5m_samples_20ep.npz --dp_epsilon=20.0 &
# CUDA_VISIBLE_DEVICES=3 python train_diffuser.py --dataset hopper-medium-replay-v2 --save_file_name hopper_5m_samples_50ep.npz --dp_epsilon=50.0 &

CUDA_VISIBLE_DEVICES=0 python train_diffuser.py --dataset hopper-medium-replay-v2 --save_file_name hopper_5m_samples_10000ep.npz --dp_epsilon=10000.0 &
CUDA_VISIBLE_DEVICES=1 python train_diffuser.py --dataset halfcheetah-medium-replay-v2 --save_file_name 5m_samples_10000ep.npz --dp_epsilon=10000.0 &

# pretrain
CUDA_VISIBLE_DEVICES=0 python train_diffuser.py

# finetune
CUDA_VISIBLE_DEVICES=0 python train_diffuser.py --load_checkpoint --dataset hopper-medium-replay-v2 --save_file_name hopper_5m_samples_5ep.npz --dp_epsilon=5.0
