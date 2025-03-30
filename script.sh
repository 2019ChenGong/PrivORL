# CUDA_VISIBLE_DEVICES=2 python cql_trajectory.py --config synther/corl/yaml/cql/maze2d/medium-dense-v1.yaml --checkpoints_path a_test_cql_trajectory

CUDA_VISIBLE_DEVICES=0 python cql_trajectory.py --config synther/corl/yaml/cql/maze2d/medium-dense-v1.yaml &
CUDA_VISIBLE_DEVICES=0 python iql_trajectory.py --config synther/corl/yaml/iql/maze2d/medium-dense-v1.yaml &
CUDA_VISIBLE_DEVICES=1 python edac_trajectory.py --config synther/corl/yaml/edac/maze2d/medium-dense-v1.yaml &
CUDA_VISIBLE_DEVICES=1 python td3_bc_trajectory.py --config synther/corl/yaml/td3_bc/maze2d/medium-dense-v1.yaml &