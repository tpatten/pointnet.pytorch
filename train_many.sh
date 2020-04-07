DATASET=/home/tpatten/v4rtemp/dataset/HandTracking/HO3D_v2/

echo 'python3 utils/train_gripper_pose_regressor.py --dataset $DATASET --data_subset XABF --tensorboard'

echo 'python3 utils/train_gripper_pose_regressor.py --dataset $DATASET --splitloss --data_subset XABF --model_loss --tensorboard'

echo 'python3 utils/train_gripper_pose_regressor.py --dataset $DATASET --splitloss --data_subset XABF --data_augmentation --tensorboard'
