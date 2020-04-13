DATASET=/home/tpatten/v4rtemp/datasets/HandTracking/HO3D_v2/

# Combined loss
echo 'python3 utils/train_gripper_pose_regressor.py --dataset $DATASET --data_subset XABF --tensorboard'

# Model loss
echo 'python3 utils/train_gripper_pose_regressor.py --dataset $DATASET --splitloss --data_subset XABF --model_loss --tensorboard'

# Global and local augmentation
echo 'python3 utils/train_gripper_pose_regressor.py --dataset $DATASET --splitloss --data_subset XABF --data_augmentation --tensorboard'
