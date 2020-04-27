DATASET=/home/tpatten/v4rtemp/datasets/HandTracking/HO3D_v2/
DATA_SUBSET=XABF
NEPOCH=85
BACTH_SIZE=64
LEARN_RATE=0.01
DROPOUT_P=0.3

python3 utils/train_gripper_pose_regressor.py --dataset $DATASET --data_subset $DATA_SUBSET --nepoch $NEPOCH --batch_size $BACTH_SIZE --learning_rate $LEARN_RATE --dropout_p $DROPOUT_P --splitloss  --data_augmentation --closing_symmetry --arch 1 --joint_set 2
#python3 utils/train_gripper_pose_regressor.py --dataset $DATASET --data_subset $DATA_SUBSET --nepoch $NEPOCH --batch_size $BACTH_SIZE --learning_rate $LEARN_RATE --dropout_p $DROPOUT_P --splitloss  --data_augmentation --closing_symmetry --arch 1 --joint_set 3 --tensorboard
#python3 utils/train_gripper_pose_regressor.py --dataset $DATASET --data_subset $DATA_SUBSET --nepoch $NEPOCH --batch_size $BACTH_SIZE --learning_rate $LEARN_RATE --dropout_p $DROPOUT_P --splitloss  --data_augmentation --closing_symmetry --arch 1 --joint_set 4 --tensorboard
#python3 utils/train_gripper_pose_regressor.py --dataset $DATASET --data_subset $DATA_SUBSET --nepoch $NEPOCH --batch_size $BACTH_SIZE --learning_rate $LEARN_RATE --dropout_p $DROPOUT_P --splitloss  --data_augmentation --closing_symmetry --arch 1 --joint_set 5 --tensorboard
#python3 utils/train_gripper_pose_regressor.py --dataset $DATASET --data_subset $DATA_SUBSET --nepoch $NEPOCH --batch_size $BACTH_SIZE --learning_rate $LEARN_RATE --dropout_p $DROPOUT_P --splitloss  --data_augmentation --closing_symmetry --arch 1 --joint_set 6 --tensorboard
#python3 utils/train_gripper_pose_regressor.py --dataset $DATASET --data_subset $DATA_SUBSET --nepoch $NEPOCH --batch_size $BACTH_SIZE --learning_rate $LEARN_RATE --dropout_p $DROPOUT_P --splitloss  --data_augmentation --closing_symmetry --arch 1 --joint_set 7 --tensorboard
#python3 utils/train_gripper_pose_regressor.py --dataset $DATASET --data_subset $DATA_SUBSET --nepoch $NEPOCH --batch_size $BACTH_SIZE --learning_rate $LEARN_RATE --dropout_p $DROPOUT_P --splitloss  --data_augmentation --closing_symmetry --arch 1 --joint_set 8 --tensorboard
#python3 utils/train_gripper_pose_regressor.py --dataset $DATASET --data_subset $DATA_SUBSET --nepoch $NEPOCH --batch_size $BACTH_SIZE --learning_rate $LEARN_RATE --dropout_p $DROPOUT_P --splitloss  --data_augmentation --closing_symmetry --arch 1 --joint_set 9 --tensorboard
#python3 utils/train_gripper_pose_regressor.py --dataset $DATASET --data_subset $DATA_SUBSET --nepoch $NEPOCH --batch_size $BACTH_SIZE --learning_rate $LEARN_RATE --dropout_p $DROPOUT_P --splitloss  --data_augmentation --closing_symmetry --arch 1 --joint_set 10 --tensorboard
#python3 utils/train_gripper_pose_regressor.py --dataset $DATASET --data_subset $DATA_SUBSET --nepoch $NEPOCH --batch_size $BACTH_SIZE --learning_rate $LEARN_RATE --dropout_p $DROPOUT_P --splitloss  --data_augmentation --closing_symmetry --arch 1 --joint_set 11 --tensorboard
#python3 utils/train_gripper_pose_regressor.py --dataset $DATASET --data_subset $DATA_SUBSET --nepoch $NEPOCH --batch_size $BACTH_SIZE --learning_rate $LEARN_RATE --dropout_p $DROPOUT_P --splitloss  --data_augmentation --closing_symmetry --arch 1 --joint_set 12 --tensorboard
#python3 utils/train_gripper_pose_regressor.py --dataset $DATASET --data_subset $DATA_SUBSET --nepoch $NEPOCH --batch_size $BACTH_SIZE --learning_rate $LEARN_RATE --dropout_p $DROPOUT_P --splitloss  --data_augmentation --closing_symmetry --arch 1 --joint_set 13 --tensorboard
