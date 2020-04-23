DATASET=/home/tpatten/v4rtemp/datasets/HandTracking/HO3D_v2/
DATA_SUBSET=XABF
NEPOCH=85
BACTH_SIZE=64
LEARN_RATE=0.01
DROPOUT_P=0.3

#python3 utils/train_gripper_pose_regressor.py --dataset $DATASET --data_subset $DATA_SUBSET --nepoch $NEPOCH --batch_size $BACTH_SIZE --learning_rate $LEARN_RATE --dropout_p $DROPOUT_P --splitloss  --data_augmentation --closing_symmetry --tensorboard --arch 1
#python3 utils/train_gripper_pose_regressor.py --dataset $DATASET --data_subset $DATA_SUBSET --nepoch $NEPOCH --batch_size $BACTH_SIZE --learning_rate $LEARN_RATE --dropout_p $DROPOUT_P --splitloss  --data_augmentation --closing_symmetry --tensorboard --arch 2
#python3 utils/train_gripper_pose_regressor.py --dataset $DATASET --data_subset $DATA_SUBSET --nepoch $NEPOCH --batch_size $BACTH_SIZE --learning_rate $LEARN_RATE --dropout_p $DROPOUT_P --splitloss  --data_augmentation --closing_symmetry --tensorboard --arch 3
#python3 utils/train_gripper_pose_regressor.py --dataset $DATASET --data_subset $DATA_SUBSET --nepoch $NEPOCH --batch_size $BACTH_SIZE --learning_rate $LEARN_RATE --dropout_p $DROPOUT_P --splitloss  --data_augmentation --closing_symmetry --tensorboard --arch 4
#python3 utils/train_gripper_pose_regressor.py --dataset $DATASET --data_subset $DATA_SUBSET --nepoch $NEPOCH --batch_size $BACTH_SIZE --learning_rate $LEARN_RATE --dropout_p $DROPOUT_P --splitloss  --data_augmentation --closing_symmetry --tensorboard --arch 5
#python3 utils/train_gripper_pose_regressor.py --dataset $DATASET --data_subset $DATA_SUBSET --nepoch $NEPOCH --batch_size $BACTH_SIZE --learning_rate $LEARN_RATE --dropout_p $DROPOUT_P --splitloss  --data_augmentation --closing_symmetry --tensorboard --arch 6
#python3 utils/train_gripper_pose_regressor.py --dataset $DATASET --data_subset $DATA_SUBSET --nepoch $NEPOCH --batch_size $BACTH_SIZE --learning_rate $LEARN_RATE --dropout_p $DROPOUT_P --splitloss  --data_augmentation --closing_symmetry --tensorboard --arch 7
#python3 utils/train_gripper_pose_regressor.py --dataset $DATASET --data_subset $DATA_SUBSET --nepoch $NEPOCH --batch_size $BACTH_SIZE --learning_rate $LEARN_RATE --dropout_p $DROPOUT_P --splitloss  --data_augmentation --closing_symmetry --tensorboard --arch 8
#python3 utils/train_gripper_pose_regressor.py --dataset $DATASET --data_subset $DATA_SUBSET --nepoch $NEPOCH --batch_size $BACTH_SIZE --learning_rate $LEARN_RATE --dropout_p $DROPOUT_P --splitloss  --data_augmentation --closing_symmetry --tensorboard --arch 10
#python3 utils/train_gripper_pose_regressor.py --dataset $DATASET --data_subset $DATA_SUBSET --nepoch $NEPOCH --batch_size $BACTH_SIZE --learning_rate $LEARN_RATE --dropout_p $DROPOUT_P --splitloss  --data_augmentation --closing_symmetry --tensorboard --arch 11
python3 utils/train_gripper_pose_regressor.py --dataset $DATASET --data_subset $DATA_SUBSET --nepoch $NEPOCH --batch_size $BACTH_SIZE --learning_rate $LEARN_RATE --dropout_p $DROPOUT_P --splitloss  --data_augmentation --closing_symmetry --tensorboard --arch 12
python3 utils/train_gripper_pose_regressor.py --dataset $DATASET --data_subset $DATA_SUBSET --nepoch $NEPOCH --batch_size $BACTH_SIZE --learning_rate $LEARN_RATE --dropout_p $DROPOUT_P --splitloss  --data_augmentation --closing_symmetry --tensorboard --arch 13
