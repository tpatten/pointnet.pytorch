TBOARD_SOURCE_DIR=/home/tpatten/logs
TBOARD_TARGET_DIR=/home/tpatten/v4rtemp/tp/ICAS_2020/tensorboard_logs/XABF
MODELS_SOURCE_DIR=/home/tpatten/Code/pointnet.pytorch
MODELS_TARGET_DIR=/home/tpatten/v4rtemp/tp/ICAS_2020/checkpoints/XABF

FILES=(XABF_batch64_nEpoch85_regLoss_dropout0-3_augmentation_splitLoss_symmetry_lr0-01_ls20_lg0-5_arch1_clamped
       XABF_batch64_nEpoch85_regLoss_dropout0-3_augmentation_splitLoss_symmetry_lr0-01_ls20_lg0-5_arch1_jointSet10
       XABF_batch64_nEpoch85_regLoss_dropout0-3_augmentation_splitLoss_symmetry_lr0-01_ls20_lg0-5_arch1_jointSet11
       XABF_batch64_nEpoch85_regLoss_dropout0-3_augmentation_splitLoss_symmetry_lr0-01_ls20_lg0-5_arch1_jointSet12
       XABF_batch64_nEpoch85_regLoss_dropout0-3_augmentation_splitLoss_symmetry_lr0-01_ls20_lg0-5_arch1_jointSet13
       XABF_batch64_nEpoch85_regLoss_dropout0-3_augmentation_splitLoss_symmetry_lr0-01_ls20_lg0-5_arch1_jointSet2
       XABF_batch64_nEpoch85_regLoss_dropout0-3_augmentation_splitLoss_symmetry_lr0-01_ls20_lg0-5_arch1_jointSet2_special
       XABF_batch64_nEpoch85_regLoss_dropout0-3_augmentation_splitLoss_symmetry_lr0-01_ls20_lg0-5_arch1_jointSet3
       XABF_batch64_nEpoch85_regLoss_dropout0-3_augmentation_splitLoss_symmetry_lr0-01_ls20_lg0-5_arch1_jointSet4
       XABF_batch64_nEpoch85_regLoss_dropout0-3_augmentation_splitLoss_symmetry_lr0-01_ls20_lg0-5_arch1_jointSet5
       XABF_batch64_nEpoch85_regLoss_dropout0-3_augmentation_splitLoss_symmetry_lr0-01_ls20_lg0-5_arch1_jointSet6
       XABF_batch64_nEpoch85_regLoss_dropout0-3_augmentation_splitLoss_symmetry_lr0-01_ls20_lg0-5_arch1_jointSet7
       XABF_batch64_nEpoch85_regLoss_dropout0-3_augmentation_splitLoss_symmetry_lr0-01_ls20_lg0-5_arch1_jointSet8
       XABF_batch64_nEpoch85_regLoss_dropout0-3_augmentation_splitLoss_symmetry_lr0-01_ls20_lg0-5_arch1_jointSet9)

for i in "${FILES[@]}"
do
  sudo mv $TBOARD_SOURCE_DIR/$i/ $TBOARD_TARGET_DIR/
  sudo mv $MODELS_SOURCE_DIR/$i/ $MODELS_TARGET_DIR/
done