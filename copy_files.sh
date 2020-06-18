REMOTE_DIR=/home/tpatten/Code/pointnet.pytorch/results
LOCAL_DIR=/home/tpatten/Data/ICAS2020/Results/ablation_hand_input

FILES=(XABF_batch64_nEpoch150_regLoss_dropout0-3_augmentation_splitLoss_symmetry_lr0-01_ls20_lg0-5_arch14_jointSet2)
#       XABF_batch64_nEpoch150_regLoss_dropout0-3_augmentation_splitLoss_symmetry_lr0-01_ls20_lg0-5_arch14_jointSet3
#       XABF_batch64_nEpoch150_regLoss_dropout0-3_augmentation_splitLoss_symmetry_lr0-01_ls20_lg0-5_arch14_jointSet4
#       XABF_batch64_nEpoch150_regLoss_dropout0-3_augmentation_splitLoss_symmetry_lr0-01_ls20_lg0-5_arch14_jointSet5
#       XABF_batch64_nEpoch150_regLoss_dropout0-3_augmentation_splitLoss_symmetry_lr0-01_ls20_lg0-5_arch14_jointSet6
#       XABF_batch64_nEpoch150_regLoss_dropout0-3_augmentation_splitLoss_symmetry_lr0-01_ls20_lg0-5_arch14_jointSet7
#       XABF_batch64_nEpoch150_regLoss_dropout0-3_augmentation_splitLoss_symmetry_lr0-01_ls20_lg0-5_arch14_jointSet8
#       XABF_batch64_nEpoch150_regLoss_dropout0-3_augmentation_splitLoss_symmetry_lr0-01_ls20_lg0-5_arch14_jointSet9
#       XABF_batch64_nEpoch150_regLoss_dropout0-3_augmentation_splitLoss_symmetry_lr0-01_ls20_lg0-5_arch14_jointSet10
#       XABF_batch64_nEpoch150_regLoss_dropout0-3_augmentation_splitLoss_symmetry_lr0-01_ls20_lg0-5_arch14_jointSet11
#       XABF_batch64_nEpoch150_regLoss_dropout0-3_augmentation_splitLoss_symmetry_lr0-01_ls20_lg0-5_arch14_jointSet12
#       XABF_batch64_nEpoch150_regLoss_dropout0-3_augmentation_splitLoss_symmetry_lr0-01_ls20_lg0-5_arch14_jointSet13)
       
for i in "${FILES[@]}"
do
  scp tpatten@racer.acin.tuwien.ac.at:$REMOTE_DIR/${i}_149.pkl $LOCAL_DIR
done
