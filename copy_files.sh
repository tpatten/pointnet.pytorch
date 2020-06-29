REMOTE_DIR=/home/tpatten/Code/pointnet.pytorch/results_new
LOCAL_DIR=/home/tpatten/Data/ICAS2020/Results/ablation_architecture_new

FILES=(XABF_batch64_nEpoch150_regLoss_dropout0-3_augmentation_splitLoss_symmetry_lr0-01_ls20_lg0-5_arch14
XBB_batch64_nEpoch150_regLoss_dropout0-3_augmentation_splitLoss_symmetry_lr0-01_ls20_lg0-5_arch14
XGPMF_batch64_nEpoch150_regLoss_dropout0-3_augmentation_splitLoss_symmetry_lr0-01_ls20_lg0-5_arch14
XGSF_batch64_nEpoch150_regLoss_dropout0-3_augmentation_splitLoss_symmetry_lr0-01_ls20_lg0-5_arch14
XMDF_batch64_nEpoch150_regLoss_dropout0-3_augmentation_splitLoss_symmetry_lr0-01_ls20_lg0-5_arch14
XSHSU_batch64_nEpoch150_regLoss_dropout0-3_augmentation_splitLoss_symmetry_lr0-01_ls20_lg0-5_arch14)
       
for i in "${FILES[@]}"
do
  scp tpatten@racer.acin.tuwien.ac.at:$REMOTE_DIR/${i}_149.pkl $LOCAL_DIR
done