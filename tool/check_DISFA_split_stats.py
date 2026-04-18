import numpy as np
from pathlib import Path

list_path_prefix = Path("../data/DISFA/list")

# 8 AU mặc định
AUs = ["AU1","AU2","AU4","AU6","AU9","AU12","AU25","AU26"]

def check_fold(fold):
    print(f"\n========== 📊 Fold {fold} ==========")
    # đọc train
    train_imgs = (list_path_prefix / f"DISFA_train_img_path_fold{fold}.txt").read_text().splitlines()
    train_labels = np.loadtxt(list_path_prefix / f"DISFA_train_label_fold{fold}.txt")
    # đọc test
    test_imgs = (list_path_prefix / f"DISFA_test_img_path_fold{fold}.txt").read_text().splitlines()
    test_labels = np.loadtxt(list_path_prefix / f"DISFA_test_label_fold{fold}.txt")

    # kiểm tra kích thước
    print(f"Train: {len(train_imgs):>6} ảnh | {train_labels.shape}")
    print(f"Test : {len(test_imgs):>6} ảnh | {test_labels.shape}")

    if len(train_imgs) != train_labels.shape[0]:
        print("⚠️  Số ảnh và số nhãn TRAIN không khớp!")
    if len(test_imgs) != test_labels.shape[0]:
        print("⚠️  Số ảnh và số nhãn TEST không khớp!")

    # Thống kê tỉ lệ dương (AU=1)
    pos_rate_train = train_labels.mean(axis=0)
    pos_rate_test = test_labels.mean(axis=0)

    print("\nAU | Train_Pos% | Test_Pos%")
    print("-- | ----------- | ----------")
    for i, au in enumerate(AUs):
        print(f"{au:<3}| {pos_rate_train[i]*100:10.2f}% | {pos_rate_test[i]*100:9.2f}%")

# Kiểm tra tất cả 3 fold
for f in range(1,4):
    check_fold(f)
