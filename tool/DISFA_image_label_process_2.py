import os
from pathlib import Path
import numpy as np
from tqdm import tqdm

# ---------------- PATH & CONFIG ----------------
label_path = Path('../data/DISFA/ActionUnit_Labels')
list_path_prefix = Path('../data/DISFA/list')
list_path_prefix.mkdir(parents=True, exist_ok=True)

part1 = ['SN002','SN010','SN001','SN026','SN027','SN032','SN030','SN009','SN016']
part2 = ['SN013','SN018','SN011','SN028','SN012','SN006','SN031','SN021','SN024']
part3 = ['SN003','SN029','SN023','SN025','SN008','SN005','SN007','SN017','SN004']

au_idx = [1, 2, 4, 6, 9, 12, 25, 26]

# ---------------- CORE FUNCTION ----------------
def read_subject_labels(fr: str) -> tuple[list[str], np.ndarray]:
    """Đọc nhãn 8 AU của 1 subject (có tqdm hiển thị tiến trình)."""
    fr_path = label_path / fr
    au1_path = fr_path / f'{fr}_au1.txt'
    with au1_path.open('r') as f:
        total_frame = sum(1 for _ in f)

    labels = np.zeros((total_frame, len(au_idx)), dtype=np.uint8)

    for col, au in enumerate(tqdm(au_idx, desc=f'[{fr}] AUs', leave=False, ncols=80)):
        p = fr_path / f'{fr}_au{au}.txt'
        if not p.is_file():
            continue
        intens = np.loadtxt(p, delimiter=',', usecols=1, dtype=np.int32, ndmin=1)
        if intens.shape[0] != total_frame:
            if intens.shape[0] < total_frame:
                pad = np.zeros(total_frame - intens.shape[0], dtype=intens.dtype)
                intens = np.concatenate([intens, pad], axis=0)
            else:
                intens = intens[:total_frame]
        labels[:, col] = (intens >= 2).astype(np.uint8)

    img_names = [f'{fr}/{i}.png' for i in range(total_frame)]
    return img_names, labels


def build_part(parts: list[str]) -> tuple[list[str], np.ndarray]:
    """Đọc toàn bộ subject trong 1 phần (có tqdm hiển thị % load)."""
    all_imgs, all_labels = [], []
    for fr in tqdm(parts, desc="Đọc subject", ncols=80):
        imgs, lbs = read_subject_labels(fr)
        all_imgs.extend(imgs)
        all_labels.append(lbs)
    return all_imgs, np.concatenate(all_labels, axis=0)

# ---------------- MAIN BUILD ----------------
print("🚀 Bắt đầu xử lý DISFA labels...")

part1_imgs, part1_labels = build_part(part1)
part2_imgs, part2_labels = build_part(part2)
part3_imgs, part3_labels = build_part(part3)

# ---- TEST ----
(list_path_prefix / 'DISFA_test_img_path_fold3.txt').write_text('\n'.join(part1_imgs) + '\n', encoding='utf-8')
np.savetxt(list_path_prefix / 'DISFA_test_label_fold3.txt', part1_labels, fmt='%d', delimiter=' ')

(list_path_prefix / 'DISFA_test_img_path_fold2.txt').write_text('\n'.join(part2_imgs) + '\n', encoding='utf-8')
np.savetxt(list_path_prefix / 'DISFA_test_label_fold2.txt', part2_labels, fmt='%d', delimiter=' ')

(list_path_prefix / 'DISFA_test_img_path_fold1.txt').write_text('\n'.join(part3_imgs) + '\n', encoding='utf-8')
np.savetxt(list_path_prefix / 'DISFA_test_label_fold1.txt', part3_labels, fmt='%d', delimiter=' ')

# ---- TRAIN ----
train1_imgs = part1_imgs + part2_imgs
train1_labels = np.concatenate([part1_labels, part2_labels], axis=0)
(list_path_prefix / 'DISFA_train_img_path_fold1.txt').write_text('\n'.join(train1_imgs) + '\n', encoding='utf-8')
np.savetxt(list_path_prefix / 'DISFA_train_label_fold1.txt', train1_labels, fmt='%d', delimiter=' ')

train2_imgs = part1_imgs + part3_imgs
train2_labels = np.concatenate([part1_labels, part3_labels], axis=0)
(list_path_prefix / 'DISFA_train_img_path_fold2.txt').write_text('\n'.join(train2_imgs) + '\n', encoding='utf-8')
np.savetxt(list_path_prefix / 'DISFA_train_label_fold2.txt', train2_labels, fmt='%d', delimiter=' ')

train3_imgs = part2_imgs + part3_imgs
train3_labels = np.concatenate([part2_labels, part3_labels], axis=0)
(list_path_prefix / 'DISFA_train_img_path_fold3.txt').write_text('\n'.join(train3_imgs) + '\n', encoding='utf-8')
np.savetxt(list_path_prefix / 'DISFA_train_label_fold3.txt', train3_labels, fmt='%d', delimiter=' ')

print("✅ Hoàn tất tạo label DISFA (có train/test cho 3 fold).")
