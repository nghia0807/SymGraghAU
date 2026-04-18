# -*- coding: utf-8 -*-
"""
Crop & align DISFA bằng MTCNN rồi resize 224x224 (chuẩn paper ME-GraphAU).
- Đầu vào:  data_root/<subject>/<frame>.(png|jpg|jpeg)
- Đầu ra:   out_root/<subject>/<idx|name>.png (RGB, 224x224 mặc định)
- Log:
    + log_dir/crop_summary.csv   (tổng hợp theo subject + tổng)
    + log_dir/crop_failures.csv  (danh sách frame thất bại + lý do)
"""

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

import os
import csv
import math
import time
import argparse
from pathlib import Path

import cv2
import numpy as np
import torch
from facenet_pytorch import MTCNN
from tqdm import tqdm

# ================== DEFAULT CONFIG ==================
DATA_ROOT = r"D:\NhomKL_AU_Emo\DISFA\raw_frames"
OUT_CROP_ROOT = r"D:\NhomKL_AU_Emo\DISFA\img_crop"
LOG_DIR = r"D:\NhomKL_AU_Emo\DISFA\crop_logs"

IMG_SIZE = 224
MARGIN_RATIO = 0.35   # biên quanh khuôn mặt theo khoảng cách mắt-miệng
# ====================================================


def parse_args():
    p = argparse.ArgumentParser("DISFA cropper (MTCNN + align + prev-box reuse + skip-existing)")
    p.add_argument("--data-root", default=DATA_ROOT, help="Thư mục chứa raw_frames")
    p.add_argument("--out-root", default=OUT_CROP_ROOT, help="Thư mục lưu img_crop")
    p.add_argument("--log-dir", default=LOG_DIR, help="Thư mục lưu log CSV")
    p.add_argument("--img-size", type=int, default=IMG_SIZE, help="Kích thước ảnh đầu ra (vuông)")
    p.add_argument("--margin", type=float, default=MARGIN_RATIO, help="Tỷ lệ biên theo khoảng cách mắt-miệng")
    p.add_argument("--device", default=None, help="cuda:0 | cpu (mặc định: tự phát hiện)")
    p.add_argument("--verbose", action="store_true", help="In chi tiết quá trình")
    p.add_argument("--keep-name", action="store_true", help="Giữ tên file gốc khi lưu (mặc định: theo index)")

    # Reuse bbox trước đó nếu detect/align lỗi
    g1 = p.add_mutually_exclusive_group()
    g1.add_argument("--reuse-prev-box", dest="reuse_prev", action="store_true",
                    help="Khi detect lỗi, dùng lại bbox của frame trước (mặc định: bật).")
    g1.add_argument("--no-reuse-prev-box", dest="reuse_prev", action="store_false",
                    help="Tắt cơ chế dùng bbox trước đó.")
    p.set_defaults(reuse_prev=True)

    # Fallback khi không có landmarks
    g2 = p.add_mutually_exclusive_group()
    g2.add_argument("--allow-no-landmark-fallback", dest="allow_no_lmk_fallback", action="store_true",
                    help="Không có landmarks thì dùng bbox mặt mở rộng 30% (mặc định: bật).")
    g2.add_argument("--no-landmark-fallback", dest="allow_no_lmk_fallback", action="store_false",
                    help="Tắt fallback bbox khi thiếu landmarks.")
    p.set_defaults(allow_no_lmk_fallback=True)

    # Skip nếu file output đã có
    g3 = p.add_mutually_exclusive_group()
    g3.add_argument("--skip-existing", dest="skip_existing", action="store_true",
                    help="Bỏ qua frame nếu file output đã tồn tại (mặc định: bật).")
    g3.add_argument("--no-skip-existing", dest="skip_existing", action="store_false",
                    help="Luôn xử lý lại, không bỏ qua.")
    p.set_defaults(skip_existing=True)

    return p.parse_args()


def safe_mkdir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


def list_subject_dirs(root: Path):
    subs = []
    if not root.exists():
        print(f"[WARN] DATA_ROOT không tồn tại: {root}")
        return subs
    for p in sorted(root.iterdir()):
        if p.is_dir():
            subs.append(p)
    return subs


def expand_square_box(cx, cy, size, w, h):
    half = float(size) / 2.0
    x1 = int(max(0, math.floor(cx - half)))
    y1 = int(max(0, math.floor(cy - half)))
    x2 = int(min(w, math.ceil(cx + half)))
    y2 = int(min(h, math.ceil(cy + half)))
    # đảm bảo hợp lệ
    x1 = max(0, min(x1, w - 1))
    y1 = max(0, min(y1, h - 1))
    x2 = max(1, min(x2, w))
    y2 = max(1, min(y2, h))
    return x1, y1, x2, y2


def clamp_box(x1, y1, x2, y2, w, h):
    x1 = max(0, min(int(x1), w - 1))
    y1 = max(0, min(int(y1), h - 1))
    x2 = max(1, min(int(x2), w))
    y2 = max(1, min(int(y2), h))
    return x1, y1, x2, y2


def bbox_from_landmarks(landmarks, img_w, img_h, margin_ratio=0.35):
    """
    landmarks: (5, 2) [left_eye, right_eye, nose, left_mouth, right_mouth]
    """
    eyes = landmarks[0:2, :]
    mouth = landmarks[3:5, :]
    eyes_center = eyes.mean(axis=0)
    mouth_center = mouth.mean(axis=0)
    d = float(np.linalg.norm(eyes_center - mouth_center))  # khoảng cách mắt-miệng
    size = d * (1.0 + margin_ratio) * 2.2  # heuristic
    cx = float(np.mean([eyes_center[0], mouth_center[0]]))
    cy = float(np.mean([eyes_center[1], mouth_center[1]]))
    x1, y1, x2, y2 = expand_square_box(cx, cy, size, img_w, img_h)
    return x1, y1, x2, y2


def align_face_by_eyes(img_rgb, lmk):
    """
    Căn chỉnh (rotate) dựa trên 2 mắt để mắt nằm ngang.
    - img_rgb: HxWx3 (RGB)
    - lmk: (5,2), thứ tự: left_eye(0), right_eye(1), ...
    Trả về ảnh đã xoay (giữ nguyên kích thước).
    """
    left_eye = lmk[0]
    right_eye = lmk[1]
    dy = right_eye[1] - left_eye[1]
    dx = right_eye[0] - left_eye[0]
    angle = math.degrees(math.atan2(dy, dx))
    h, w = img_rgb.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), angle, 1.0)
    rotated = cv2.warpAffine(img_rgb, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    return rotated


def crop_by_box(img_rgb, box_xyxy, out_size):
    h, w = img_rgb.shape[:2]
    x1, y1, x2, y2 = clamp_box(*box_xyxy, w, h)
    if x2 <= x1 or y2 <= y1:
        return None
    face = img_rgb[y1:y2, x1:x2]
    if face.size == 0:
        return None
    face = cv2.resize(face, (out_size, out_size), interpolation=cv2.INTER_AREA)
    return face


def crop_one_image(img_bgr, mtcnn: MTCNN, img_size: int, margin_ratio: float,
                   allow_no_lmk_fallback: bool, do_align: bool, verbose: bool = False):
    """
    Trả về: (face_rgb_or_None, used_box_or_None, reason_or_None)
    - used_box: bbox (x1,y1,x2,y2) đã dùng để crop.
    """
    if img_bgr is None or img_bgr.size == 0:
        return None, None, "empty_image"

    h, w = img_bgr.shape[:2]
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    try:
        boxes, probs, lmks = mtcnn.detect(img_rgb, landmarks=True)
    except Exception as e:
        if verbose:
            print(f"[ERR] mtcnn.detect exception: {e}")
        return None, None, "mtcnn_exception"

    if boxes is None or len(boxes) == 0:
        return None, None, "no_face"

    # Ép kiểu
    try:
        boxes = np.asarray(boxes, dtype=np.float32)
    except Exception:
        return None, None, "boxes_cast_fail"

    if probs is not None:
        try:
            probs = np.asarray(probs, dtype=np.float32)
        except Exception:
            probs = None

    if lmks is not None:
        try:
            lmks = np.asarray(lmks, dtype=np.float32)  # (N,5,2)
        except Exception:
            lmks = None

    # Chọn mặt tốt nhất
    if probs is not None and probs.size > 0 and np.isfinite(probs).any():
        best_idx = int(np.nanargmax(probs))
    else:
        areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
        best_idx = int(np.argmax(areas))

    box = boxes[best_idx]
    lmk = None
    if lmks is not None and len(lmks) > best_idx:
        lmk = lmks[best_idx]  # (5,2)

    use_lmk = (
        lmk is not None
        and isinstance(lmk, np.ndarray)
        and lmk.dtype.kind in "fc"
        and lmk.shape == (5, 2)
        and np.all(np.isfinite(lmk))
    )

    # Align nếu có landmarks
    if do_align and use_lmk:
        img_rgb = align_face_by_eyes(img_rgb, lmk)

    # Tính bbox để crop
    if use_lmk:
        x1, y1, x2, y2 = bbox_from_landmarks(lmk, w, h, margin_ratio=margin_ratio)
    else:
        if not allow_no_lmk_fallback:
            return None, None, "no_landmarks"
        # fallback: dùng bbox mặt, mở rộng 30%
        bw = float(box[2] - box[0])
        bh = float(box[3] - box[1])
        cx = float(box[0] + bw / 2.0)
        cy = float(box[1] + bh / 2.0)
        size = max(bw, bh) * 1.3
        x1, y1, x2, y2 = expand_square_box(cx, cy, size, w, h)

    # Cắt và resize
    x1c, y1c, x2c, y2c = map(int, (x1, y1, x2, y2))
    if x2c <= x1c or y2c <= y1c:
        return None, None, "invalid_crop_box"

    face = img_rgb[y1c:y2c, x1c:x2c]
    if face.size == 0:
        return None, None, "empty_crop"
    face = cv2.resize(face, (img_size, img_size), interpolation=cv2.INTER_AREA)
    return face, (x1c, y1c, x2c, y2c), None  # RGB img_size x img_size


def main():
    args = parse_args()

    data_root = Path(args.data_root)
    out_root = Path(args.out_root)
    log_dir = Path(args.log_dir)
    img_size = int(args.img_size)
    margin_ratio = float(args.margin)
    allow_fallback_lmk = bool(args.allow_no_lmk_fallback)
    verbose = bool(args.verbose)
    keep_name = bool(args.keep_name)
    reuse_prev = bool(args.reuse_prev)
    skip_existing = bool(args.skip_existing)

    t0 = time.time()
    safe_mkdir(out_root)
    safe_mkdir(log_dir)
    summary_csv = log_dir / "crop_summary.csv"
    failure_csv = log_dir / "crop_failures.csv"

    # Thiết lập device
    if args.device is None:
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device

    print("========== DISFA Cropper (MTCNN + Align + Prev-Box Reuse + Skip-Existing) ==========")
    print(f"[INFO] PyTorch CUDA available: {torch.cuda.is_available()}")
    print(f"[INFO] Using device: {device}")
    print(f"[INFO] DATA_ROOT       = {data_root}")
    print(f"[INFO] OUT_CROP_ROOT   = {out_root}")
    print(f"[INFO] LOG_DIR         = {log_dir}")
    print(f"[INFO] IMG_SIZE        = {img_size}")
    print(f"[INFO] MARGIN_RATIO    = {margin_ratio}")
    print(f"[INFO] LMK_FALLBACK    = {allow_fallback_lmk}")
    print(f"[INFO] REUSE_PREV_BOX  = {reuse_prev}")
    print(f"[INFO] SKIP_EXISTING   = {skip_existing}")
    print("===============================================================================")

    # Khởi tạo MTCNN
    mtcnn = MTCNN(
        keep_all=True,
        post_process=False,
        device=device,
        # thresholds=(0.6, 0.7, 0.7),
    )

    subjects = list_subject_dirs(data_root)
    if not subjects:
        print("[WARN] Không tìm thấy subject nào, dừng.")
        return

    with open(failure_csv, "w", newline="", encoding="utf-8") as f_fail:
        fail_writer = csv.writer(f_fail, lineterminator="\n")
        fail_writer.writerow(["subject", "frame_name", "reason"])

        rows = [("subject", "total_frames", "ok_detect", "ok_reused",
                "skipped_existing", "fail", "ok_rate")]
        grand_total = grand_ok_detect = grand_ok_reused = grand_skipped = grand_fail = 0

        pbar_subjects = tqdm(subjects, desc="Subjects", unit="subj")
        for subj_dir in pbar_subjects:
            out_dir = out_root / subj_dir.name
            safe_mkdir(out_dir)

            frames = sorted([p for p in subj_dir.iterdir()
                             if p.is_file() and p.suffix.lower() in [".png", ".jpg", ".jpeg"]])
            total = len(frames)
            ok_detect = 0
            ok_reused = 0
            skipped_existing = 0
            fail_cnt = 0

            # Bbox trước đó cho subject này
            prev_box_xyxy = None

            pbar_frames = tqdm(frames, desc=f"{subj_dir.name}", unit="img", leave=False)
            for idx, f in enumerate(pbar_frames):
                # ========== SKIP nếu ảnh đã xử lý ==========
                if keep_name:
                    out_path = out_dir / f"{Path(f).stem}.png"
                else:
                    out_path = out_dir / f"{idx}.png"

                if skip_existing and out_path.exists():
                    skipped_existing += 1
                    if verbose:
                        print(f"[SKIP] {subj_dir.name}/{f.name} -> đã có {out_path.name}")
                    continue
                # ===========================================

                img_bgr = cv2.imread(str(f), cv2.IMREAD_COLOR)
                if img_bgr is None:
                    fail_cnt += 1
                    fail_writer.writerow([subj_dir.name, f.name, "imread_fail"])
                    if verbose:
                        print(f"[{subj_dir.name}] {f.name} -> imread_fail")
                    continue

                # 1) Thử detect/align như bình thường
                face_rgb, used_box, reason = crop_one_image(
                    img_bgr,
                    mtcnn=mtcnn,
                    img_size=img_size,
                    margin_ratio=margin_ratio,
                    allow_no_lmk_fallback=allow_fallback_lmk,
                    do_align=True,
                    verbose=verbose,
                )

                # 2) Nếu lỗi và có bật reuse_prev + có prev_box: dùng lại bbox trước đó để cứu
                if face_rgb is None and reuse_prev and prev_box_xyxy is not None:
                    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
                    # Nới nhẹ 10% mỗi cạnh
                    x1, y1, x2, y2 = prev_box_xyxy
                    w_box = x2 - x1
                    h_box = y2 - y1
                    pad_x = int(0.1 * w_box)
                    pad_y = int(0.1 * h_box)
                    reuse_box = (x1 - pad_x, y1 - pad_y, x2 + pad_x, y2 + pad_y)

                    face_from_prev = crop_by_box(img_rgb, reuse_box, img_size)
                    if face_from_prev is not None:
                        face_rgb = face_from_prev
                        used_box = clamp_box(*reuse_box, img_rgb.shape[1], img_rgb.shape[0])
                        reason = None  # đã cứu được
                        ok_reused += 1
                        prev_box_xyxy = used_box
                        if verbose:
                            print(f"[{subj_dir.name}] {f.name} -> REUSED prev_box")
                    else:
                        # Không cứu được bằng prev_box -> thất bại thật
                        fail_cnt += 1
                        fail_writer.writerow([subj_dir.name, f.name, reason or "unknown"])
                        if verbose:
                            print(f"[{subj_dir.name}] {f.name} -> FAIL ({reason})")
                        continue

                # 3) Nếu detect OK ngay từ đầu
                elif face_rgb is None:
                    fail_cnt += 1
                    fail_writer.writerow([subj_dir.name, f.name, reason or "unknown"])
                    if verbose:
                        print(f"[{subj_dir.name}] {f.name} -> FAIL ({reason})")
                    continue
                else:
                    ok_detect += 1
                    if used_box is not None:
                        prev_box_xyxy = used_box

                # 4) Lưu file
                cv2.imwrite(str(out_path), cv2.cvtColor(face_rgb, cv2.COLOR_RGB2BGR))

            ok_cnt = ok_detect + ok_reused
            ok_rate = (ok_cnt / total * 100.0) if total > 0 else 0.0
            rows.append((subj_dir.name, total, ok_detect, ok_reused, skipped_existing,
                         total - ok_cnt - skipped_existing, f"{ok_rate:.2f}%"))

            grand_total += total
            grand_ok_detect += ok_detect
            grand_ok_reused += ok_reused
            grand_skipped += skipped_existing
            grand_fail += (total - ok_cnt - skipped_existing)

            print(f"[SUMMARY] {subj_dir.name}: total={total}  ok_detect={ok_detect}  ok_reused={ok_reused}  "
                  f"skipped={skipped_existing}  fail={total - ok_cnt - skipped_existing}  ok_rate={ok_rate:.2f}%")

    # Ghi summary + tổng
    with open(summary_csv, "w", newline="", encoding="utf-8") as fp:
        cw = csv.writer(fp, lineterminator="\n")
        cw.writerows(rows)
        cw.writerow(("", "", "", "", "", "", ""))
        total_ok = grand_ok_detect + grand_ok_reused
        total_rate = (total_ok / grand_total * 100.0) if grand_total > 0 else 0.0
        cw.writerow(("__TOTAL__", grand_total, grand_ok_detect, grand_ok_reused,
                     grand_skipped, grand_fail, f"{total_rate:.2f}%"))

    dt = time.time() - t0
    print("===============================================================================")
    print(f"[DONE] Time: {dt:.1f}s")
    print(f"[DONE] Summary CSV : {summary_csv}")
    print(f"[DONE] Failures CSV: {failure_csv}")
    print(f"[DONE] Total={grand_total}  OK_detect={grand_ok_detect}  OK_reused={grand_ok_reused}  "
          f"SKIPPED={grand_skipped}  FAIL={grand_fail}  OK_rate={( (grand_ok_detect+grand_ok_reused)/grand_total*100.0 if grand_total>0 else 0.0):.2f}%")
    print("===============================================================================")


if __name__ == "__main__":
    main()
