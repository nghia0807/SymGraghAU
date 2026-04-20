import cv2
import numpy as np
import argparse
from pathlib import Path
from tqdm import tqdm
import face_alignment


def extract_landmarks(image_path, fa, img_size=224):
   
    image = cv2.imread(str(image_path))
    if image is None:
        return None

    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    preds = fa.get_landmarks(image_rgb)

    if preds is None or len(preds) == 0:
        return None

    # Lấy face đầu tiên, pixel coordinates (68, 2)
    landmarks = preds[0].astype(np.float32)

    # Normalize về [0, 1]
    landmarks[:, 0] /= img_size  # x
    landmarks[:, 1] /= img_size  # y
    landmarks = np.clip(landmarks, 0.0, 1.0)

    return landmarks


def main():
    parser = argparse.ArgumentParser(description='Extract facial landmarks for DISFA')
    parser.add_argument('--data-root', type=str, default='data/DISFA/img',
                        help='Thư mục chứa ảnh đã crop (default: data/DISFA/img)')
    parser.add_argument('--output-dir', type=str, default='data/DISFA/landmarks',
                        help='Thư mục output .npy (default: data/DISFA/landmarks)')
    parser.add_argument('--img-size', type=int, default=224,
                        help='Kích thước ảnh để normalize (default: 224)')
    parser.add_argument('--device', type=str, default='cpu', choices=['cpu', 'cuda'],
                        help='Device cho FAN (default: cpu)')
    parser.add_argument('--no-skip', action='store_true', default=False,
                        help='Xử lý lại ảnh dù .npy đã tồn tại')
    args = parser.parse_args()

    data_root  = Path(args.data_root)
    output_dir = Path(args.output_dir)
    skip_existing = not args.no_skip

    if not data_root.exists():
        print(f"ERROR: data-root không tồn tại: {data_root}")
        return

    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 65)
    print("  DISFA Landmark Extraction  (face_alignment FAN-68)")
    print("=" * 65)
    print(f"  Input     : {data_root}")
    print(f"  Output    : {output_dir}")
    print(f"  Device    : {args.device}")
    print(f"  Img size  : {args.img_size}")
    print(f"  Skip exist: {skip_existing}")
    print("=" * 65)

    # Khởi tạo FAN (lần đầu tải model ~180 MB về cache)
    print("\nInitializing FAN model...")
    fa = face_alignment.FaceAlignment(
        face_alignment.LandmarksType.TWO_D,
        device=args.device,
        flip_input=False
    )
    print("FAN ready.\n")

    subjects = sorted([d for d in data_root.iterdir() if d.is_dir()])
    print(f"Subjects found: {len(subjects)}\n")

    total = skipped = success = failed = 0
    failed_list = []

    for subject_dir in subjects:
        out_subj = output_dir / subject_dir.name
        out_subj.mkdir(parents=True, exist_ok=True)

        imgs = sorted(
            list(subject_dir.glob('*.png')) + list(subject_dir.glob('*.jpg'))
        )
        if not imgs:
            continue

        for img_path in tqdm(imgs, desc=f"  {subject_dir.name}", unit="img"):
            total += 1
            npy_path = out_subj / f"{img_path.stem}.npy"

            if skip_existing and npy_path.exists():
                skipped += 1
                continue

            lm = extract_landmarks(img_path, fa, img_size=args.img_size)
            if lm is not None:
                np.save(npy_path, lm)
                success += 1
            else:
                failed += 1
                failed_list.append(str(img_path))

    print("\n" + "=" * 65)
    print("  SUMMARY")
    print("=" * 65)
    print(f"  Total       : {total:>8,}")
    print(f"  Saved       : {success:>8,}")
    print(f"  Skipped     : {skipped:>8,}")
    print(f"  Failed      : {failed:>8,}  ({100.*failed/max(total,1):.2f}%)")
    print("=" * 65)

    if failed_list:
        log = output_dir / 'failed_detections.txt'
        log.write_text('\n'.join(failed_list))
        print(f"  Failed list : {log}")

    status = "PASSED" if failed / max(total, 1) < 0.05 else "WARNING: high failure rate"
    print(f"\n  {status}")
    print(f"  Output format: .npy  shape=(68,2)  dtype=float32  range=[0,1]")
    print("=" * 65)


if __name__ == '__main__':
    main()
