import numpy as np
import random
from PIL import Image
from torch.utils.data import Dataset
import os
import torch


def make_dataset(image_list, label_list, au_relation=None):
    len_ = len(image_list)
    if au_relation is not None:
        images = [(image_list[i].strip(),  label_list[i, :],au_relation[i,:]) for i in range(len_)]
    else:
        images = [(image_list[i].strip(),  label_list[i, :]) for i in range(len_)]
    return images


def pil_loader(path):
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')


def default_loader(path):
    return pil_loader(path)


class BP4D(Dataset):
    def __init__(self, root_path, train=True, fold = 1, transform=None, crop_size = 224, stage=1, loader=default_loader):

        assert fold>0 and fold <=3, 'The fold num must be restricted from 1 to 3'
        assert stage>0 and stage <=2, 'The stage num must be restricted from 1 to 2'
        self._root_path = root_path
        self._train = train
        self._stage = stage
        self._transform = transform
        self.crop_size = crop_size
        self.loader = loader
        self.img_folder_path = os.path.join(root_path,'img')
        if self._train:
            # img
            train_image_list_path = os.path.join(root_path, 'list', 'BP4D_train_img_path_fold' + str(fold) +'.txt')
            train_image_list = open(train_image_list_path).readlines()
            # img labels
            train_label_list_path = os.path.join(root_path, 'list', 'BP4D_train_label_fold' + str(fold) + '.txt')
            train_label_list = np.loadtxt(train_label_list_path)

            # AU relation
            if self._stage == 2:
                au_relation_list_path = os.path.join(root_path, 'list', 'BP4D_train_AU_relation_fold' + str(fold) + '.txt')
                au_relation_list = np.loadtxt(au_relation_list_path)
                self.data_list = make_dataset(train_image_list, train_label_list, au_relation_list)
            else:
                self.data_list = make_dataset(train_image_list, train_label_list)

        else:
            # img
            test_image_list_path = os.path.join(root_path, 'list', 'BP4D_test_img_path_fold' + str(fold) + '.txt')
            test_image_list = open(test_image_list_path).readlines()

            # img labels
            test_label_list_path = os.path.join(root_path, 'list', 'BP4D_test_label_fold' + str(fold) + '.txt')
            test_label_list = np.loadtxt(test_label_list_path)
            self.data_list = make_dataset(test_image_list, test_label_list)

    def __getitem__(self, index):
        if self._stage == 2 and self._train:
            img, label, au_relation = self.data_list[index]
            img = self.loader(os.path.join(self.img_folder_path, img))

            w, h = img.size
            offset_y = random.randint(0, h - self.crop_size)
            offset_x = random.randint(0, w - self.crop_size)
            flip = random.randint(0, 1)
            if self._transform is not None:
                img = self._transform(img, flip, offset_x, offset_y)
            return img, label, au_relation
        else:
            img, label = self.data_list[index]
            img = self.loader(os.path.join(self.img_folder_path, img))

            if self._train:
                w, h = img.size
                offset_y = random.randint(0, h - self.crop_size)
                offset_x = random.randint(0, w - self.crop_size)
                flip = random.randint(0, 1)
                if self._transform is not None:
                    img = self._transform(img, flip, offset_x, offset_y)
            else:
                if self._transform is not None:
                    img = self._transform(img)
            return img, label

    def __len__(self):
        return len(self.data_list)


class DISFA(Dataset):
    def __init__(self, root_path, train=True, fold = 1, transform=None, crop_size = 224, stage=1, loader=default_loader):

        assert fold>0 and fold <=3, 'The fold num must be restricted from 1 to 3'
        assert stage>0 and stage <=2, 'The stage num must be restricted from 1 to 2'
        self._root_path = root_path
        self._train = train
        self._stage = stage
        self._transform = transform
        self.crop_size = crop_size
        self.loader = loader
        self.img_folder_path = os.path.join(root_path,'img')
        if self._train:
            # img
            train_image_list_path = os.path.join(root_path, 'list', 'DISFA_train_img_path_fold' + str(fold) + '.txt')
            train_image_list = open(train_image_list_path).readlines()
            # img labels
            train_label_list_path = os.path.join(root_path, 'list', 'DISFA_train_label_fold' + str(fold) + '.txt')
            train_label_list = np.loadtxt(train_label_list_path)

            # AU relation
            if self._stage == 2:
                au_relation_list_path = os.path.join(root_path, 'list', 'DISFA_train_AU_relation_fold' + str(fold) + '.txt')
                au_relation_list = np.loadtxt(au_relation_list_path)
                self.data_list = make_dataset(train_image_list, train_label_list, au_relation_list)
            else:
                self.data_list = make_dataset(train_image_list, train_label_list)

        else:
            # img
            test_image_list_path = os.path.join(root_path, 'list', 'DISFA_test_img_path_fold' + str(fold) + '.txt')
            test_image_list = open(test_image_list_path).readlines()

            # img labels
            test_label_list_path = os.path.join(root_path, 'list', 'DISFA_test_label_fold' + str(fold) + '.txt')
            test_label_list = np.loadtxt(test_label_list_path)
            self.data_list = make_dataset(test_image_list, test_label_list)

    def __getitem__(self, index):
        if self._stage == 2 and self._train:
            img, label, au_relation = self.data_list[index]
            img = self.loader(os.path.join(self.img_folder_path, img))

            w, h = img.size
            offset_y = random.randint(0, h - self.crop_size)
            offset_x = random.randint(0, w - self.crop_size)
            flip = random.randint(0, 1)
            if self._transform is not None:
                img = self._transform(img, flip, offset_x, offset_y)
            return img, label, au_relation
        else:
            img, label = self.data_list[index]

            # path = os.path.normpath(os.path.join(self.img_folder_path, img))
            # print(f"[DEBUG] checking: {path}")
            # if not os.path.isfile(path):
            #     print(f"[ERROR] Missing file: {path}")
            # else:
            #     print(f"[OK] Exists: {path}") 
            
            img = self.loader( os.path.normpath(os.path.join(self.img_folder_path, img)) )
            
            if self._train:
                w, h = img.size
                offset_y = random.randint(0, h - self.crop_size)
                offset_x = random.randint(0, w - self.crop_size)
                flip = random.randint(0, 1)
                if self._transform is not None:
                    img = self._transform(img, flip, offset_x, offset_y)
            else:
                if self._transform is not None:
                    img = self._transform(img)
            return img, label

    def __len__(self):
        return len(self.data_list)


# ── DISFA_Landmark ────────────────────────────────────────────────────────────

class DISFA_Landmark(Dataset):
    """
    Dataset DISFA có thêm facial landmarks.

    Giống hệt class DISFA cho Stage 1, nhưng __getitem__ trả về thêm
    tensor landmark (68, 2) float32 được load từ file .npy pre-computed.

    Args:
        root_path      : đường dẫn tới data/DISFA
        landmark_root  : đường dẫn tới thư mục chứa .npy
                         (default: <root_path>/landmarks)
        train, fold, transform, crop_size, loader : giống class DISFA
        landmark_zeros_on_miss : nếu True, khi thiếu file .npy trả về
                                 zeros thay vì raise lỗi (an toàn khi
                                 extraction chưa xong toàn bộ)

    Returns (per sample):
        img       : tensor ảnh sau transform
        landmark  : torch.FloatTensor (68, 2), tọa độ trong [0, 1]
        label     : torch.FloatTensor (N_a,)
    """

    def __init__(self, root_path, train=True, fold=1, transform=None,
                 crop_size=224, loader=default_loader,
                 landmark_root=None,
                 landmark_zeros_on_miss=True):

        assert 1 <= fold <= 3, 'fold phải là 1, 2 hoặc 3'

        self._root_path  = root_path
        self._train      = train
        self._transform  = transform
        self.crop_size   = crop_size
        self.loader      = loader
        self.landmark_zeros_on_miss = landmark_zeros_on_miss

        self.img_folder_path = os.path.join(root_path, 'img')
        self.landmark_root   = landmark_root or os.path.join(root_path, 'landmarks')

        split           = 'train' if train else 'test'
        img_list_path   = os.path.join(root_path, 'list', f'DISFA_{split}_img_path_fold{fold}.txt')
        label_list_path = os.path.join(root_path, 'list', f'DISFA_{split}_label_fold{fold}.txt')

        image_list = open(img_list_path).readlines()
        label_list = np.loadtxt(label_list_path)

        self.data_list = make_dataset(image_list, label_list)

    def _load_landmark(self, img_rel_path: str) -> torch.Tensor:
        """
        Load landmark từ file .npy tương ứng.
        img_rel_path ví dụ: 'SN001/1.png'  →  landmark_root/SN001/1.npy
        """
        stem     = os.path.splitext(img_rel_path.strip())[0]
        npy_path = os.path.normpath(os.path.join(self.landmark_root, stem + '.npy'))

        if os.path.isfile(npy_path):
            return torch.from_numpy(np.load(npy_path).astype(np.float32))
        if self.landmark_zeros_on_miss:
            return torch.zeros(68, 2, dtype=torch.float32)
        raise FileNotFoundError(f'Landmark not found: {npy_path}')

    def __getitem__(self, index):
        img_path, label = self.data_list[index]

        img      = self.loader(os.path.normpath(os.path.join(self.img_folder_path, img_path)))
        landmark = self._load_landmark(img_path)

        if self._train:
            w, h     = img.size
            offset_y = random.randint(0, h - self.crop_size)
            offset_x = random.randint(0, w - self.crop_size)
            flip     = random.randint(0, 1)
            if self._transform is not None:
                img = self._transform(img, flip, offset_x, offset_y)
            if flip:
                landmark[:, 0] = 1.0 - landmark[:, 0]   # flip landmark x
        else:
            if self._transform is not None:
                img = self._transform(img)

        return img, landmark, torch.from_numpy(label.astype(np.float32))

    def __len__(self):
        return len(self.data_list)
