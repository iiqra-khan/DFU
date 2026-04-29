import torch
from torch.utils.data import Dataset
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2
import glob
from pathlib import Path
import numpy as np


IMAGE_EXTENSIONS = ('*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG')
DPM_GRADE_ALIASES = {
    'grade1': ['grade1', 'grade_1', 'grade01'],
    'grade2': ['grade2', 'grade_2', 'grade02'],
    'grade3': ['grade3', 'grade_3', 'grade03'],
    'grade4': ['grade4', 'grade_4', 'grade04'],
}


def _collect_files_with_extensions(directory, patterns):
    files = []
    for pattern in patterns:
        files.extend(glob.glob(f"{directory}/{pattern}"))
    return sorted(files)


def _stem(path):
    return Path(path).stem.lower()


def _normalized_name(name):
    return name.lower().replace(' ', '').replace('-', '').replace('__', '_')


def _list_images(directory):
    files = []
    for pattern in IMAGE_EXTENSIONS:
        files.extend(Path(directory).glob(pattern))
    return sorted(files)


def _split_dir_map(root, split):
    split_root = _resolve_split_root(root, split)
    if not split_root.exists():
        return {}
    return {
        _normalized_name(p.name): p
        for p in split_root.iterdir()
        if p.is_dir()
    }


def _pick_grade_dir(mapping, grade):
    for alias in DPM_GRADE_ALIASES[grade]:
        found = mapping.get(_normalized_name(alias))
        if found is not None:
            return found
    return None


def _resolve_split_root(root, split):
    root = Path(root)
    split_aliases = {
        'train': ['train'],
        'valid': ['valid', 'val'],
        'val': ['val', 'valid'],
        'test': ['test'],
    }
    for candidate in split_aliases.get(split, [split]):
        split_root = root / candidate
        if split_root.exists():
            return split_root
    return root / split

class FUSegDataset(Dataset):
    """Binary segmentation dataset for FUSeg"""
    def __init__(self, image_dir, mask_dir, transform=None):
        self.images = sorted(glob.glob(f"{image_dir}/*.jpg"))
        self.masks = sorted(glob.glob(f"{mask_dir}/*.png"))
        self.transform = transform

        # Kaggle datasets often contain mixed image extensions; keep existing defaults,
        # then expand to common image types if nothing was found.
        if len(self.images) == 0:
            self.images = _collect_files_with_extensions(
                image_dir,
                ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']
            )

        if len(self.masks) == 0:
            self.masks = _collect_files_with_extensions(
                mask_dir,
                ['*.png', '*.jpg', '*.jpeg', '*.PNG', '*.JPG', '*.JPEG']
            )

        # Pair by filename stem to avoid index errors when one folder has extras.
        image_map = {_stem(p): p for p in self.images}
        mask_map = {_stem(p): p for p in self.masks}
        common_keys = sorted(set(image_map.keys()) & set(mask_map.keys()))

        self.images = [image_map[k] for k in common_keys]
        self.masks = [mask_map[k] for k in common_keys]

        # Raise an explicit error so users can fix path mounts quickly instead of getting
        # a DataLoader "num_samples=0" error later.
        if len(self.images) == 0 or len(self.masks) == 0:
            raise ValueError(
                "FUSegDataset is empty. "
                f"image_dir={image_dir} (found {len(self.images)} files), "
                f"mask_dir={mask_dir} (found {len(self.masks)} files). "
                "Verify Config.FUSEG_PATH and folder names in Kaggle."
            )

    def __getitem__(self, idx):
        image = np.array(Image.open(self.images[idx]).convert('RGB'))
        mask = np.array(Image.open(self.masks[idx]).convert('L'))
        mask = (mask > 0).astype(np.float32)

        if self.transform:
            aug = self.transform(image=image, mask=mask)
            image, mask = aug['image'], aug['mask']

        return image, mask.float()

    def __len__(self):
        return len(self.images)

class DPMDataset(Dataset):
    """Wagner grade classification dataset"""
    def __init__(self, data_dir, split='train', transform=None):
        self.data_dir = _resolve_split_root(data_dir, split)
        self.transform = transform
        self.samples = []
        grade_map = _split_dir_map(data_dir, split)

        for grade in ['grade1', 'grade2', 'grade3', 'grade4']:
            grade_dir = _pick_grade_dir(grade_map, grade)
            if grade_dir is None:
                continue
            images = _list_images(grade_dir)
            label = int(grade[-1]) - 1  # 0-indexed
            self.samples.extend([(img, label) for img in images])

        if len(self.samples) == 0:
            raise ValueError(
                "DPMDataset is empty. "
                f"data_dir={data_dir}, split={split}. "
                "Expected folders like train/grade1..grade4 or aliases grade_1/grade01."
            )

    def __getitem__(self, idx):
        image_path, label = self.samples[idx]
        image = np.array(Image.open(image_path).convert('RGB'))

        if self.transform:
            image = self.transform(image=image)['image']

        return image, label

    def __len__(self):
        return len(self.samples)

# Augmentations
def get_transforms(stage='train'):
    if stage == 'train':
        return A.Compose([
            A.Resize(512, 512),  # SegFormer expects 512x512
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.Rotate(limit=30, p=0.5),
            A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.0, hue=0.0, p=0.5),
            A.GaussianBlur(blur_limit=(3, 7), p=0.3),
            A.Normalize(),
            ToTensorV2()
        ])
    else:
        return A.Compose([
            A.Resize(512, 512),  # SegFormer expects 512x512
            A.Normalize(),
            ToTensorV2()
        ])
