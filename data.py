import os
from typing import Callable, Optional, Tuple

from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T


class SatelliteSegmentationDataset(Dataset):
    """
    Dataset for loading satellite images and corresponding binary masks.
    Images and masks are matched by filename.
    """

    def __init__(
        self,
        image_dir: str,
        mask_dir: str,
        image_size: int = 256,
        augment: bool = False,
        transform: Optional[Callable] = None,
    ) -> None:
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.image_size = image_size
        self.augment = augment
        self.transform = transform

        self.image_filenames = sorted(
            [
                f
                for f in os.listdir(image_dir)
                if f.lower().endswith((".png", ".jpg", ".jpeg", ".tif", ".tiff"))
            ]
        )

        if not self.image_filenames:
            raise RuntimeError(f"No image files found in {image_dir}")

        self._base_transform = T.Compose(
            [
                T.Resize((image_size, image_size)),
                T.ToTensor(),
            ]
        )

        # Masks are single-channel (0/1), converted to float tensor
        self._mask_transform = T.Compose(
            [
                T.Resize((image_size, image_size), interpolation=T.InterpolationMode.NEAREST),
                T.ToTensor(),
            ]
        )

        if augment:
            self._augment = T.RandomChoice(
                [
                    T.RandomHorizontalFlip(p=1.0),
                    T.RandomVerticalFlip(p=1.0),
                    T.RandomRotation(degrees=15),
                ]
            )
        else:
            self._augment = None

    def __len__(self) -> int:
        return len(self.image_filenames)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        img_name = self.image_filenames[idx]
        img_path = os.path.join(self.image_dir, img_name)

        # Guess mask filename: same name or with _mask
        candidate_masks = [
            img_name,
            img_name.replace(".png", "_mask.png").replace(".jpg", "_mask.png"),
        ]
        mask_path = None
        for cand in candidate_masks:
            cand_path = os.path.join(self.mask_dir, cand)
            if os.path.exists(cand_path):
                mask_path = cand_path
                break

        if mask_path is None:
            # Fallback: same name, any extension
            base = os.path.splitext(img_name)[0]
            for f in os.listdir(self.mask_dir):
                if os.path.splitext(f)[0] == base:
                    mask_path = os.path.join(self.mask_dir, f)
                    break

        if mask_path is None:
            raise RuntimeError(f"Could not find mask for image {img_name}")

        image = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")

        if self._augment is not None:
            # Apply same geometric transform to both image and mask
            seed = torch.seed()
            torch.manual_seed(seed)
            image = self._augment(image)
            torch.manual_seed(seed)
            mask = self._augment(mask)

        image = self._base_transform(image)
        mask = self._mask_transform(mask)

        # Ensure mask is binary {0,1}
        mask = (mask > 0.5).float()

        if self.transform is not None:
            image = self.transform(image)

        return image, mask


def create_dataloaders(
    train_image_dir: str,
    train_mask_dir: str,
    val_image_dir: str,
    val_mask_dir: str,
    image_size: int = 256,
    batch_size: int = 4,
    num_workers: int = 4,
) -> Tuple[DataLoader, DataLoader]:
    train_dataset = SatelliteSegmentationDataset(
        train_image_dir, train_mask_dir, image_size=image_size, augment=True
    )
    val_dataset = SatelliteSegmentationDataset(
        val_image_dir, val_mask_dir, image_size=image_size, augment=False
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    return train_loader, val_loader


