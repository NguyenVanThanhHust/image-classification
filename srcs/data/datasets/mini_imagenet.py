import torch
from torch.utils.data import Dataset
from PIL import Image
import os
from typing import Any, Callable, cast, Optional, Union, Tuple, List, Dict

# Define common image extensions for ImageFolder
IMG_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif', '.tiff', '.webp')

def pil_loader(path: str) -> Image.Image:
    """Default image loader for ImageFolder, uses PIL to open images."""
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB') # Ensure consistent 3 channels

def is_image_file(filename: str) -> bool:
    """Checks if a file has an allowed image extension."""
    return any(filename.lower().endswith(extension) for extension in IMG_EXTENSIONS)

class CustomImageFolder(Dataset):
    """
    A conceptual implementation of torchvision.datasets.ImageFolder.

    This class assumes the following directory structure:
    root/class_x/xxx.png
    root/class_x/xxy.png
    root/class_y/123.png
    root/class_y/nsdfa.png
    ...
    """
    def __init__(
        self,
        root: str,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        loader: Callable[[str], Any] = pil_loader,
        is_valid_file: Optional[Callable[[str], bool]] = None,
        allow_empty: bool = False
    ):
        self.root = root
        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader
        
        if is_valid_file is None:
            self.is_valid_file = is_image_file
        else:
            self.is_valid_file = is_valid_file

        self.classes, self.class_to_idx = self._find_classes(self.root, allow_empty)
        self.samples = self._make_dataset(self.root, self.class_to_idx, self.is_valid_file, allow_empty)

        if not self.samples:
            raise RuntimeError(f"Found 0 files in subfolders of: {self.root}. "
                               f"Supported extensions are: {', '.join(IMG_EXTENSIONS)}")

    def _find_classes(self, directory: str, allow_empty: bool) -> Tuple[List[str], Dict[str, int]]:
        """
        Finds the class folders in a dataset.

        Args:
            directory (str): Root directory path.
            allow_empty (bool): If True, empty folders are considered valid classes.

        Returns:
            (List[str], Dict[str, int]): List of all classes and dictionary mapping each class to an index.
        """
        class_file = os.path.join(directory, "classes.txt")
        with open(class_file, "r") as handle:
            lines = handle.readlines()
            classes = [l.rstrip() for l in lines]

        # classes = sorted([d.name for d in os.scandir(directory) if d.is_dir()])
        
        # if not classes and not allow_empty:
        #      raise FileNotFoundError(f"Couldn't find any class folder in {directory}.")w
        class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
        return classes, class_to_idx

    def _make_dataset(
        self,
        directory: str,
        class_to_idx: Dict[str, int],
        is_valid_file: Callable[[str], bool],
        allow_empty: bool
    ) -> List[Tuple[str, int]]:
        """
        Generates a list of (image_path, class_index) tuples.

        Args:
            directory (str): Root directory path.
            class_to_idx (dict): Dictionary mapping class names to class indices.
            is_valid_file (callable): A function to check if a file is valid.
            allow_empty (bool): If True, allows classes with no valid files.

        Returns:
            List[Tuple[str, int]]: List of (image_path, class_index) tuples.
        """
        instances = []
        available_classes = set()
        
        for target_class in sorted(class_to_idx.keys()):
            class_index = class_to_idx[target_class]
            target_dir = os.path.join(directory, target_class)
            if not os.path.isdir(target_dir):
                continue # Skip if the directory somehow doesn't exist (e.g., deleted after scanning)

            for root, _, fnames in sorted(os.walk(target_dir, followlinks=True)):
                for fname in sorted(fnames):
                    path = os.path.join(root, fname)
                    if is_valid_file(path):
                        item = path, class_index
                        instances.append(item)
                        available_classes.add(target_class)

        if not allow_empty:
            empty_classes = set(class_to_idx.keys()) - available_classes
            if empty_classes:
                raise RuntimeError(f"Found no valid file for the classes {', '.join(sorted(empty_classes))}. "
                                   f"Supported extensions are: {', '.join(IMG_EXTENSIONS)}")

        return instances

    def __len__(self) -> int:
        """Returns the total number of samples in the dataset."""
        return len(self.samples)

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        path, target = self.samples[index]
        sample = self.loader(path)

        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return sample, target

if __name__ == '__main__':
    dataset = CustomImageFolder("../Datasets/split_mini_imagenet")
    print(dataset.__len__())