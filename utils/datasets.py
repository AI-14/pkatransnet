import pandas as pd
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from .tokenizers import CustomTokenizer
from PIL import Image
from transformers import PreTrainedTokenizer


class FgCustomDataset(Dataset):
    def __init__(
        self,
        reports_filepath: str,
        images_dir: str,
        tokenizer: CustomTokenizer,
        transforms: transforms.Compose,
        tag_tokenizer: PreTrainedTokenizer,
        tag_seq_len: int,
        impression_tokenizer: PreTrainedTokenizer,
        impression_seq_len: int,
        dataset_name: str,
    ) -> None:
        """Initializes FgCustomDataset class.

        Args:
            reports_filepath (str): Path to reports csv file.
            images_dir (str): Path to images directory.
            tokenizer (CustomTokenizer): Tokenizer.
            transforms (transforms.Compose): Transforms.
            tag_tokenizer (PreTrainedTokenizer): Tag tokenizer.
            tag_seq_len (int): Tag sequence length.
            impression_tokenizer (PreTrainedTokenizer): Impression tokenizer.
            impression_seq_len (int): Impression sequence length.
            dataset_name (str): Dataset name.
        """

        self.df = pd.read_csv(reports_filepath, encoding="utf-8")
        self.images_dir = images_dir
        self.transforms = transforms
        self.tokenizer = tokenizer
        self.tag_tokenizer = tag_tokenizer
        self.tag_seq_len = tag_seq_len
        self.impression_tokenizer = impression_tokenizer
        self.impression_seq_len = impression_seq_len
        self.dataset_name = dataset_name

    def __len__(self) -> int:
        """Calculates total length of dataframe.

        Returns:
            int: Length.
        """

        return len(self.df)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor | int]:
        """Prepares a single item for dataloaders.

        Args:
            idx (int): Index.

        Returns:
            dict[str, torch.Tensor | int]: Dictionary with all the related content in it.
        """

        if self.dataset_name in ["covctr", "pgross"]:
            image_paths = self.df["image_id"].iloc[idx]
            img1 = Image.open(f"{self.images_dir}/{image_paths}").convert("RGB")
            img2 = Image.open(f"{self.images_dir}/{image_paths}").convert("RGB")
        else:
            image_paths = self.df["image_paths"].iloc[idx].split(",")
            image1_path, image2_path = image_paths[0], image_paths[1]
            img1 = Image.open(f"{self.images_dir}/{image1_path}").convert("RGB")
            img2 = Image.open(f"{self.images_dir}/{image2_path}").convert("RGB")

        img1 = self.transforms(img1)
        img2 = self.transforms(img2)

        findings = str(self.df["findings"].iloc[idx])
        input_ids, pad_mask, label = self.tokenizer(findings)

        impression = str(self.df["impression"].iloc[idx])
        impression_tokenized = self.impression_tokenizer(
            impression,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=self.impression_seq_len,
        )
        impression_input_ids = impression_tokenized["input_ids"].squeeze(0)
        impression_attention_mask = impression_tokenized["attention_mask"].squeeze(0)

        tags = str(self.df["tags"].iloc[idx])
        tags_tokenized = self.tag_tokenizer(
            tags,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=self.tag_seq_len,
        )
        tags_input_ids = tags_tokenized["input_ids"].squeeze(0)
        tags_attention_mask = tags_tokenized["attention_mask"].squeeze(0)

        identifier = self.df["identifier"].iloc[idx]

        return {
            "image1": img1,
            "image2": img2,
            "input_ids": input_ids,
            "pad_mask": pad_mask,
            "label": label,
            "impression_input_ids": impression_input_ids,
            "impression_attention_mask": impression_attention_mask,
            "tags_input_ids": tags_input_ids,
            "tags_attention_mask": tags_attention_mask,
            "identifier": identifier,
        }
