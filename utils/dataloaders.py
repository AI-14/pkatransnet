from torch.utils.data import DataLoader
from torchvision.models import Swin_V2_B_Weights
from .datasets import FgCustomDataset
from .tokenizers import CustomTokenizer
from transformers import PreTrainedTokenizer


class FgDataloaders:
    @staticmethod
    def get_train_loader(
        dataset_name: str,
        images_dir: str,
        train_filepath: str,
        batch_size: int,
        tokenizer: CustomTokenizer,
        tag_tokenizer: PreTrainedTokenizer,
        tag_seq_len: int,
        impression_tokenizer: PreTrainedTokenizer,
        impression_seq_len: int,
    ) -> DataLoader:
        """Prepares the train dataloader.

        Args:
            dataset_name (str): Dataset name.
            images_dir (str): Path to images directory.
            train_filepath (str): Path to train csv file.
            batch_size (int): Batch size.
            tokenizer (CustomTokenizer): Tokenizer.
            tag_tokenizer (PreTrainedTokenizer): Tag tokenizer.
            tag_seq_len (int): Tag sequence length.
            impression_tokenizer (PreTrainedTokenizer): Impression tokenizer.
            impression_seq_len (int): Impression sequence length.

        Returns:
            DataLoader: Train dataloader.
        """

        train_dataset = FgCustomDataset(
            train_filepath,
            images_dir,
            tokenizer,
            Swin_V2_B_Weights.IMAGENET1K_V1.transforms(),
            tag_tokenizer,
            tag_seq_len,
            impression_tokenizer,
            impression_seq_len,
            dataset_name,
        )

        return DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
        )

    @staticmethod
    def get_val_loader(
        dataset_name: str,
        images_dir: str,
        val_filepath: str,
        batch_size: int,
        tokenizer: CustomTokenizer,
        tag_tokenizer: PreTrainedTokenizer,
        tag_seq_len: int,
        impression_tokenizer: PreTrainedTokenizer,
        impression_seq_len: int,
    ) -> DataLoader:
        """Prepares the val dataloader.

        Args:
            dataset_name (str): Dataset name.
            images_dir (str): Path to images directory.
            val_filepath (str): Path to val csv file.
            batch_size (int): Batch size.
            tokenizer (CustomTokenizer): Tokenizer.
            tag_tokenizer (PreTrainedTokenizer): Tag tokenizer.
            tag_seq_len (int): Tag sequence length.
            impression_tokenizer (PreTrainedTokenizer): Impression tokenizer.
            impression_seq_len (int): Impression sequence length.

        Returns:
            DataLoader: Val dataloader.
        """

        val_dataset = FgCustomDataset(
            val_filepath,
            images_dir,
            tokenizer,
            Swin_V2_B_Weights.IMAGENET1K_V1.transforms(),
            tag_tokenizer,
            tag_seq_len,
            impression_tokenizer,
            impression_seq_len,
            dataset_name,
        )

        return DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=True,
        )

    @staticmethod
    def get_test_loader(
        dataset_name: str,
        images_dir: str,
        test_filepath: str,
        tokenizer: CustomTokenizer,
        tag_tokenizer: PreTrainedTokenizer,
        tag_seq_len: int,
        impression_tokenizer: PreTrainedTokenizer,
        impression_seq_len: int,
    ) -> DataLoader:
        """Prepares the test dataloader.

        Args:
            dataset_name (str): Dataset name.
            images_dir (str): Path to images directory.
            test_filepath (str): Path to test csv file.
            tokenizer (CustomTokenizer): Tokenizer.
            tag_tokenizer (PreTrainedTokenizer): Tag tokenizer.
            tag_seq_len (int): Tag sequence length.
            impression_tokenizer (PreTrainedTokenizer): Impression tokenizer.
            impression_seq_len (int): Impression sequence length.

        Returns:
            DataLoader: Test dataloader.
        """

        test_dataset = FgCustomDataset(
            test_filepath,
            images_dir,
            tokenizer,
            Swin_V2_B_Weights.IMAGENET1K_V1.transforms(),
            tag_tokenizer,
            tag_seq_len,
            impression_tokenizer,
            impression_seq_len,
            dataset_name,
        )

        return DataLoader(
            test_dataset,
            batch_size=1,
        )
