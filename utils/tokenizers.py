import json
import os
from collections import Counter

import pandas as pd
import torch


class CustomTokenizer:
    def __init__(
        self,
        train_filepath: str,
        min_freq: int,
        token2id_filepath: str,
        id2token_filepath: str,
        seq_len: int,
        column_name: str,
    ) -> None:
        """Initializes CustomTokenizer class.

        Args:
            train_filepath (str): Path to train csv file.
            min_freq (int): Minimum frequency to keep vocab words.
            token2id_filepath (str): Path to token2id json file.
            id2token_filepath (str): Path to id2token json file.
            seq_len (int): Maximum sequence length.
            column_name (str): Column name.
        """

        self.df = pd.read_csv(train_filepath, encoding="utf-8")
        self.column_name = column_name
        self.min_freq = min_freq
        self.seq_len = seq_len

        if os.path.exists(token2id_filepath) and os.path.exists(id2token_filepath):
            with open(token2id_filepath, "r") as f:
                self.token2id = {token: int(idx) for token, idx in json.load(f).items()}

            with open(id2token_filepath, "r") as f:
                self.id2token = {int(idx): token for idx, token in json.load(f).items()}
        else:
            self.token2id, self.id2token = self.create_vocabulary()

            with open(token2id_filepath, "w") as f:
                json.dump(self.token2id, f)

            with open(id2token_filepath, "w") as f:
                json.dump(self.id2token, f)

    def create_vocabulary(self) -> tuple[dict[str, int], dict[int, str]]:
        """Creates a vocabulary on the column.

        Returns:
            tuple[dict[str, int], dict[int, str]]: Dictionaries of token2id and id2token.
        """

        total_tokens: list[str] = []
        text = self.df[self.column_name].values

        for f in text:
            total_tokens.extend(f.split())

        token_freq = Counter(total_tokens)
        vocabulary = [
            token for token, freq in token_freq.items() if freq >= self.min_freq
        ]
        vocabulary.sort()

        token2id: dict[str, int] = {"<sos>": 1, "<eos>": 2, "<pad>": 0, "<unk>": 3}
        id2token: dict[int, str] = {1: "<sos>", 2: "<eos>", 0: "<pad>", 3: "<unk>"}

        for idx, token in enumerate(vocabulary, start=4):
            token2id[token] = idx
            id2token[idx] = token

        return token2id, id2token

    def get_vocab_size(self) -> int:
        """TReturns total vocab size.

        Returns:
            int: Total vocab size.
        """

        return len(self.token2id)

    def get_token_by_id(self, id: int) -> str:
        """Returns the token given an id.

        Args:
            id (int): Id.

        Returns:
            str: Tokken.
        """

        return self.id2token[id]

    def get_id_by_token(self, token: str) -> int:
        """Returns the id given a token.

        Args:
            token (str): Token.

        Returns:
            int: Id.
        """

        if token not in self.token2id:
            return self.token2id["<unk>"]
        return self.token2id[token]

    def decode_by_ids(self, ids: list[int]) -> str:
        """Decodes the ids into tokens.

        Args:
            ids (list[int]): All the ids.

        Returns:
            str: Text.
        """

        text: str = ""
        for i in ids:
            if self.get_token_by_id(i) not in ["<sos>", "<eos>", "<pad>"]:
                text += self.get_token_by_id(i) + " "
        return text.strip()

    def __call__(self, report: str) -> list[torch.Tensor]:
        """Converts the report into input_ids, generates the pad_mask, and generates the label (output shifted right).

        Args:
            report (str): Report.

        Returns:
            list[torch.Tensor]: Input ids, pad masks, and labels.
        """

        ids: list[int] = []
        tokens = report.split()[: self.seq_len - 2]
        for token in tokens:
            ids.append(self.get_id_by_token(token))

        input_ids = torch.cat(
            [
                torch.tensor([self.get_id_by_token("<sos>")], dtype=torch.int64),
                torch.tensor(ids, dtype=torch.int64),
                torch.tensor(
                    [self.get_id_by_token("<pad>")] * (self.seq_len - len(ids) - 1),
                    dtype=torch.int64,
                ),
            ]
        )  # [seq_len] i.e. [<sos>, ...., <pad>, ..., <pad>]

        pad_mask = (
            (input_ids != self.get_id_by_token("<pad>")).unsqueeze(0).type_as(input_ids)
        )  # [1, seq_len]

        label = torch.cat(
            [
                torch.tensor(ids, dtype=torch.int64),
                torch.tensor([self.get_id_by_token("<eos>")], dtype=torch.int64),
                torch.tensor(
                    [self.get_id_by_token("<pad>")] * (self.seq_len - len(ids) - 1),
                    dtype=torch.int64,
                ),
            ]
        )  # [seq_len] i.e. [..., <eos>, <pad>, ..., <pad>]

        return [input_ids, pad_mask, label]
