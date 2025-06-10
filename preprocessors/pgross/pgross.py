import re
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import json

tqdm.pandas()


class PreprocessPeirgrossDataset:
    def __init__(
        self,
        images_dir: str,
        captions_filepath: str,
        tags_filepath: str,
        defn_filepath: str,
        train_filepath: str,
        val_filepath: str,
        test_filepath: str,
    ) -> None:
        """Initializes PreprocessPeirgrossDataset class.

        Args:
            images_dir (str): Path to images directory.
            captions_filepath (str): Path to captions json file.
            tags_filepath (str): Path to tags json file.
            defn_filepath (str): Path to def json file.
            train_filepath (str): Path to save train csv file.
            val_filepath (str): Path to save val csv file.
            test_filepath (str): Path to save test csv file.
        """

        self.images_dir = images_dir
        self.captions_filepath = captions_filepath
        self.tags_filepath = tags_filepath
        self.defn_filepath = defn_filepath
        self.train_filepath = train_filepath
        self.val_filepath = val_filepath
        self.test_filepath = test_filepath

    def load_and_merge_dfs(self) -> pd.DataFrame:
        """Loads and merge captions and tags json files into dataframe.

        Returns:
            pd.DataFrame: Merged dataframe.
        """

        # Captions
        with open(self.captions_filepath) as f:
            data = json.load(f)
        caption_df = pd.DataFrame.from_dict(data.items())
        caption_df.columns = ["image_id", "findings"]

        # Tags
        with open(self.tags_filepath) as f:
            data = json.load(f)
        tags_df = pd.DataFrame.from_dict(data.items())
        tags_df.columns = ["image_id", "tags"]
        tags_df["tags"] = tags_df["tags"].apply(
            lambda text: f"Tags: {', '.join(text)}."
        )

        merged_df = pd.merge(caption_df, tags_df, on="image_id")

        # Extra
        with open(self.defn_filepath) as f:
            defns = json.load(f)

        def populate(text: str, defns: dict[str, str]) -> str:
            bodyp = text.split(":")[1].lower().strip()
            for bp, d in defns.items():
                if bp == bodyp:
                    return d

        merged_df["impression"] = merged_df["findings"].apply(
            lambda text: populate(text, defns)
        )

        return merged_df

    def clean_reports(self, report: str) -> str:
        """Cleans the report.

        Args:
            report (str): Report.

        Returns:
            str: Cleaned report.
        """

        def report_cleaner(t):
            return (
                t.replace("..", ".")
                .replace("..", ".")
                .replace("..", ".")
                .replace("1. ", "")
                .replace(". 2. ", ". ")
                .replace(". 3. ", ". ")
                .replace(". 4. ", ". ")
                .replace(". 5. ", ". ")
                .replace(" 2. ", ". ")
                .replace(" 3. ", ". ")
                .replace(" 4. ", ". ")
                .replace(" 5. ", ". ")
                .replace(". .", ".")
                .strip()
                .lower()
                .split(". ")
            )

        def sent_cleaner(t):
            return re.sub(
                "[.,?;*!%^&_+():-\\[\\]{}]",
                "",
                t.replace('"', "")
                .replace("/", "")
                .replace("\\", "")
                .replace("'", "")
                .strip()
                .lower(),
            )

        tokens = [
            sent_cleaner(sent)
            for sent in report_cleaner(report)
            if sent_cleaner(sent) != []
        ]
        report = " . ".join(tokens) + " ."
        return report

    def preprocess(self) -> None:
        """Preprocesses the dataset."""

        df = self.load_and_merge_dfs()
        df.dropna(inplace=True)
        df["findings"] = df["findings"].progress_apply(
            lambda text: self.clean_reports(text.split(":")[-1].strip())
        )
        df["identifier"] = np.arange(0, len(df), dtype=np.int64)

        # Data splits
        train, val_test = train_test_split(
            df, test_size=0.3, shuffle=True, random_state=1234
        )
        val, test = train_test_split(
            val_test, test_size=0.66, shuffle=True, random_state=1234
        )

        train.to_csv(self.train_filepath, index=False)
        val.to_csv(self.val_filepath, index=False)
        test.to_csv(self.test_filepath, index=False)

        train = pd.read_csv(self.train_filepath, encoding="utf-8").dropna()
        val = pd.read_csv(self.val_filepath, encoding="utf-8").dropna()
        test = pd.read_csv(self.test_filepath, encoding="utf-8").dropna()

        train.to_csv(self.train_filepath, index=False)
        val.to_csv(self.val_filepath, index=False)
        test.to_csv(self.test_filepath, index=False)

        print(
            f"Splits info: train_size={len(train)} | val_size={len(val)} | test_size={len(test)}"
        )
        print(
            f"Saved {self.train_filepath}, {self.val_filepath}, {self.test_filepath} files"
        )


def main() -> None:
    """Executes the main flow."""

    PreprocessPeirgrossDataset(
        "datasets/pgross/images",
        "datasets/pgross/captions.json",
        "datasets/pgross/tags.json",
        "datasets/pgross/def.json",
        "datasets/pgross/train.csv",
        "datasets/pgross/val.csv",
        "datasets/pgross/test.csv",
    ).preprocess()


if __name__ == "__main__":
    main()
