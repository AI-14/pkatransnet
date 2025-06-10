import os
import re
from xml.etree import ElementTree
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import stanza
from ordered_set import OrderedSet
from collections import Counter

tqdm.pandas()


class PreprocessIuxrayDataset:
    def __init__(
        self,
        images_dir: str,
        reports_dir: str,
        train_filepath: str,
        val_filepath: str,
        test_filepath: str,
    ) -> None:
        """Initializes PreprocessIuxrayDataset class.

        Args:
            images_dir (str): Path to images directory.
            reports_dir (str): Path to reports directory.
            train_filepath (str): Path to save train csv file.
            val_filepath (str): Path to save val csv file.
            test_filepath (str): Path to save test csv file.
        """

        self.images_dir = images_dir
        self.reports_dir = reports_dir
        self.train_filepath = train_filepath
        self.val_filepath = val_filepath
        self.test_filepath = test_filepath

        stanza.download("en", package="mimic", processors={"ner": "radiology"})
        stanza.download("en", package="mimic", processors={"ner": "i2b2"})
        self.radnlp = stanza.Pipeline(
            "en", package="mimic", processors={"ner": "radiology"}
        )
        self.i2b2nlp = stanza.Pipeline(
            "en", package="mimic", processors={"ner": "i2b2"}
        )

    def extract_data_from_xml_files(self) -> pd.DataFrame:
        """Extracts core information from xml files and prepares a dataframe.

        Returns:
            pd.DataFrame: Dataframe with all the key contents extracted.
        """

        df = pd.DataFrame(
            columns=[
                "xml_filename",
                "image_paths",
                "impression",
                "findings",
            ]
        )

        for idx, file_name in enumerate(tqdm(os.listdir(self.reports_dir))):
            tree = ElementTree.parse(self.reports_dir + "/" + file_name)

            impression = tree.find(".//AbstractText[@Label='IMPRESSION']").text
            findings = tree.find(".//AbstractText[@Label='FINDINGS']").text

            if not findings:
                continue

            if not impression:
                impression = "infer from tags"

            # Keeping only two images per report
            image_names: list[str] = []
            for iterator in tree.findall("parentImage"):
                image_names.append(iterator.attrib["id"] + ".png")

            if len(image_names) == 0:
                continue

            if len(image_names) == 1:
                image_names.append(image_names[0])

            if len(image_names) > 2:
                image_names = image_names[:2]

            df.loc[idx] = [
                file_name,
                ",".join(image_names),
                impression,
                findings,
            ]

        return df

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

    def get_tags(self, text: str) -> str:
        """Extracts tags (NER) from text.

        Args:
            text (str): Text.

        Returns:
            str: Extracted tags from the text.
        """

        i2b2doc = self.i2b2nlp(text)
        raddoc = self.radnlp(text)

        problem_ents: list[str] = []
        for ent in i2b2doc.entities:
            if ent.type == "PROBLEM":
                problem_ents.append(ent.text)
        problem_ents = (
            "problems: "
            + (", ".join(list(OrderedSet(problem_ents))) if problem_ents else "none")
            + ". "
        )

        anatomy_ents: list[str] = []
        for ent in raddoc.entities:
            if ent.type == "ANATOMY":
                anatomy_ents.append(ent.text)
        anatomy_ents = (
            "anatomies: " + (", ".join(list(OrderedSet(anatomy_ents))) if anatomy_ents else "none") + ". "
        )

        observation_ents: list[str] = []
        for ent in raddoc.entities:
            if ent.type == "OBSERVATION":
                observation_ents.append(ent.text)
        observation_ents = (
            "observations: "
            + (", ".join(list(OrderedSet(observation_ents))) if observation_ents else "none")
            + ". "
        )

        uncertainty_ents: list[str] = []
        for ent in raddoc.entities:
            if ent.type == "UNCERTAINTY":
                uncertainty_ents.append(ent.text)
        uncertainty_ents = (
            "uncertainties: "
            + (", ".join(list(OrderedSet(uncertainty_ents))) if uncertainty_ents else "none")
            + "."
        )

        all_ents = problem_ents + anatomy_ents + observation_ents + uncertainty_ents

        return all_ents.strip() if all_ents else "empty tags."

    def extract_meta_information(self, text: str, idx: int) -> str:
        """Extracts meta information from tags column.

        Args:
            text (str): Text.
            idx (int): Index to specify which among the entities [problem, anatomy, observation, uncertainty] to choose from.

        Returns:
            str: Meta information.
        """

        meta_info = []
        text = text.split(".")
        text = text[idx].split(":")[1]
        text = text.split(".")[0]
        text = text.split(",")
        for word in text:
            meta_info.append(word.strip())

        if len(meta_info) > 1:
            return " , ".join(meta_info) + " ."
        else:
            return "".join(meta_info) + " ."

    def get_most_common(self, col_name: str, df: pd.DataFrame, topk: int) -> list[tuple[str, int]]:
        """Extracts most common entity in a column.

        Args:
            col_name (str): Column name.
            df (pd.DataFrame): Dataframe.
            topk (int): Most common topk elements.

        Returns:
            list[tuple[str, int]]: Frequencies of the most common entities.
        """

        all_words: list[str] = []
        for x in df[col_name].values.tolist():
            x = x[:-2].split(" , ")
            all_words.extend(x)
        freq = Counter(all_words)
        return freq.most_common(topk)

    def populate(self, text: str, most_com: list[tuple[str, int]]) -> str:
        """Populates the most common entities in given column text.

        Args:
            text (str): Text.
            most_com (list[tuple[str, int]]): Frequencies of the most common entities.

        Returns:
            str: Populated text.
        """
        
        words = []
        for phrase in text[:-2].split(" , "):
            for word, _ in most_com:
                if word.strip() == phrase.strip():
                    words.append(word)
        return ", ".join(words) + " ."
    
    def preprocess(self) -> None:
        """Preprocesses the dataset."""

        # Preprocess dataset
        print(f"Total images in iuxray dataset: {len(os.listdir(self.images_dir))}")
        print(f"Total reports in iuxray dataset: {len(os.listdir(self.reports_dir))}")
        df = self.extract_data_from_xml_files()
        df.dropna(inplace=True)
        df["impression"] = df["impression"].progress_apply(self.clean_reports)
        df["findings"] = df["findings"].progress_apply(self.clean_reports)
        df["tags"] = df["findings"].progress_apply(self.get_tags)

        # Preprocess entities
        df["problem"] = df["tags"].apply(lambda x: self.extract_meta_information(x, 0))
        df["anatomy"] = df["tags"].apply(lambda x: self.extract_meta_information(x, 1))
        df["observation"] = df["tags"].apply(
            lambda x: self.extract_meta_information(x, 2)
        )
        df["uncertainty"] = df["tags"].apply(
            lambda x: self.extract_meta_information(x, 3)
        )
        df["identifier"] = np.arange(0, len(df), dtype=np.int64)

        # Data splits
        train, val_test = train_test_split(
            df, test_size=0.3, shuffle=True, random_state=1234
        )
        val, test = train_test_split(
            val_test, test_size=0.66, shuffle=True, random_state=1234
        )

        # Populate most common entities
        most_com = self.get_most_common("problem", train, 20)
        train["problem"] = train["problem"].apply(lambda text: self.populate(text, most_com))
        val["problem"] = val["problem"].apply(lambda text: self.populate(text, most_com))
        test["problem"] = test["problem"].apply(lambda text: self.populate(text, most_com))

        most_com = self.get_most_common("anatomy", train, 20)
        train["anatomy"] = train["anatomy"].apply(lambda text: self.populate(text, most_com))
        val["anatomy"] = val["anatomy"].apply(lambda text: self.populate(text, most_com))
        test["anatomy"] = test["anatomy"].apply(lambda text: self.populate(text, most_com))

        most_com = self.get_most_common("observation", train, 20)
        train["observation"] = train["observation"].apply(lambda text: self.populate(text, most_com))
        val["observation"] = val["observation"].apply(lambda text: self.populate(text, most_com))
        test["observation"] = test["observation"].apply(lambda text: self.populate(text, most_com))

        most_com = self.get_most_common("uncertainty", train, 20)
        train["uncertainty"] = train["uncertainty"].apply(lambda text: self.populate(text, most_com))
        val["uncertainty"] = val["uncertainty"].apply(lambda text: self.populate(text, most_com))
        test["uncertainty"] = test["uncertainty"].apply(lambda text: self.populate(text, most_com))

        # Combine the most common entities in a single tags column
        def combine(col1: str, col2: str, col3: str, col4: str) -> str:
            return f"problems: {col1} anatomies: {col2} observations: {col3} uncertainties: {col4}".replace(
                " , ", ", "
            ).replace(" .", ".")

        train["tags"] = train[
            [
                "problem",
                "anatomy",
                "observation",
                "uncertainty",
            ]
        ].apply(lambda x: combine(*x), axis=1)
        val["tags"] = val[
            [
                "problem",
                "anatomy",
                "observation",
                "uncertainty",
            ]
        ].apply(lambda x: combine(*x), axis=1)
        test["tags"] = test[
            [
                "problem",
                "anatomy",
                "observation",
                "uncertainty",
            ]
        ].apply(lambda x: combine(*x), axis=1)

    
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

    PreprocessIuxrayDataset(
        "datasets/iuxray/images",
        "datasets/iuxray/reports",
        "datasets/iuxray/train.csv",
        "datasets/iuxray/val.csv",
        "datasets/iuxray/test.csv",
    ).preprocess()


if __name__ == "__main__":
    main()
