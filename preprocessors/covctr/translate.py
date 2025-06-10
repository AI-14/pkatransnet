import pandas as pd
import translators as ts
from tqdm import tqdm
import time


def main() -> None:
    """Executes the main flow."""
    df = pd.read_csv("datasets/covctr/reports.csv")
    
    impression = []
    for _, t in tqdm(
        enumerate(df["impression"].values.tolist()), total=len(df), leave=True
    ):
        impression.append(ts.translate_text(t, translator="bing"))
        time.sleep(0.2)

    df["impression"] = impression
    df["findings"] = df["reports_En"]
    df.drop(["reports_En", "COVID", "terminologies"], axis=1, inplace=True)

    df.to_csv("datasets/covctr/t-reports.csv", index=False)
    print("Saved datasets/covctr/t-reports.csv file")


if __name__ == "__main__":
    main()
