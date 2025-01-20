import os
import re
from typing import Dict

import pandas as pd
from datasets import load_dataset
from tqdm import tqdm


def load_accounting_edits_dataset() -> pd.DataFrame:
    """
    Load the qa accounting edits dataset from HuggingFace and return a DataFrame with two LLM answer columns:
    - llm_answer_with_knowledge: LLM answer + gold answer
    - llm_answer_without_knowledge: LLM answer only

    Returns:
        pd.DataFrame: DataFrame containing the dataset with the following columns:
            - edit_text: human edited text
            - llm_answer_with_knowledge: LLM answer concatenated with gold answer
            - llm_answer_without_knowledge: original LLM answer
            - edit_time: time taken for the edit
    """
    human_edits = load_dataset("Tiime/fr-qa-accounting-edits", name="human_edits")
    df = human_edits["human_edits"].to_pandas()
    df["llm_answer_with_knowledge"] = df["llm_answer"] + " " + df["gold_answer"]
    df["llm_answer_without_knowledge"] = df["llm_answer"]
    df.rename(columns={"llm_answer_edit": "edit_text"}, inplace=True)

    return df


def clean_text(text: str) -> str:
    return re.sub(r"[^\x20-\x7E]", "", text)


def load_iwslt_dataset(data_directory: str) -> Dict[str, pd.DataFrame]:
    """
    Load IWSLT2019 data from local files and return a dictionary of DataFrames.

    Args:
        data_directory (str): Path to the directory containing IWSLT2019 files

    Returns:
        Dict[str, pd.DataFrame]: Dictionary with annotator names as keys and their
                               corresponding DataFrames as values
    """
    header_path = os.path.join(data_directory, "header")
    with open(header_path, "r", encoding="utf-8") as f:
        header_line = f.readline().strip()
    columns = header_line.split("\t")

    # Process each annotator's file
    data_files = ["ann0", "ann1", "ann2", "ann3", "ann4"]
    annotator_data = {}

    for f_name in data_files:
        full_path = os.path.join(data_directory, f_name)
        rows = []

        # Read and process file line by line
        with open(full_path, "r", encoding="utf-8") as f_in:
            lines = f_in.readlines()
            for line in tqdm(lines, desc=f"Loading {f_name}", leave=False):
                parts = line.strip().split("\t")
                if len(parts) != len(columns):
                    continue

                row_dict = dict(zip(columns, parts))
                row_dict["<MT>"] = clean_text(row_dict.get("<MT>", ""))
                row_dict["<PE>"] = clean_text(row_dict.get("<PE>", ""))

                for field in ["<time>", "<keystrokes>"]:
                    if field in row_dict:
                        try:
                            row_dict[field] = float(row_dict[field])
                        except (ValueError, TypeError):
                            row_dict[field] = None

                rows.append(row_dict)

        # Create DataFrame and process columns
        df = pd.DataFrame(rows)
        df["edit_text"] = df["<PE>"]
        df["llm_answer"] = df["<MT>"]
        df.rename(
            columns={"<time>": "time", "<keystrokes>": "keystrokes"}, inplace=True
        )

        # Clean up DataFrame
        df.dropna(subset=["edit_text", "llm_answer", "time"], inplace=True)
        df.reset_index(drop=True, inplace=True)

        annotator_data[f_name] = df

    return annotator_data
