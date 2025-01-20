from typing import Dict, List, Union

import evaluate
import nltk
import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from tqdm import tqdm

from compression_distance import compression_distance

# Download required NLTK data
nltk.download("punkt", quiet=True)

# Load HuggingFace metrics once
HF_METRICS = {
    "bleu": evaluate.load("bleu"),
    "ter": evaluate.load("ter"),
    "bertscore": evaluate.load("bertscore"),
    "meteor": evaluate.load("meteor"),
    "rouge": evaluate.load("rouge"),
    "character": evaluate.load("character"),
}


def compute_metric(metric_name: str, prediction: str, reference: str) -> float:
    """
    Compute a single metric for one prediction-reference pair.

    Args:
        metric_name: Name of the metric to compute
        prediction: predicted text
        reference: reference text

    Returns:
        float: Computed metric value
    """
    if metric_name == "compression":
        return compression_distance(reference, prediction)

    elif metric_name == "levenshtein":
        return nltk.edit_distance(reference, prediction)

    elif metric_name in HF_METRICS:
        if metric_name == "bleu":
            result = HF_METRICS["bleu"].compute(
                predictions=[prediction], references=[[reference]]
            )
            return result["bleu"]

        elif metric_name == "ter":
            result = HF_METRICS["ter"].compute(
                predictions=[prediction], references=[reference]
            )
            return result["score"]

        elif metric_name == "bertscore":
            result = HF_METRICS["bertscore"].compute(
                predictions=[prediction], references=[reference], lang="en"
            )
            return result["f1"][0]

        elif metric_name == "meteor":
            result = HF_METRICS["meteor"].compute(
                predictions=[prediction], references=[reference]
            )
            return result["meteor"]

        elif metric_name == "rouge":
            result = HF_METRICS["rouge"].compute(
                predictions=[prediction], references=[reference]
            )
            return result["rougeL"]

        elif metric_name == "character":
            result = HF_METRICS["character"].compute(
                predictions=[prediction], references=[reference]
            )
            return result["cer_score"]

    return np.nan


def compute_metrics_for_df(
    df: pd.DataFrame,
    metrics_list: List[str],
    prediction_col: str = "edit_text",
    reference_col: str = "llm_answer",
) -> pd.DataFrame:
    """
    Add columns to df for each metric in metrics_list.

    Args:
        df: Input dataframe
        metrics_list: List of metric names to compute
        prediction_col: Column name with predicted text
        reference_col: Column name with reference text

    Returns:
        pd.DataFrame: DataFrame with additional metric columns
    """
    for metric_name in metrics_list:
        metric_column = f"{metric_name}"
        scores = []

        # Compute metric for each text pair
        for pred, ref in tqdm(
            zip(df[prediction_col], df[reference_col]),
            total=len(df),
            desc=f"Computing {metric_name}",
        ):
            score = compute_metric(metric_name, pred, ref)
            scores.append(score)

        df[metric_column] = scores

    return df


def run_correlation_experiment(
    df: Union[pd.DataFrame, Dict[str, pd.DataFrame]],
    metrics_list: List[str],
    target_column: str = "edit_time",
    group_by_annotator: bool = False,
) -> None:
    """
    For each metric in metrics_list, compute Pearson correlation with target_column.

    Args:
        df: Input DataFrame or dict of DataFrames (if grouped by annotator)
        metrics_list: List of metrics to analyze
        target_column: Column to correlate with metrics
        group_by_annotator: Whether to process data per annotator
    """
    if group_by_annotator:
        for annotator, annotator_df in df.items():
            print(f"\n--- {annotator} ---\n")
            _compute_correlations(annotator_df, metrics_list, target_column)
    else:
        print(f"\n--- Pearson Correlation (with '{target_column}') ---\n")
        _compute_correlations(df, metrics_list, target_column)


def _compute_correlations(
    df: pd.DataFrame, metrics_list: List[str], target_column: str
) -> None:
    """Helper function to compute correlations for a single DataFrame."""
    for metric in metrics_list:
        if metric not in df.columns:
            continue

        temp_df = df.dropna(subset=[metric, target_column])
        if temp_df.empty:
            print(f"No valid data for {metric}. Skipping.")
            continue

        corr, pval = pearsonr(temp_df[target_column], temp_df[metric])
        print(f"{metric} correlation: {corr:.4f}, p-value: {pval:.4e}")


def run_knn_experiment(
    df: Union[pd.DataFrame, Dict[str, pd.DataFrame]],
    metrics_list: List[str],
    target_column: str = "edit_time",
    consolidate_annotators: bool = False,
) -> None:
    """
    For each metric in metrics_list, run a simple 1D KNN regression:
      - train/test split
      - compute R^2 on test set

    Args:
        df: Input DataFrame or dict of DataFrames
        metrics_list: List of metrics to analyze
        target_column: Target column for regression
        consolidate_annotators: Whether to combine data from all annotators
    """
    if consolidate_annotators:
        df = pd.concat(df.values(), ignore_index=True)

    df = df.dropna(subset=[target_column] + metrics_list)
    print(f"\n--- KNN R^2 Scores (target='{target_column}') ---\n")

    for metric in metrics_list:
        X = df[[metric]].values
        y = df[target_column].values

        # Split data and train KNN
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        knn = KNeighborsRegressor(n_neighbors=5)
        knn.fit(X_train, y_train)

        # Evaluate
        y_pred = knn.predict(X_test)
        r2_val = r2_score(y_test, y_pred)
        print(f"{metric} -> R^2: {r2_val:.4f}")
