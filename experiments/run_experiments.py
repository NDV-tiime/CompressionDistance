import pandas as pd
from dataset_loaders import load_accounting_edits_dataset, load_iwslt_dataset

from experiments import (
    compute_metrics_for_df,
    run_correlation_experiment,
    run_knn_experiment,
)


def main():
    # Available metrics (comment to exclude)
    metrics = [
        "bleu",
        "ter",
        "bertscore",
        "meteor",
        "rouge",
        "character",
        "levenshtein",
        "compression",
    ]

    # Load datasets
    print("\n=== LOADING DATASETS ===")

    print("Loading accounting edits dataset...")
    custom_df = load_accounting_edits_dataset()
    print(f"✓ Accounting edits dataset loaded: {len(custom_df)} entries")

    # Compute metrics for both versions
    print("\n=== COMPUTING METRICS ===")

    print("\nComputing metrics for accounting edits dataset (WITH KNOWLEDGE)...")
    custom_metrics_with_knowledge = compute_metrics_for_df(
        custom_df.copy(),
        metrics,
        prediction_col="edit_text",
        reference_col="llm_answer_with_knowledge",
    )

    print("\nComputing metrics for accounting edits dataset (WITHOUT KNOWLEDGE)...")
    custom_metrics_without_knowledge = compute_metrics_for_df(
        custom_df.copy(),
        metrics,
        prediction_col="edit_text",
        reference_col="llm_answer_without_knowledge",
    )

    # Load and compute metrics for IWSLT
    print("\nLoading IWSLT dataset...")
    iwslt_df = load_iwslt_dataset("./iwslt2019")
    print(f"✓ IWSLT dataset loaded: {len(iwslt_df)} annotators")

    print("\nComputing metrics for IWSLT dataset (all annotators)...")
    # Temporarily concatenate all annotator DataFrames
    combined_df = pd.concat(iwslt_df.values(), keys=iwslt_df.keys())
    combined_df = compute_metrics_for_df(combined_df, metrics)

    for annotator in iwslt_df:
        iwslt_df[annotator] = combined_df.loc[annotator]

    # Run experiments
    print("\n=== RUNNING EXPERIMENTS ===")

    print("\n===== ACCOUNTING EDITS DATASET (WITH KNOWLEDGE) =====")
    run_correlation_experiment(
        custom_metrics_with_knowledge, metrics, target_column="edit_time"
    )
    run_knn_experiment(
        custom_metrics_with_knowledge, metrics, target_column="edit_time"
    )

    print("\n===== ACCOUNTING EDITS DATASET (WITHOUT KNOWLEDGE) =====")
    run_correlation_experiment(
        custom_metrics_without_knowledge, metrics, target_column="edit_time"
    )
    run_knn_experiment(
        custom_metrics_without_knowledge, metrics, target_column="edit_time"
    )

    print("\n===== IWSLT2019 DATASET =====")

    # Experiments with edit time
    print("\n--- With edit time ---")
    run_correlation_experiment(
        iwslt_df, metrics, target_column="time", group_by_annotator=True
    )
    run_knn_experiment(
        iwslt_df, metrics, target_column="time", consolidate_annotators=True
    )

    # Experiments with keystrokes
    print("\n--- With keystrokes ---")
    run_correlation_experiment(
        iwslt_df, metrics, target_column="keystrokes", group_by_annotator=True
    )
    run_knn_experiment(
        iwslt_df, metrics, target_column="keystrokes", consolidate_annotators=True
    )


if __name__ == "__main__":
    main()
