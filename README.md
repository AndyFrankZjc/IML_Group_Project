# Home Credit Neural Network - Using Processed Team Features

This version is designed to use the processed CSV files created by the team.

## Expected files

Place these files into:

```text
data/processed/
```

Required baseline processed files:

```text
application_train_processed.csv
application_test_processed.csv
```

Required merged-feature processed files:

```text
application_train_processed_merge_features.csv
application_test_processed_merge_features.csv
```

Note: if your test files have names such as `application_test_processed(1).csv`, rename them to:

```text
application_test_processed.csv
application_test_processed_merge_features.csv
```

## Run application-only processed features

```bash
python src/train_nn_processed.py --feature-set application
```

## Run processed features with merged historical features

```bash
python src/train_nn_processed.py --feature-set merged
```

## Run quick tuning

```bash
python src/train_nn_processed.py --feature-set merged --tune
```

## Outputs

```text
outputs/metrics_processed_<feature_set>.json
outputs/tuning_results_processed_<feature_set>.csv
outputs/confusion_matrix_processed_<feature_set>.png
outputs/training_auc_processed_<feature_set>.png
outputs/training_loss_processed_<feature_set>.png
outputs/nn_predictions_internal_test_processed_<feature_set>.csv
outputs/kaggle_submission_nn_processed_<feature_set>.csv
outputs/best_mlp_model_processed_<feature_set>.keras
```

## What this version assumes

The team has already completed the main data processing steps, including:

1. feature screening and duplicate removal
2. cleaning ambiguous categorical values
3. missing indicators for features with missing rate >= 15%
4. aggregation of EXT_SOURCE_1, EXT_SOURCE_2, EXT_SOURCE_3
5. six ratio features
6. winsorization for outlier handling
7. missing value imputation
8. categorical encoding through one-hot encoding and frequency encoding
9. merging additional dataset features

Therefore, this neural network script does **not** repeat those feature engineering steps.
It only performs neural-network-specific steps:

1. train / validation / internal test split
2. feature alignment between train and official test
3. median imputation as safety check
4. standardisation
5. class weighting
6. MLP training and hyperparameter tuning
7. evaluation and output generation
