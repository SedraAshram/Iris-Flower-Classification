# Iris Flower Classification

A reproducible machine learning project that explores the Iris dataset and
compares classifiers for predicting Setosa, Versicolor, and Virginica.

## Dataset and objective

The project uses the official Iris data file from the [UCI Machine Learning
Repository](https://archive.ics.uci.edu/dataset/53/iris), stored under
`data/uci_iris/`. It contains 150 samples, four numeric measurements (sepal
length and width, petal length and width), and three species labels. Source,
license, and citation details are in `data/uci_iris/README.md`.

## Project workflow

1. Load the dataset into a pandas dataframe with consistent feature names.
2. Validate required columns, numeric feature values, plausible measurement
   ranges, missing values, and target labels. Retain repeated observations and
   report their count rather than assuming they are accidental duplicates.
3. Explore class counts, descriptive statistics, feature correlations, and a
   species-colored feature pairplot.
4. Create a reproducible, stratified 80/20 train/test split.
5. Train Logistic Regression, K-Nearest Neighbors, Decision Tree, and Support
   Vector Machine classifiers on the training partition.
6. Compare candidates using five-fold stratified cross-validation on training
   data, then report accuracy, macro/weighted precision, recall and F1, plus a
   labeled confusion matrix on the held-out test partition.
7. Select the leading model by mean cross-validation macro F1. The test set is
   kept out of model selection.

Imputation and scaling are kept inside model pipelines to avoid data leakage.
Decision Tree does not require feature scaling.

## Run

From Windows PowerShell, create and activate the virtual environment, install
the dependencies, and run the project:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python main.py
```

The workflow writes:

- `results/data_summary.json`: dataset checks and the selected model.
- `results/model_metrics.json`: per-model test and cross-validation metrics.
- `results/model_comparison.csv`: sortable scalar metrics for comparing models.
- `results/eda/`: feature distributions, class comparisons, correlations, and
  the feature pairplot.
- `results/confusion_matrices/`: a labeled confusion matrix for each model.

To regenerate only the EDA plots and console summaries, run
`python notebooks/01_exploratory_data_analysis.py` from the project root.

Run the unit tests with:

```bash
python -m unittest discover -s tests
```

## Structure

- `src/data_loader.py`: dataset loading and shared schema.
- `src/preprocessing.py`: cleaning audit and stratified splitting.
- `src/eda.py`: dataset summary and visualization.
- `src/models.py`: reproducible classifier definitions.
- `src/evaluation.py`: cross-validation and held-out metrics.
- `main.py`: end-to-end workflow.
- `tests/`: unit tests for loading, cleaning, and splitting.

## Current run

With the fixed stratified split and random seed, SVM had the highest mean
five-fold training macro F1 (0.967) and 0.967 test accuracy. Logistic
Regression, KNN, and Decision Tree each reached 0.933 test accuracy. These
scores summarize this small dataset and one held-out split; the cross-validation
spread is reported in `results/model_comparison.csv` to show variation across
training folds.
