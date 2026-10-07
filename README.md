# BT2024021 Polynomial Regression Assignment

This folder contains Kanav Kumar's polynomial regression solution for var1 and var2.

The personalized datasets are not stored in this repository. Before running the
code, create a `Dataset` folder and place these files inside it:

```text
BT2024021_train_var1.csv
BT2024021_test_var1.csv
BT2024021_train_var2.csv
BT2024021_test_var2.csv
sample_submission.csv
```

## Run

From this folder, install the standard dependencies if needed and run:

```bash
python3 -m pip install -r requirements.txt
python3 polynomial_regression.py
```

The script inspects the datasets, performs training-only model selection, saves the validation tables and plots, retrains the chosen models on all training data, creates both prediction CSV files, and checks the deliverables.

## Selected models

| Dataset | Degree | Ridge alpha | Validation MSE | Validation R² |
|---|---:|---:|---:|---:|
| var1 | 5 | 10 | 0.422622 | 0.954957 |
| var2 | 12 | 1 | 0.175754 | 0.996560 |
