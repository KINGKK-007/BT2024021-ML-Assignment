# Polynomial Regression Assignment

**Name:** Kanav Kumar  
**Roll No:** BT2024021

This assignment builds polynomial regression models for two personalized datasets. The workflow inspects the data, uses an 80/20 training-validation split, compares all permitted polynomial degrees, and applies feature scaling with Ridge regression for stability. The selected models are retrained on the complete training data to generate the final test predictions.

## Results

| Dataset | Selected degree | Ridge alpha | Validation MSE | Validation R² |
|---|---:|---:|---:|---:|
| var1 | 5 | 10 | 0.422622 | 0.954957 |
| var2 | 12 | 1 | 0.175754 | 0.996560 |

The complete write-up is available in [BT2024021_Report.pdf](BT2024021_Report.pdf).

## Run the code

Place the four personalized CSV files and `sample_submission.csv` in a `Dataset/` folder, then run:

```bash
python3 -m pip install -r requirements.txt
python3 polynomial_regression.py
```

The script creates the validation results, plots, and both prediction CSV files. The personalized datasets are intentionally not included in this repository.
