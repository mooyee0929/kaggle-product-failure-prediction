# Kaggle Product Failure Prediction

This is a tabular binary-classification pipeline for the Kaggle [Tabular Playground Series - Aug 2022](https://www.kaggle.com/competitions/tabular-playground-series-aug-2022) competition. Given a product's code, loading, four attributes and 18 measurements, it predicts the probability that the product fails a test. Submissions are scored by ROC AUC. The pipeline fills missing values, engineers aggregate features, label-encodes categorical columns, selects features by mutual information, and fits a strongly regularized logistic regression. Training and inference live in two Jupyter notebooks.

## Method

1. **Missing values.** The numeric columns are filled with their mean using scikit-learn's `SimpleImputer`, fit on train and test together.
2. **Feature engineering.**
   - `attribute_2*3`: the product of `attribute_2` and `attribute_3`. The attributes are fixed for each product code, and train (codes A-E) and test (codes F-I) share no product codes.
   - `meas_gr1_avg` / `meas_gr1_std`: row-wise mean and standard deviation of `measurement_3`, `5`, `6` and `8`.
   - `meas_gr2_avg` / `meas_gr2_std`: row-wise mean and standard deviation of `measurement_7`, `9` and `13`.
   - Missing-value indicator features for `measurement_3` and `measurement_5` were tried and are left commented out. The write-up reports that they lowered the leaderboard score.
3. **Encoding.** `product_code`, `attribute_0` and `attribute_1` are converted to integers with `LabelEncoder`, fit on train and test together.
4. **Feature selection.** `mutual_info_classif` (with `random_state=1`) scores every feature, and those with a score above `1e-3` are kept. The 12 selected features are `measurement_17`, `attribute_0`, `measurement_1`, `product_code`, `attribute_2`, `attribute_1`, `measurement_12`, `measurement_2`, `meas_gr2_avg`, `measurement_7`, `measurement_3` and `loading`.
5. **Model.** `LogisticRegression(C=0.0001, penalty='l2', solver='newton-cg')`. The write-up compares solver/penalty combinations (`lbfgs`, `liblinear`, `newton-cg`, `newton-cholesky`, `sag`, `saga`), and `newton-cg` with L2 performed best.

## Results

| Metric | ROC AUC |
| --- | --- |
| Training set (in-sample, printed by the training notebook) | 0.5919 |
| Kaggle private leaderboard (late submission, from the write-up) | 0.59041 |
| Kaggle public leaderboard (late submission, from the write-up) | 0.58573 |

## Repository structure

```
.
├── train.ipynb             # EDA, preprocessing, feature selection, training; saves the model
├── inference.ipynb         # preprocessing of test.csv, prediction, submission file
├── model.joblib            # trained LogisticRegression (joblib)
├── submission.csv          # generated submission (id, failure probability)
├── report.pdf              # write-up of the approach and leaderboard result
├── train.csv               # 26,570 labeled rows (5,649 failures)
├── test.csv                # 20,775 unlabeled rows
└── sample_submission.csv   # Kaggle submission template
```

## Setup

The notebooks use `numpy`, `pandas`, `scikit-learn`, `matplotlib` and `joblib`. They call `DataFrame.append` and `Series.append`, which were removed in pandas 2.0, so use pandas 1.x:

```bash
pip install "pandas<2" numpy scikit-learn matplotlib joblib jupyter
```

## Usage

All paths are relative, so run both notebooks from the repository root.

1. Open `train.ipynb` and run all cells. It reads `train.csv` and `test.csv`, fits the model, prints its training AUC, saves it as `model.joblib`, and prints the selected feature list in the "Print selected columns to inference" cell.
2. If the selected features changed, paste the printed list into the "Select Features from Training" cell of `inference.ipynb`. The committed notebook already contains the 12 features listed above.
3. Run all cells of `inference.ipynb`. It loads `model.joblib`, predicts failure probabilities for `test.csv`, and writes `submission.csv` in the `sample_submission.csv` format.

The trained model is committed as `model.joblib` (the saved notebook output still shows its earlier file name). A copy is also on [Google Drive](https://drive.google.com/file/d/1VBpWg3UUS8D1l15xNiaLaQuIYrV8vxmA/view?usp=share_link).

Note: the inference notebook refits the imputer and label encoders on `test.csv` alone instead of reusing the train+test fit from training.

## Acknowledgements

- The data comes from the Kaggle [Tabular Playground Series - Aug 2022](https://www.kaggle.com/competitions/tabular-playground-series-aug-2022) competition.
- The write-up cites these references for the preprocessing, feature engineering and modelling ideas:
  - [TPS 2022 Aug: EDA + baseline logistic regression](https://www.kaggle.com/code/scgupta/tps-2022-aug-eda-baseline-logistic-regression) (scgupta)
  - [TPS Aug 2022](https://www.kaggle.com/code/sfktrkl/tps-aug-2022/notebook#Modelling) (sfktrkl)
  - [TPS Aug 22: logistic regression + CV + FE](https://www.kaggle.com/code/vishnu123/tps-aug-22-top-2-logistic-regression-cv-fe) (vishnu123)
  - Competition discussions [342126](https://www.kaggle.com/competitions/tabular-playground-series-aug-2022/discussion/342126) and [342319](https://www.kaggle.com/competitions/tabular-playground-series-aug-2022/discussion/342319)
  - [scikit-learn imputation guide](https://scikit-learn.org/stable/modules/impute.html) and a [CSDN article](https://blog.csdn.net/weixin_46072771/article/details/106188753)
