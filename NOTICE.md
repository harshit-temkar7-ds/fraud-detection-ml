# Data attribution and notices

**Dataset.** The real-data results in this repository were produced on the *Credit Card Fraud Detection* dataset
(European card transactions, September 2013; features V1–V28 are anonymised PCA components) published on Kaggle by the
Machine Learning Group of ULB (Université Libre de Bruxelles) with Worldline, under the Database Contents License (DbCL) v1.0:
<https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud>.

Suggested citation (as requested by the dataset authors):
Andrea Dal Pozzolo, Olivier Caelen, Reid A. Johnson and Gianluca Bontempi.
*Calibrating Probability with Undersampling for Unbalanced Classification.* IEEE Symposium on Computational Intelligence and Data Mining (CIDM), 2015.

**What is redistributed.** The full dataset is **not** included (`data/*.csv` is git-ignored). `model/replay_sample.csv` contains
3,000 rows of that dataset (all 75 held-out test frauds plus a sample of legitimate rows) so the dashboard and the Kafka
replay can run. Check the dataset's licence terms before redistributing this file, and remove it (the dashboard replay pages
will then be empty) if you cannot.

**Not a production system.** Reason codes explain the model on anonymised features, not real-world causes. No regulatory
compliance (GDPR, EU AI Act) is claimed. See `docs/MODEL_CARD.md`.
