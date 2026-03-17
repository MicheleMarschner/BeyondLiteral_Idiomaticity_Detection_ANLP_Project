# Dataset Preprocessing & Splitting

This repository contains the preprocessing pipeline and controlled data splitting strategy used to create zero-shot and one-shot training, development, and test datasets for multilingual contexts.

The dataset includes three languages:

EN – English

PT – Portuguese

GL – Galician

## Original Data Sources

The preprocessing pipeline uses the following raw files:

train_zero_shot.csv

train_one_shot_1.csv

train_one_shot_2.csv

dev.csv

dev_gold.csv


## Data Cleaning & Normalization
NaN values and duplication have been removed (No duplicated sample were in the datat).

All datasets were concatenated into a unified dataframe (whole_sets_df) before splitting.

Cleaning steps:

- Unicode normalization (NFKC)

- Removal of control and formatting characters

- Standardization of quotation marks

- Whitespace normalization

- Replacement of missing values (NaN in the Next column -> " ")



## Overlap Analysis

Before splitting, we verified:

- No sample-level overlap between:
     - zero-shot train and dev
     - one-shot train and dev
     - zero-shot train and one-shot train

- MWE-level overlap analysis revealed:

     - Dev MWEs fully overlap with one-shot (non-Galician) training MWEs
     - No overlap between zero-shot training and dev MWEs

## Dataset Statistics (Full Merged Data)

Total unique MWEs: 486

Unique MWEs per language:
| Language | Unique MWEs |
| -------- | ----------- |
| EN       | 273         |
| PT       | 163         |
| GL       | 50          |

Language distribution is imbalanced:

EN: 3953 samples

PT: 1563 samples

GL: 63 samples


# Zero_shot setting Splitting Strategy:

## Create train/dev/test splits:

- Split based on Language and MWE:

     - 50 unique MWEs -> Dev

     - 50 unique MWEs -> Test

     - Remaining 386 MWEs -> Train



- Ensure no MWE overlap across splits

- Language distribution matches shared task proportions

- Label distribution closely mirrors original data


## Proportional MWE Allocation

Allocation was computed proportionally:

𝑟𝑜𝑢𝑛𝑑(50 × unique_language_MWEs / 486)

| Language | MWEs per split |
| -------- | -------------- |
| EN       | 28             |
| PT       | 17             |
| GL       | 5              |

Splitted datasets:
- zero_shot_train.csv
- zero_shot_dev.csv
- zero_shot_test.csv

# One_shot Splitting Strategy:
The one-shot setting requires:

For each MWE in dev/test, at least one labeled example (0 and/or 1) must exist in the training set.


## Controlled Swapping Procedure

Starting from zero-shot splits:

For each MWE in dev/test:

1. Move into training set:

- One sample with label 0

- One sample with label 1 (if exists)

2. To preserve distribution:

- Replace moved samples with train samples

- Replacement must:

     - Match language

     - Match label

     - Come from an MWE that appears exactly the same number of moved samples in train

However, for 2 PT MWEs and 40 EN MWEs, there was not sufficient samples to move from train set to dev and test sets all in all. But still the distribution of samples between the zero-shot sets and one-shot sets are so close to each other as shown below:

| Data Sets              | MWEs | English | Portuguese | Galician | All  |
|------------------------|------|---------|------------|----------|------|
| Zero Shot train_set    | 386  | 3160    | 1215       | 51       | 4426 |
| Zero Shot dev_set      | 50   | 369     | 141        | 6        | 516  |
| Zero Shot test_set     | 50   | 424     | 207        | 6        | 637  |
| One Shot train_set     | 434  | 3200    | 1217       | 51       | 4468 |
| One Shot dev_set       | 82   | 363     | 141        | 6        | 510  |
| One Shot test_set      | 70   | 390     | 205        | 6        | 601  |



Splitted datasets:
- one_shot_train.csv
- one_shot_dev.csv
- one_shot_test.csv

# Alternative Stratified Experiment (Label-Aware)
We also experimented with stratifying MWEs by label distribution:
Compute "label ratio" per MWE:

- only_0

- mixed

- only_1

Perform stratified splitting by label ratio

Result:

Distribution nearly identical to primary zero-shot splits

Final experiments continued with primary zero-shot splits.

# Data Integrity Checks

After splitting:

- Verified no MWE overlap between splits

- Verified label distribution consistency

- Verified language distribution stability

- Confirmed no duplicate samples

- Confirmed no missing values

# Design Principles

- Reproducibility (fixed random_state=42)

- No leakage between splits

- Language-aware balancing

- Label-aware validation

- MWE-level splitting (not sentence-level)

- Controlled one-shot exposure


