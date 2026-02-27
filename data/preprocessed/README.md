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

- Replacement of missing values (NaN in Next column → " ")


After cleaning:

No remaining NaN values

No duplicate rows

No suspicious Unicode characters



## Overlap Analysis

Before splitting, we verified:

- No sample-level overlap between:

- zero-shot train and dev

- one-shot train and dev

- zero-shot train and one-shot train

MWE-level overlap analysis revealed:

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

##Create train/dev/test splits:

- 50 unique MWEs → Dev

- 50 unique MWEs → Test

- Remaining 386 MWEs → Train

- Group by Language and MWE

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

     * Match language

     * Match label

     * Come from an MWE that appears exactly the same number of times in train

This ensures:

- Balanced exposure in training

- Stable label distribution

- Split sizes remain unchanged


No MWE overlap across One-Shot split sets are checked.


Splitted datasets:
- one_shot_train.csv
- one_shot_dev.csv
- one_shot_test.csv

# Alternative Stratified Experiment (Label-Aware)
We also experimented with stratifying MWEs by label distribution:
Compute label ratio per MWE

Categorize into:

only_0

mostly_0

balanced

mostly_1

only_1

Perform stratified splitting by label bucket

Result:

Distribution nearly identical to proportional split

Final experiments continued with proportional zero-shot split

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


