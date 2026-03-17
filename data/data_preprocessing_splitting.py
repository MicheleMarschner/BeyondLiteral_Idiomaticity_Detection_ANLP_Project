import pandas as pd
import unicodedata
import re
import string
from sklearn.model_selection import train_test_split
import zipfile

df_zero_shot = pd.read_csv("raw/Data/train_zero_shot.csv")
df_one_shot1 = pd.read_csv("raw/Data/train_one_shot.csv") # no GL, Overlap
df_one_shot2 = pd.read_csv("raw/TestData/train_one_shot.csv") # In test folder: Galician, no overlap
df_dev_labels = pd.read_csv("raw/Data/dev_gold.csv")
df_dev_Context = pd.read_csv("raw/Data/dev.csv")

print("zero: ", len(df_zero_shot))
print("One: ", len(df_one_shot1))
print("test_one", len(df_one_shot2))
print("dev_labels: ", len(df_dev_labels))
print("dev_context", len(df_dev_Context))

# Merge df_dev_Context with df_dev_labels on 'ID'
df_dev = df_dev_Context.merge(
  df_dev_labels[['ID', 'Label']],  # only bring necessary columns
  on='ID',
  how='left'  # keeps all rows from df_dev_Context
)

print(df_dev.head())

print(df_dev.columns)

print("Number of samples per lanuage in zero_shot set:\n", df_zero_shot["Language"].value_counts(), "\n")
print("Number of samples per label in zero_shot set:\n", df_zero_shot["Label"].value_counts(), "\n")
print("Number of samples per lanuage in dev set:\n", df_dev["Language"].value_counts(), "\n")
print("Number of samples per label in dev set:\n", df_dev["Label"].value_counts())

# Merge One_shot train sets
df_one_shot = pd.concat(
  [df_one_shot1[["Language", "MWE", "Setting", "Previous", "Target", "Next", "Label"]],
    df_one_shot2[["Language", "MWE", "Setting", "Previous", "Target", "Next", "Label"]]],
  ignore_index=True
)

# No duplication existed, 349 samples
df_one_shot = df_one_shot.drop_duplicates()

# Check NaN values
print(df_one_shot.isna().sum().sum())   # No NaN
print(df_dev.isna().sum().sum())        # No NaN
print(df_zero_shot.isna().sum().sum())  # 2 samples in train_zero_shot has NaN values in "Next" column
print(df_zero_shot.isna().sum())

# What are the samples include NaN
print(df_zero_shot[df_zero_shot.isna().any(axis=1)])

# Replace NaN them with space
df_zero_shot["Next"] = df_zero_shot["Next"].fillna(" ")
print(df_zero_shot.isna().sum().sum())

# Is there any sample in both train_one_shot sets and dev set
common = pd.merge(df_one_shot[["Language", "MWE", "Previous", "Target", "Next"]],
                  df_dev[["Language", "MWE", "Previous", "Target", "Next"]],
                  how='inner'
                  )

print(common.shape[0])  # number of overlapping rows -> 0

# Is there any sample in both train_zero_shot set and dev set
common = pd.merge(df_zero_shot[["Language", "MWE", "Previous", "Target", "Next"]],
                  df_dev[["Language", "MWE", "Previous", "Target", "Next"]],
                  how='inner'
                  )

print(common.shape[0])  # number of overlapping rows -> 0

# **No** samples from dev set in **one_shot** train set nor **zero_shot** train set.

"""**No** samples from dev set in **one_shot** train set nor **zero_shot** train
set.
"""

# No common sample between train_zero_shot and one_shot sets
common = pd.merge(df_zero_shot[["Language", "MWE", "Previous", "Target", "Next"]],
                  df_one_shot[["Language", "MWE", "Previous", "Target", "Next"]],
                  how='inner'
                  )

print(common.shape[0])  # number of overlapping rows -> 0

"""*   Merge zero-shot & both one-shot & dev sets
*   split Merged data -> zero-shot sets
*   split Merged data -> one-shot sets

All sets have all 3 lang. samples





"""

# MWE overlapping between train_zero_shot and dev sets
dev_mwes = set(df_dev["MWE"])
zero_mwes = set(df_zero_shot["MWE"])

overlap = dev_mwes & zero_mwes

print("Number of overlapping MWEs:", len(overlap))
# No overlapping between train_zero_shot and dev sets

# MWE overlapping between train_one_shot (one without Galician lang.) and dev sets
dev_mwes = set(df_dev["MWE"])
one_mwes1 = set(df_one_shot1["MWE"])

overlap = dev_mwes & one_mwes1

print("Number of overlapping MWEs:", len(overlap))
# 50 MWE overlapping between train_one_shot (one without Galician lang.) and dev sets

# MWE overlapping between test/train_one_shot (one with Galician lang.) and dev sets
dev_mwes = set(df_dev["MWE"])
one_mwes2 = set(df_one_shot2["MWE"])

overlap = dev_mwes & one_mwes2

print("Number of overlapping MWEs:", len(overlap))
# No MWE overlapping between test/train_one_shot (one with Galician lang.) and dev sets

print(len(zero_mwes)) # 236 unique MWEs in train_zero_shot
print(len(one_mwes1)) # 100 unique MWEs in train_one_shot_1 (one without Galician lang.)
print(len(one_mwes2)) # 150 unique MWEs in train_one_shot_2 (one with Galician lang.)

print(len(dev_mwes)) # 50 unique MWEs in dev set
# All 50 unique MWEs in dev set overlapped with train_one_shot (one without Galician lang.)

"""All 50 unique MWEs in dev are in one_shot1 (no Galician set)"""

# Merge all data across all sets of train_Zero_shot, train_one_shots, dev_set
df_dev["Setting"] = "dev"

whole_sets_df = pd.concat(
  [df_zero_shot[["Language", "MWE", "Setting", "Previous", "Target", "Next", "Label"]],
    df_one_shot[["Language", "MWE", "Setting", "Previous", "Target", "Next", "Label"]],
    df_dev[["Language", "MWE", "Setting", "Previous", "Target", "Next", "Label"]]],
  ignore_index=True
)

# Add ID column
whole_sets_df.insert(0, "ID", range(len(whole_sets_df)))

# No Nan in merged set
print("Any Nan values in while data set: ", whole_sets_df.isna().sum().sum())

# No duplication were in merged data
whole_sets_df = whole_sets_df.drop_duplicates()
len(whole_sets_df)

def clean_text(text):
  # standardize Unicode characters to compatibility form (visually similar characters)
  text = unicodedata.normalize("NFKC", text)

  # Remove control/format characters
  text = "".join(
    c for c in text
    if unicodedata.category(c) not in ["Cf", "Cc"]
  )

  text = text.replace("“", '"').replace("”", '"')
  text = text.replace("„", '"').replace("«", '"').replace("»", '"')

  text = re.sub(r"\s+", " ", text)

  return text.strip()

temp = whole_sets_df.copy()
temp["Target_original"] = temp["Target"]
print(temp.head())

# Check the changes after cleaning
text_columns = ["Language", "MWE", "Previous", "Target", "Next"]
temp[text_columns] = temp[text_columns].applymap(clean_text)

changed_rows = temp[
  temp["Target_original"] != temp["Target"]
][["Target_original", "Target"]]

print(changed_rows["Target"].iloc[16])
print(changed_rows["Target_original"].iloc[16])

text_columns = ["Language", "MWE", "Previous", "Target", "Next"]

whole_sets_df[text_columns] = whole_sets_df[text_columns].applymap(clean_text)

# print("First rows of whole data set:\n", whole_sets_df.head())

# Check for any remained suspicious characters
def has_suspicious_unicode(text):
  for c in str(text):
      if unicodedata.category(c) in ["Cf", "Cc"]:
          return True
  return False

def find_suspicious_chars(text):
  chars = []
  for c in str(text):
      if unicodedata.category(c) in ["Cf", "Cc"]:
          chars.append((c, unicodedata.name(c, "UNKNOWN"), hex(ord(c))))
  return chars

print("Is there any suspicious characters left: ")
if whole_sets_df["Target"].apply(has_suspicious_unicode).sum():
  print(len(whole_sets_df.loc[
    whole_sets_df["Target"].apply(has_suspicious_unicode),
    "Target"
  ].apply(find_suspicious_chars)))
else:
  print("No")

# Remove "Setting" column:
whole_sets_df = whole_sets_df.drop(columns=["Setting"])
print(whole_sets_df.columns)

"""Zero_shot splitting"""

# Zero_shot splitting:
whole_mwe_counts = whole_sets_df["MWE"].value_counts()

print(len(whole_mwe_counts))  # 486 unique MWEs across whole sets

# Number of unique MWEs per language across whole sets
unique_mwe_per_lang = (
  whole_sets_df
  .groupby("Language")["MWE"]
  .nunique()
)

print(unique_mwe_per_lang)
# EN:   273
# GL:   50
# PT:   163

# In the shared task, they considered 50 unique MWEs for dev set
dev_mwe_counts = df_dev["MWE"].value_counts()
print(len(dev_mwe_counts))

unique_mwe_per_lang = (
    df_dev
    .groupby("Language")["MWE"]
    .nunique()
)

print(unique_mwe_per_lang)
# In the shared task, per language in dev set, number of unique MWE
# EN:    30
# PT:    20

# Summary of whole_sets_df: # of MWEs across languages and labels
df = whole_sets_df.copy()

mwe_summary = (
  df.groupby("MWE")
    .agg(
      Total_Samples=("MWE", "count"),
      English=("Language", lambda x: (x == "EN").sum()),
      Portuguese=("Language", lambda x: (x == "PT").sum()),
      Galician=("Language", lambda x: (x == "GL").sum()),
      Label_0=("Label", lambda x: (x == 0).sum()),
      Label_1=("Label", lambda x: (x == 1).sum())
    )
    .reset_index()
)

print(mwe_summary)

"""For each language:

EN: round(50 * 273 / 486) ~ 28 MWEs

PT: round(50 * 163 / 486) ~ 17 MWEs

GL: round(50 * 50 / 486) ~ 5 MWEs
"""

# Number of each MWE per language
mwe_lang = (
  df.groupby(["Language", "MWE"])
    .size()
    .reset_index(name="Count")
)

print(mwe_lang.head())

# Splitting whole data in train, dev and test
train_mwes = []
dev_mwes = []
test_mwes = []

# How many MWEs per language considering for dev and test sets
alloc = {
  "EN": 28,
  "PT": 17,
  "GL": 5
}

# For each languages pick the alloc number of MWE
for lang in ["EN", "PT", "GL"]:
  lang_mwes = mwe_lang[mwe_lang["Language"] == lang]["MWE"].unique()

  # First picking Dev MWEs
  dev_lang, remaining = train_test_split(
    lang_mwes,
    test_size=len(lang_mwes) - alloc[lang],
    random_state=42
  )

  # Then picking Test MWEs
  test_lang, train_lang = train_test_split(
    remaining,
    test_size=len(remaining) - alloc[lang],
    random_state=42
  )

  # Collect the selected MWE identifiers for each split
  dev_mwes.extend(dev_lang)
  test_mwes.extend(test_lang)
  train_mwes.extend(train_lang)

# Set all samples to corresponding sets
dev_set = df[df["MWE"].isin(dev_mwes)]
test_set = df[df["MWE"].isin(test_mwes)]
train_set = df[df["MWE"].isin(train_mwes)]

# Check the distribuntion across datasets and compare with the distribution of datsets in shared task
# Almost the same number of samples across datasets in comparison with shared task ones
print("train_set:\n", "len: ", len(train_set), "\n", train_set["Language"].value_counts(), "\n")
print("dev_set:\n", "len: ", len(dev_set), "\n", dev_set["Language"].value_counts(), "\n")
print("test_set:\n", "len: ", len(test_set), "\n", test_set["Language"].value_counts())

print("shared task:\n", "zero_shot:\n", "len: ", len(df_zero_shot), "\n", df_zero_shot["Language"].value_counts(), "\n")
print("dev_set:\n", "len: ", len(df_dev), "\n", df_dev["Language"].value_counts(), "\n")

# check the distribution per language and labels and compare with shared task ones
print("train_set:\n", train_set.groupby(["Language", "Label"]).size(), "\n")
print("dev_set:\n", dev_set.groupby(["Language", "Label"]).size(), "\n")
print("test_set:\n", test_set.groupby(["Language", "Label"]).size())

print("shared task\n Zero_shot:\n len:", len(df_zero_shot), "\n", df_zero_shot.groupby(["Language","Label"]).size(), "\n")
print("dev set:\n len:", len(df_dev), "\n", df_dev.groupby(["Language","Label"]).size())

# The Distribution datasets across languages and labels is the same as the distribution of datasets in shared task

# No overlap between train and dev and test sets
train_mwes = set(train_set["MWE"])
dev_mwes = set(dev_set["MWE"])
test_mwes = set(test_set["MWE"])

overlap_dev = train_mwes & dev_mwes
overlap_test = train_mwes & test_mwes

print("Number of overlapping MWEs:", len(overlap_dev), ", ", len(overlap_test))

print("Number of MWES in zero_shot train_set: ", len(train_mwes))
print("Number of MWES in zero_shot dev_set: ",  len(dev_mwes))
print("Number of MWES in zero_shot test_set: ", len(test_mwes))

# changing the column name of "Label" to "label"
train_set.rename(columns={"Label": "label"}, inplace=True)
dev_set.rename(columns={"Label": "label"}, inplace=True)
test_set.rename(columns={"Label": "label"}, inplace=True)

# Save datasets files
train_set.to_csv("zero_shot_train.csv", index=False)
dev_set.to_csv("zero_shot_dev.csv", index=False)
test_set.to_csv("zero_shot_test.csv", index=False)

with zipfile.ZipFile("zero_shot_splits.zip", "w", zipfile.ZIP_DEFLATED) as zipf:
    zipf.write("zero_shot_train.csv")
    zipf.write("zero_shot_dev.csv")
    zipf.write("zero_shot_test.csv")

"""One_shot splitting"""

# Explore the number of each MWEs across whole data sets
mwe_counts = whole_sets_df["MWE"].value_counts()

print(len(mwe_counts[mwe_counts == 1]))
print(len(mwe_counts[mwe_counts == 2]))
print(len(mwe_counts[mwe_counts > 2]))

print(df_dev["MWE"].value_counts())

# Explore with some MWEs in the dev set shared task how many samples and with which
#labels they have overlap with train_one_shot set shared task.
print("MWE: bad hat")
print("dev:", (df_dev["MWE"] == "bad hat").sum())
print("train:", (df_one_shot1["MWE"] == "bad hat").sum())
print(df_one_shot1.loc[df_one_shot1["MWE"] == "bad hat", "Label"])

print("MWE: life vest")
print("dev:", (df_dev["MWE"] == "life vest").sum())
print(df_dev.loc[df_dev["MWE"] == "life vest", "Label"])
print("train:", (df_one_shot["MWE"] == "life vest").sum())
print(df_one_shot.loc[df_one_shot["MWE"] == "life vest", "Label"])

cols = ["Language", "MWE", "Previous", "Target", "Next", "Label"]  #"Setting",

life_vest_rows = df_one_shot1.loc[
    df_one_shot1["MWE"] == "life vest",
    cols
]

check = life_vest_rows.merge(
    df_dev[cols],
    on=cols,
    how="inner",
    indicator=True
)

check["_merge"].value_counts()

# Check the context fields of shared MWE between dev and train_one_shot is in
#both dev and train_one_shot sets or samples are different?
subset = df_one_shot[df_one_shot["MWE"] == "life vest"]
print("Sample of life vest in train_one_shot set:\n", subset)

print("Sample of life vest in dev set:\n", df_dev[df_dev["MWE"] == "life vest"])
# The whole sample does not exist in both dev and train_one_shot,
#just MWE is overlaped with different context between dev and train_one_shot

# Check there are any MWE samples with only one or two samples for later swapping
# EN
df_per_lang = train_set[train_set["Language"] == "EN"]
mwe_counts = df_per_lang["MWE"].value_counts()
single = mwe_counts[mwe_counts == 1] .sum()
double = mwe_counts[mwe_counts == 2].sum()

print("number of MWEs with one sample in trainset for EN:", single)
print("number of MWEs with two samples in trainset for EN:", double)

# PT
df_per_lang = train_set[train_set["Language"] == "PT"]
mwe_counts = df_per_lang["MWE"].value_counts()
single = mwe_counts[mwe_counts == 1] .sum()
double = mwe_counts[mwe_counts == 2].sum()

print("\nnumber of MWEs with one sample in trainset for PT:", single)
print("number of MWEs with two samples in trainset for PT:", double)

# GL
df_per_lang = train_set[train_set["Language"] == "GL"]
mwe_counts = df_per_lang["MWE"].value_counts()
single = mwe_counts[mwe_counts == 1] .sum()
double = mwe_counts[mwe_counts == 2].sum()

print("\nnumber of MWEs with one sample in trainset for GL:", single)
print("number of MWEs with two samples in trainset for GL:", double)

# For one_shot sets splitting:
# To preserve the distribution between the zero-shot sets and one-shot sets,
# whenever one or two samples of an MWE are moved from dev/test to train,
# move the same number of samples from train to dev/test with the same language and label.
#
# If one sample of an MWE is moved from dev/test to train,
# select an MWE in train set that has exactly one sample in train set.
#
# If two samples of an MWE are moved from dev/test to train,
# select an MWE in train set that has exactly two samples in train set.

def one_shot_process(current_train_set, split_set):

  train = current_train_set.copy()
  split = split_set.copy()

  adding_rows = []

  # Iterate over each MWE in split set
  for mwe in split["MWE"].unique():
    mwe_rows = split[split["MWE"] == mwe]
    lang = mwe_rows["Language"].iloc[0] # Keep the MWe lang.

    labels_present = mwe_rows["label"].unique() # What are the labels for MWE 0 / 1 or both

    # If MWE has label 0 or 1 or both, move one sample for each existed label
    labels_to_move = []  # which labels to move
    if 0 in labels_present:
      labels_to_move.append(0)
    if 1 in labels_present:
      labels_to_move.append(1)

    moved_rows = []

    # Move sample(s) from split set to train
    for label in labels_to_move:
      # Pick the first sample from split set with the desired MWE and label
      candidate = mwe_rows[mwe_rows["label"] == label].iloc[[0]]

      # Move to train
      adding_rows.append(candidate)
      # train = pd.concat([train, candidate])
      split = split.drop(candidate.index)

      moved_rows.append((lang, label)) # Keep moved lang and label for later replacement

    num_moved = len(moved_rows)

    # Replacement step: from train to split
    for lang, label in moved_rows:
      # Find candidate MWE in train with exact count = num_moved
      train_counts = current_train_set.groupby(["Language", "MWE"]).size()

      eligible_mwes = train_counts[
          (train_counts == num_moved)
      ].index

      replacement_candidates = current_train_set[
          current_train_set.set_index(["Language", "MWE"]).index.isin(eligible_mwes)
          & (current_train_set["Language"] == lang)
          & (current_train_set["label"] == label)
      ]

      if len(replacement_candidates) == 0:
        continue  # skip if no replacement found

      replacement = replacement_candidates.iloc[[0]] # Pick the first Cadidate MWE

      # Move replacement to split
      split = pd.concat([split, replacement])
      current_train_set = current_train_set.drop(replacement.index)

  return current_train_set, split, adding_rows

# One_shot Splitting using created zero_shot sets but satisfy One_shot constraint
modified_train_set, one_shot_dev_set, dev_adding_rows = one_shot_process(train_set, dev_set)
modified_train_set, one_shot_test_set, test_adding_rows = one_shot_process(modified_train_set, test_set)

for row in (dev_adding_rows + test_adding_rows):
  modified_train_set = pd.concat([modified_train_set, row])

one_shot_train_set = modified_train_set

#Check the number of samples across data sets preserved.
print("length of zero_shot train set: ", len(train_set))
print("length of one_shot train set: ", len(one_shot_train_set), "\n")

print("\nlength of zero_shot dev set: ", len(dev_set))
print("length of one_shot dev set: ", len(one_shot_dev_set), "\n")

print("\nlength of zero_shot test set: ", len(test_set))
print("length of one_shot test set: ", len(one_shot_test_set))

# Check there is an overlap between one_shot train_set and one_shot dev_set and one_shot test_set
print("one_shot train and dev sets are disjoint: ", set(one_shot_train_set["MWE"]).isdisjoint(one_shot_dev_set["MWE"]))
print("one_shot train and test sets are disjoint: ", set(one_shot_train_set["MWE"]).isdisjoint(test_set["MWE"]))

# No overlap between train and dev and test sets
train_mwes = set(one_shot_train_set["MWE"])
dev_mwes = set(one_shot_dev_set["MWE"])
test_mwes = set(one_shot_test_set["MWE"])

overlap_dev = train_mwes & dev_mwes
overlap_test = train_mwes & test_mwes

print("Number of overlapping MWEs:", len(overlap_dev), ", ", len(overlap_test))

print("Number of MWES in one_shot train_set: ", len(train_mwes))
print("Number of MWES in one_shot dev_set: ",  len(dev_mwes))
print("Number of MWES in one_shot test_set: ", len(test_mwes))

# Check the distribuntion across datasets and compare with the distribution of datsets of zero-shot sets
# Almost the same number of samples across datasets in comparison with zero-shot sets
print("one_shot_train_set:\n", "len: ", len(one_shot_train_set), "\n", one_shot_train_set["Language"].value_counts(), "\n")
print("one_shot_dev_set:\n", "len: ", len(one_shot_dev_set), "\n", one_shot_dev_set["Language"].value_counts(), "\n")
print("one_shot_test_set:\n", "len: ", len(one_shot_test_set), "\n", one_shot_test_set["Language"].value_counts())

# Check the distribution of samples per languages and labels across sets
# And compare with the distribution of samples in one_shot sets
print("One_shot_train_set:\n", one_shot_train_set.groupby(["Language","label"]).size(), "\n")
print("one_shot_dev_set:\n", one_shot_dev_set.groupby(["Language","label"]).size(), "\n")
print("One_shot_test_set:\n", one_shot_test_set.groupby(["Language","label"]).size())

one_shot_train_set.to_csv("one_shot_train.csv", index=False)
one_shot_dev_set.to_csv("one_shot_dev.csv", index=False)
one_shot_test_set.to_csv("one_shot_test.csv", index=False)

with zipfile.ZipFile("one_shot_splits.zip", "w", zipfile.ZIP_DEFLATED) as zipf:
    zipf.write("one_shot_train.csv")
    zipf.write("one_shot_dev.csv")
    zipf.write("one_shot_test.csv")

"""Splitting whole data into zero_shot sets but considering the label distibution as well as language and number of MWE. The ratio of Label for each MWE per language is considered in splitting.

But the obtained sets have the same distribution as previous zero_shot splitting have. So we continued with the first zero_shot splits.  
"""

mwe_summary = (
    df.groupby(["Language", "MWE"])
      .agg(
          total_samples=("Label", "count"),
          label_0=("Label", lambda x: (x == 0).sum()),
          label_1=("Label", lambda x: (x == 1).sum())
      )
      .reset_index()
)

print(mwe_summary.head())

# Compute the proportion of label 1 for each MWE
mwe_summary["label_ratio"] = (
  mwe_summary["label_1"] / mwe_summary["total_samples"]
)

# Categorize each MWE based on the distribution of labels
def ratio_bucket(r):
  if r == 0:
    return "only_0"
  elif r == 1:
    return "only_1"
  else:
    return "mixed"


# Assign a label distribution category to each MWE
mwe_summary["label_bucket"] = mwe_summary["label_ratio"].apply(ratio_bucket)
print(mwe_summary)

# How many MWEs per language considering for dev and test sets
alloc = {
    "EN": 28,
    "PT": 17,
    "GL": 5
}

dev_mwes = []
test_mwes = []
train_mwes = []

# Perform language-wise stratified splitting
for lang in ["EN", "PT", "GL"]:

    # Select MWEs belonging to the current language
    lang_df = mwe_summary[mwe_summary["Language"] == lang]

    # First split for dev set
    # stratified by label distribution bucket to preserve balance
    dev_lang, remaining = train_test_split(
        lang_df,
        test_size=len(lang_df) - alloc[lang],
        stratify=lang_df["label_bucket"],
        random_state=42
    )

    # Then split remaining for test set
    # again using stratified sampling
    test_lang, train_lang = train_test_split(
        remaining,
        test_size=len(remaining) - alloc[lang],
        stratify=remaining["label_bucket"],
        random_state=42
    )

    # Collect the selected MWE identifiers for each split
    dev_mwes.extend(dev_lang["MWE"])
    test_mwes.extend(test_lang["MWE"])
    train_mwes.extend(train_lang["MWE"])

# Set all samples to corresponding sets
another_dev_set = df[df["MWE"].isin(dev_mwes)]
another_test_set = df[df["MWE"].isin(test_mwes)]
another_train_set = df[df["MWE"].isin(train_mwes)]

print("another_train_set:\n", another_train_set.groupby(["Language", "Label"]).size(), "\n")
print("another_dev_set:\n", another_dev_set.groupby(["Language", "Label"]).size(), "\n")
print("another_test_set:\n", another_test_set.groupby(["Language", "Label"]).size())

