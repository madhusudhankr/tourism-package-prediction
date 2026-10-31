# for data manipulation
import pandas as pd
# for splitting the data into train and test sets
from sklearn.model_selection import train_test_split

DATASET_PATH = "tourism_project/data/tourism.csv"

# Load the dataset directly from the repository data folder
df = pd.read_csv(DATASET_PATH)
print("Dataset loaded successfully.")
print(f"Shape before cleaning: {df.shape}")

# ---------------- Data cleaning ----------------
# 1. Drop columns that carry no predictive information:
#    - the unnamed leftover index column
#    - CustomerID (a unique identifier)
unnamed_cols = [c for c in df.columns if c.startswith("Unnamed")]
df = df.drop(columns=unnamed_cols + ["CustomerID"])

# 2. Fix the inconsistent category label in Gender ("Fe Male" -> "Female")
df["Gender"] = df["Gender"].replace({"Fe Male": "Female"})

# 3. Remove exact duplicate customer records
df = df.drop_duplicates()

# 4. Fill any missing values (safety net for new data that flows through the pipeline):
#    median for numeric columns, most frequent value for text columns
for col in df.columns:
    if df[col].isnull().any():
        if pd.api.types.is_numeric_dtype(df[col]):
            df[col] = df[col].fillna(df[col].median())
        else:
            df[col] = df[col].fillna(df[col].mode()[0])

print(f"Shape after cleaning: {df.shape}")

# ---------------- Train-test split ----------------
target_col = "ProdTaken"

# Split into X (features) and y (target)
X = df.drop(columns=[target_col])
y = df[target_col]

# stratify=y keeps the same share of buyers (~19%) in both train and test sets
Xtrain, Xtest, ytrain, ytest = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Save the splits locally; the workflow passes them to the next job as an artifact
Xtrain.to_csv("Xtrain.csv", index=False)
Xtest.to_csv("Xtest.csv", index=False)
ytrain.to_csv("ytrain.csv", index=False)
ytest.to_csv("ytest.csv", index=False)

print(f"Train set: {Xtrain.shape}, Test set: {Xtest.shape}")
print("Data prepared: train/test splits written.")
