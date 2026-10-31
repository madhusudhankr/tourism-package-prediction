# for data manipulation
import pandas as pd

# The dataset lives inside the GitHub repo itself, so "registering" it means
# loading it from the repo's data folder and validating its structure.
DATASET_PATH = "tourism_project/data/tourism.csv"

# Load the raw dataset
df = pd.read_csv(DATASET_PATH)

# Validate that every expected column (as per the data dictionary) is present
expected_columns = [
    "CustomerID", "ProdTaken", "Age", "TypeofContact", "CityTier",
    "DurationOfPitch", "Occupation", "Gender", "NumberOfPersonVisiting",
    "NumberOfFollowups", "ProductPitched", "PreferredPropertyStar",
    "MaritalStatus", "NumberOfTrips", "Passport", "PitchSatisfactionScore",
    "OwnCar", "NumberOfChildrenVisiting", "Designation", "MonthlyIncome",
]
missing = [c for c in expected_columns if c not in df.columns]
if missing:
    raise ValueError(f"Dataset is missing expected columns: {missing}")

# Validate that the target column only contains the classes 0 and 1
if not set(df["ProdTaken"].unique()).issubset({0, 1}):
    raise ValueError("Target column 'ProdTaken' must only contain 0 and 1.")

# Print a short summary of the registered dataset
print("Dataset registered successfully.")
print(f"Rows: {df.shape[0]}, Columns: {df.shape[1]}")
print("Columns:", list(df.columns))
print(f"Total missing values: {int(df.isnull().sum().sum())}")
print(f"Duplicate rows (ignoring ID columns): {int(df.drop(columns=['CustomerID'] + [c for c in df.columns if c.startswith('Unnamed')]).duplicated().sum())}")

print("\nTarget distribution (ProdTaken):")
print(df["ProdTaken"].value_counts(normalize=True).round(3))

print("\nPurchase rate by key customer attributes:")
for col in ["Passport", "MaritalStatus", "ProductPitched"]:
    print(f"\n{col}:")
    print(df.groupby(col)["ProdTaken"].mean().round(3))
