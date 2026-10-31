# for data manipulation
import pandas as pd
# for data preprocessing and pipeline creation
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline
# for model training, tuning, and evaluation
import xgboost as xgb
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
# for model serialization
import joblib
# for experiment tracking
import mlflow

# The workflow starts an MLflow server on port 5000 before this script runs
mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("tourism-package-training")

# Xtrain/Xtest/ytrain/ytest are downloaded from the previous job's artifact
Xtrain = pd.read_csv("Xtrain.csv")
Xtest = pd.read_csv("Xtest.csv")
ytrain = pd.read_csv("ytrain.csv").squeeze()
ytest = pd.read_csv("ytest.csv").squeeze()

# Define numeric and categorical features
numeric_features = [
    "Age", "CityTier", "DurationOfPitch", "NumberOfPersonVisiting",
    "NumberOfFollowups", "PreferredPropertyStar", "NumberOfTrips", "Passport",
    "PitchSatisfactionScore", "OwnCar", "NumberOfChildrenVisiting", "MonthlyIncome",
]
categorical_features = [
    "TypeofContact", "Occupation", "Gender", "ProductPitched",
    "MaritalStatus", "Designation",
]

# Only ~19% of customers bought the package, so give the minority class more weight
class_weight = ytrain.value_counts()[0] / ytrain.value_counts()[1]

# Preprocessor: scale numeric columns, one-hot encode text columns
preprocessor = make_column_transformer(
    (StandardScaler(), numeric_features),
    (OneHotEncoder(handle_unknown="ignore"), categorical_features),
)

# Define base XGBoost Classifier
xgb_model = xgb.XGBClassifier(
    scale_pos_weight=class_weight, random_state=42, n_jobs=-1, eval_metric="logloss"
)

# Hyperparameter grid (kept small so the pipeline runs quickly on GitHub Actions)
param_grid = {
    "xgbclassifier__n_estimators": [50, 100],
    "xgbclassifier__max_depth": [3, 5],
    "xgbclassifier__learning_rate": [0.05, 0.1],
    "xgbclassifier__colsample_bytree": [0.7, 0.9],
}

# Pipeline
model_pipeline = make_pipeline(preprocessor, xgb_model)

with mlflow.start_run():
    # Grid Search - F1 balances finding buyers (recall) with not wasting calls (precision)
    grid_search = GridSearchCV(model_pipeline, param_grid, cv=3, n_jobs=-1, scoring="f1")
    grid_search.fit(Xtrain, ytrain)

    # Log every tuned parameter combination and its mean cross-validation score
    results = grid_search.cv_results_
    for i in range(len(results["params"])):
        param_set = results["params"][i]
        mean_score = results["mean_test_score"][i]

        with mlflow.start_run(nested=True):
            mlflow.log_params(param_set)
            mlflow.log_metric("mean_cv_f1", mean_score)

    # Log best parameters separately in the main run
    mlflow.log_params(grid_search.best_params_)

    # Best model
    best_model = grid_search.best_estimator_
    print("Best params:", grid_search.best_params_)

    # Predictions
    y_pred_train = best_model.predict(Xtrain)
    y_pred_test = best_model.predict(Xtest)

    # Evaluation
    print("\nClassification report (train):")
    print(classification_report(ytrain, y_pred_train))
    print("Classification report (test):")
    print(classification_report(ytest, y_pred_test))

    # Log metrics
    mlflow.log_metrics({
        "train_accuracy": accuracy_score(ytrain, y_pred_train),
        "test_accuracy": accuracy_score(ytest, y_pred_test),
        "train_precision": precision_score(ytrain, y_pred_train),
        "test_precision": precision_score(ytest, y_pred_test),
        "train_recall": recall_score(ytrain, y_pred_train),
        "test_recall": recall_score(ytest, y_pred_test),
        "train_f1": f1_score(ytrain, y_pred_train),
        "test_f1": f1_score(ytest, y_pred_test),
    })

    # Save next to app.py so the Streamlit app can load it directly, and log
    # it as an MLflow artifact for traceability
    model_path = "tourism_project/deployment/best_tourism_model_v1.joblib"
    joblib.dump(best_model, model_path)
    mlflow.log_artifact(model_path, artifact_path="model")
    print(f"Model saved to {model_path}")
