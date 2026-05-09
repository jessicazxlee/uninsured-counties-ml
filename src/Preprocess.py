"""
Imports for preprocessing
"""
import os
import pandas as pd
from datacommons_client.client import DataCommonsClient

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer

def fetch_county_dataset():
    """
    Fetch data - datacommons_pandas is being phased out and doesn't have all data so switched to DataCommonsClient
    """
    client = DataCommonsClient(api_key="27AXb78fWqWGrxMZB6rJee47I4RXfhMEjNVGwtrLLDrDPMqM")
    variables = [
        "Count_Household_NoHealthInsurance",
        "Count_Person",
        "Median_Income_Household",
        "UnemploymentRate_Person",
    ]

    response = client.observation.fetch(
        variable_dcids=variables,
        entity_expression="country/USA<-containedInPlace+{typeOf:County}",
        date="latest"
    )

    records = response.to_observation_records().model_dump()
    df = pd.DataFrame(records)
    
    df = df.groupby(["entity", "variable"])["value"].mean().reset_index()
    df = df.pivot(index="entity", columns="variable", values="value").reset_index()

    df.rename(columns={"entity": "place"}, inplace=True)

    return df
    
def save_data(df, filename="county_dataset.csv"):
    base_dir = os.path.dirname(__file__)
    filepath = os.path.join(base_dir, "..", "data", filename)

    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    df.to_csv(filepath, index=False)

    return filepath
    
def load_data(filename):
    base_dir = os.path.dirname(__file__)
    filepath = os.path.join(base_dir, "..", "data", filename)
    df = pd.read_csv(filepath)
    return df


def split_data(df, target_column):
    """
    Split data into train and test sets BEFORE preprocessing
    to prevent data leakage
    """
    X = df.drop(columns=[target_column])
    Y = df[target_column]
    # 80% training and 20% test
    X_train, X_test, y_train, y_test = train_test_split(
        X, Y, test_size=0.2, random_state=42
    )
    return X_train, X_test, y_train, y_test

def get_feature_types(X):
    """
    Separate numeric and categorical columns
    """
    numeric_features = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
    categorical_features = X.select_dtypes(
        include=["object", "category"]
    ).columns.tolist()
    return numeric_features, categorical_features


def build_preprocessor(numeric_features, categorical_features):
    """
    Create a preprocessing pipeline that prevent leakage
    """
    # Numberic preprocessing
    numeric_pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),  # Jessica
            ("scaler", StandardScaler()),  # Mona
        ]
    )
    # Categorical prepocessing
    categorical_pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),  # Jessica
            ("encoder", OneHotEncoder(handle_unknown="ignore")),  # Daanish
        ]
    )
    # Combine pipelines
    preprocessor = ColumnTransformer(
        [
            ("num", numeric_pipeline, numeric_features),
            ("cat", categorical_pipeline, categorical_features),
        ]
    )
    return preprocessor



# Runs preprocessing
def preprocess_data(df, target_column):
    """
    Split data and apply preprocessing pipeline
    """
    X_train, X_test, y_train, y_test = split_data(df, target_column)

    numeric_features, categorical_features = get_feature_types(X_train)

    preprocessor = build_preprocessor(numeric_features, categorical_features)

    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)

    return X_train_processed, X_test_processed, y_train, y_test, preprocessor


"""
Testing / debugging
"""
if __name__ == "__main__":

    import numpy as np

    df = fetch_county_dataset()

    if "place" in df.columns:
        df = df.drop(columns=["place"])

    df["UninsuredRate"] = (
        df["Count_Household_NoHealthInsurance"] / df["Count_Person"]
    )

    df = df.dropna(subset=["UninsuredRate"])
    target_column = "UninsuredRate"

    X_train_processed, X_test_processed, y_train, y_test, preprocessor = preprocess_data(
        df, target_column
    )

    print("\nFirst few processed rows:")
    print(X_train_processed[:5])

    print("\nColumn means (should be close to 0):")
    print(np.mean(X_train_processed, axis=0))

    print("\nColumn std devs (should be close to 1):")
    print(np.std(X_train_processed, axis=0))