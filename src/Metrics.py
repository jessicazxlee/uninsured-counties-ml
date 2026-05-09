import os
import matplotlib.pyplot as plt

from Preprocess import (
    fetch_county_dataset as fetch_en_dataset,
    build_preprocessor as build_en_preprocessor,
    get_feature_types as get_en_feature_types,
    split_data as split_en_data
)

from preprocess_rf import (
    fetch_county_dataset as fetch_rf_dataset,
    build_preprocessor as build_rf_preprocessor,
    get_feature_types as get_rf_feature_types,
    split_data as split_rf_data
)

from models.elastic_net import train_elastic_net
from models.random_forest import train_random_forest


def save_plot(y_test, y_pred, title, filename):
    base_dir = os.path.dirname(__file__)
    figures_dir = os.path.join(base_dir, "..", "docs", "figures")
    os.makedirs(figures_dir, exist_ok=True)

    path = os.path.join(figures_dir, filename)

    plt.figure()
    plt.scatter(y_test, y_pred)
    plt.plot(
        [y_test.min(), y_test.max()],
        [y_test.min(), y_test.max()],
        linestyle="--"
    )
    plt.xlabel("Actual")
    plt.ylabel("Predicted")
    plt.title(title)
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.show()

    print(f"Saved: {path}")


def prepare_dataset(df):
    if "place" in df.columns:
        df = df.drop(columns=["place"])

    df["UninsuredRate"] = (
        df["Count_Household_NoHealthInsurance"] / df["Count_Person"]
    )

    df = df.dropna(subset=["UninsuredRate"])
    df = df.drop(columns=[
        "Count_Household_NoHealthInsurance",
        "Count_Person"
    ])

    return df, "UninsuredRate"


def run():
    en_df = fetch_en_dataset()
    en_df, en_target = prepare_dataset(en_df)

    en_X_train, en_X_test, en_y_train, en_y_test = split_en_data(en_df, en_target)

    print("\n=== ELASTIC NET FEATURES BEING USED ===")
    print(en_X_train.columns)

    en_num, en_cat = get_en_feature_types(en_X_train)
    en_preprocessor = build_en_preprocessor(en_num, en_cat)

    en_model, en_pred, en_metrics = train_elastic_net(
        en_X_train, en_y_train, en_X_test, en_y_test, en_preprocessor
    )

    print("\nElasticNet Metrics:", en_metrics)

    save_plot(
        en_y_test,
        en_pred,
        "ElasticNet: Actual vs Predicted",
        "elasticnet.png"
    )

    rf_df = fetch_rf_dataset()
    rf_df, rf_target = prepare_dataset(rf_df)

    rf_X_train, rf_X_test, rf_y_train, rf_y_test = split_rf_data(rf_df, rf_target)

    print("\n=== RANDOM FOREST FEATURES BEING USED ===")
    print(rf_X_train.columns)

    rf_num, rf_cat = get_rf_feature_types(rf_X_train)
    rf_preprocessor = build_rf_preprocessor(rf_num, rf_cat)

    rf_model, rf_pred, rf_metrics = train_random_forest(
        rf_X_train, rf_y_train, rf_X_test, rf_y_test, rf_preprocessor
    )

    print("\nRandom Forest Metrics:", rf_metrics)

    save_plot(
        rf_y_test,
        rf_pred,
        "Random Forest: Actual vs Predicted",
        "random_forest.png"
    )

    print("\n=== COMPARISON ===")
    print("ElasticNet:", en_metrics)
    print("RandomForest:", rf_metrics)


if __name__ == "__main__":
    run()