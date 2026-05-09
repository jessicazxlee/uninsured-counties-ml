import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


def train_random_forest(X_train, y_train, X_test, y_test, preprocessor):
    """
    Train a tuned RandomForestRegressor.
    Designed to avoid overfitting on small feature sets
    and scale well when more features are added.
    """

    rf = RandomForestRegressor(
        n_estimators=150,
        max_depth=6,
        min_samples_split=5,
        min_samples_leaf=3,
        max_features="sqrt",
        bootstrap=True,
        random_state=42,
        n_jobs=-1
    )

    model = Pipeline([
        ("preprocessor", preprocessor),
        ("regressor", rf)
    ])

    print("\n[RF] Training Random Forest...")

    model.fit(X_train, y_train)

    print("[RF] Training complete.")

    y_pred = model.predict(X_test)

    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print("\nRandom Forest Results")
    print("---------------------")
    print(f"MSE:  {mse:.6f}")
    print(f"RMSE: {rmse:.6f}")
    print(f"MAE:  {mae:.6f}")
    print(f"R^2:  {r2:.4f}")

    return model, y_pred, {
        "mse": mse,
        "rmse": rmse,
        "mae": mae,
        "r2": r2
    }