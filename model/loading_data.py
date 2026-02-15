import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

def load_data():
    # Load train & test CSVs
    train_df = pd.read_csv("data/train.csv")
    test_df = pd.read_csv("data/test.csv")

    if "id" in test_df.columns:
        test_df = test_df.drop("id", axis=1)
    # Features and target
    X = train_df.drop("price_range", axis=1)
    y = train_df["price_range"]

    # Scale (fit ONLY on train)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    joblib.dump(scaler, "scaler.pkl")

    # Create validation set (for evaluation)
    X_train, X_val, y_train, y_val = train_test_split(
        X_scaled, y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    # Prepare test data (NO labels)
    X_test_final = scaler.transform(test_df)

    return X_train, X_val, y_train, y_val, X_test_final
