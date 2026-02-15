from loading_data import load_data
from model_training import train_models
from model_evaluation import evaluate

X_train, X_val, y_train, y_val, X_test_final = load_data()
models = train_models(X_train, y_train)

for name, model in models.items():
    print("\n====================")
    print(name)

    # Evaluate on VALIDATION set
    metrics = evaluate(model, X_val, y_val)
    for k, v in metrics.items():
        print(f"{k}: {v}")

# OPTIONAL: generate predictions on final test set
print("\nGenerating final test predictions...")
for name, model in models.items():
    preds = model.predict(X_test_final)
    print(f"{name} predictions shape:", preds.shape)
