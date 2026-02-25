import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import joblib

def train_ranking_model():
    print("📂 Loading preprocessed data for ranking...")
    df = pd.read_csv('data/processed/train_cleaned.csv')

    # Define features (Exclude IDs that are too unique and the target)
    features = [
        'source_system_tab', 'source_screen_name', 'source_type',
        'city', 'bd', 'gender', 'registered_via', 'account_age',
        'artist_name', 'song_length'
    ]
    
    X = df[features]
    y = df['target']

    # Split data: 80% Train, 20% Validation
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    print("🌲 Training LightGBM Ranker...")
    
    # Parameters optimized for binary classification on tabular music data
    params = {
        'objective': 'binary',
        'metric': 'auc',
        'boosting_type': 'gbdt',
        'verbosity': -1,
        'learning_rate': 0.05,
        'num_leaves': 31,
        'feature_fraction': 0.9,
        'bagging_fraction': 0.8,
        'bagging_freq': 5
    }

    # Create LightGBM datasets
    train_data = lgb.Dataset(X_train, label=y_train)
    val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)

    # Train with early stopping to prevent overfitting
    model = lgb.train(
        params,
        train_data,
        num_boost_round=1000,
        valid_sets=[val_data],
        callbacks=[lgb.early_stopping(stopping_rounds=50)]
    )

    # 📈 Evaluation
    preds = model.predict(X_val)
    auc_score = roc_auc_score(y_val, preds)
    print(f"✅ Training Complete! Validation AUC: {auc_score:.4f}")

    # Save the model
    joblib.dump(model, 'src/ranker_model.pkl')
    print("💾 Model saved to src/ranker_model.pkl")

if __name__ == "__main__":
    train_ranking_model()