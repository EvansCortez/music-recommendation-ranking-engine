import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
import implicit
import pickle

def run_retrieval():
    print("📥 Loading processed data...")
    df = pd.read_csv('data/processed/train_cleaned.csv')

    # 1. Create Sparse Matrix (User-Item Interactions)
    # We use 'target' as a proxy for 'confidence' 
    # (1 = they liked it enough to repeat, so it's a strong signal)
    user_items = csr_matrix((df['target'].astype(float), 
                             (df['msno'], df['song_id'])))

    # 2. Initialize the ALS Model
    # factors: size of the 'vibe' vector
    # iterations: how many times to refine the math
    print("🎸 Training ALS model (Collaborative Filtering)...")
    model = implicit.als.AlternatingLeastSquares(factors=64, iterations=20, regularization=0.1)
    
    # Train the model (Implicit expects Item-User for training)
    model.fit(user_items.T.tocsr())

    # 3. Generate Candidates for a sample user
    # In a real app, you'd do this for all users in your test set
    user_id = 0
    ids, scores = model.recommend(user_id, user_items[user_id], N=100)
    
    print(f"✅ Generated {len(ids)} candidates for User {user_id}")

    # 4. Save the model to use in our final pipeline
    with open('src/retrieval_model.pkl', 'wb') as f:
        pickle.dump(model, f)

if __name__ == "__main__":
    run_retrieval()