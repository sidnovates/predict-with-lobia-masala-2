import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# --- 1. Master Preprocessing Pipeline ---

# Load Raw Data
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')
test_raw = pd.read_csv('test.csv') 

# Target Handling 
train = train.dropna(subset=['spend_category'])
y = train['spend_category']

# Keep 'spend_category' in train_features for Target Encoding
train_features = train.drop(['trip_id'], axis=1) 
test_features = test.drop(['trip_id'], axis=1)


# --- Target Encoding for Country ---

# 1. Calculate the mean 'spend_category' for each country using the training data
country_target_map = train_features.groupby('country')['spend_category'].mean()
global_mean_target = y.mean()

# 2. Apply the mapping to both training and test data
train_features['country_target_encoded'] = train_features['country'].map(country_target_map).fillna(global_mean_target)
test_features['country_target_encoded'] = test_features['country'].map(country_target_map).fillna(global_mean_target)

# 3. Drop the original 'country' and 'spend_category' from the feature sets before combining
train_features = train_features.drop(['country', 'spend_category'], axis=1)
test_features = test_features.drop('country', axis=1)

# Combine for consistent imputation/encoding of other features
all_data = pd.concat([train_features, test_features], axis=0).reset_index(drop=True)


# --- Imputation, Feature Engineering, & Encoding ---

# Imputation 
num_cols = ['num_females', 'num_males']
for col in num_cols:
    all_data[col] = all_data[col].fillna(all_data[col].median())

all_data['has_special_requirements'] = all_data['has_special_requirements'].fillna('none')
cat_cols = all_data.select_dtypes(include=['object']).columns
for col in cat_cols:
    if all_data[col].isnull().sum() > 0:
        all_data[col] = all_data[col].fillna(all_data[col].mode()[0])

# Feature Engineering
all_data['total_people'] = all_data['num_females'] + all_data['num_males']

# Ordinal Encoding
age_map = {'<18': 0, '18-24': 1, '25-44': 2, '45-64': 3, '65+': 4}
all_data['age_group'] = all_data['age_group'].map(age_map).fillna(-1)
trip_map = {'1-6': 0, '7-14': 1, '15-30': 2, '30+': 3}
all_data['total_trip_days'] = all_data['total_trip_days'].map(trip_map).fillna(-1)
all_data['days_booked_before_trip'] = all_data['days_booked_before_trip'].str.strip()
book_map = {'1-7': 0, '8-14': 1, '15-30': 2, '31-60': 3, '61-90': 4, '90+': 5}
all_data['days_booked_before_trip'] = all_data['days_booked_before_trip'].map(book_map).fillna(-1)

# One-Hot Encoding
all_data = pd.get_dummies(all_data, drop_first=True)

# Split back
X = all_data.iloc[:len(y), :]
X_test = all_data.iloc[len(y):, :]


# --- 2. Random Forest Training and Submission ---

# Split for Validation
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print("Training Random Forest Model with Class Weight Balancing...")
model = RandomForestClassifier(
    n_estimators=300,        # Number of trees
    max_depth=15,            # Max depth to prevent deep overfitting
    min_samples_leaf=5,      # Ensure a minimum number of samples per leaf
    random_state=42,
    n_jobs=-1,
    # CRITICAL: This handles the imbalance by weighting the minority class higher
    class_weight='balanced' 
)

model.fit(X_train, y_train)

# Evaluation (Validation Set)
val_preds = model.predict(X_val)
print(f"\nValidation Accuracy (Random Forest): {accuracy_score(y_val, val_preds):.4f}")
print("Classification Report (Check 'Recall' for Class 2):")
print(classification_report(y_val, val_preds))


# Final Prediction and Submission
final_predictions = model.predict(X_test)

submission = pd.DataFrame({
    'trip_id': test_raw['trip_id'], 
    'spend_category': final_predictions.astype(int) 
})

submission.to_csv('submission_randomforest.csv', index=False)
print("Done! Submission saved to 'submission_randomforest.csv'")