import numpy as np
import pandas as pd
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
import pickle

# Load dataset
df = pd.read_excel(r"C:\Users\sravi\Downloads\ICTAK\project stage 1\E Commerce Dataset .xlsx", sheet_name=1)

# Data preprocessing
df['PreferredLoginDevice'] = df['PreferredLoginDevice'].replace(['Phone'], ['Mobile Phone'])
df['PreferedOrderCat'] = df['PreferedOrderCat'].replace(['Mobile'], ['Mobile Phone'])
df['PreferredPaymentMode'] = df['PreferredPaymentMode'].replace(['Cash on Delivery'], ['COD'])
df['PreferredPaymentMode'] = df['PreferredPaymentMode'].replace(['Credit Card'], ['CC'])

cats = ['PreferredLoginDevice', 'PreferredPaymentMode', 'Gender', 'PreferedOrderCat', 'MaritalStatus']
ordinal = ['CityTier', 'SatisfactionScore']
binary = ['Churn', 'Complain']
nums = df.loc[:, ~df.columns.isin(cats + binary + ordinal)].columns[1:]

df['Tenure'].fillna(df['Tenure'].median(), inplace=True)
df['WarehouseToHome'].fillna(df['WarehouseToHome'].median(), inplace=True)
df['OrderAmountHikeFromlastYear'].fillna(df['OrderAmountHikeFromlastYear'].median(), inplace=True)
df['CouponUsed'].fillna(df['CouponUsed'].median(), inplace=True)
df['OrderCount'].fillna(df['OrderCount'].median(), inplace=True)
df['DaySinceLastOrder'].fillna(df['DaySinceLastOrder'].median(), inplace=True)
df['HourSpendOnApp'].fillna(df['HourSpendOnApp'].mean(), inplace=True)

filtered_entries = np.array([True] * len(df))
for col in nums:
    zscore = abs(stats.zscore(df[col]))
    filtered_entries = (zscore < 3) & filtered_entries
df = df[filtered_entries]

df['Gender'] = df['Gender'].replace(['Female', 'Male'], [0, 1])
df['PreferredLoginDevice'] = df['PreferredLoginDevice'].replace(['Mobile Phone'], ['Handphone'])
df['PreferredPaymentMode'] = df['PreferredPaymentMode'].replace(['CC', 'COD'], ['Credit Card', 'Cash on Delivery'])
df['PreferedOrderCat'] = df['PreferedOrderCat'].replace(['Mobile Phone', 'Laptop & Accessory'], ['Electronics', 'Electronics'])

df = pd.get_dummies(df, columns=['MaritalStatus'], prefix='Marital')
df = pd.get_dummies(df)
df['AvgCashback'] = df['CashbackAmount'] / df['OrderCount']

# Update the feature set
features_to_include = ['PreferredPaymentMode', 'HourSpendOnApp', 'NumberOfDeviceRegistered', 'PreferedOrderCat', 'MaritalStatus']
for feature in features_to_include:
    if feature not in df.columns:
        df[feature] = 0  # Add new feature columns filled with 0 if not present

columns_to_scale = ['Tenure', 'CityTier', 'WarehouseToHome', 'HourSpendOnApp', 'NumberOfDeviceRegistered', 
                    'SatisfactionScore', 'NumberOfAddress', 'OrderAmountHikeFromlastYear', 'CouponUsed', 
                    'OrderCount', 'DaySinceLastOrder', 'CashbackAmount', 'AvgCashback']
for col in columns_to_scale:
    df[col] = StandardScaler().fit_transform(df[col].values.reshape(len(df), 1))

correlated_features = [col for col in df.columns if abs(df['Churn'].corr(df[col])) > 0.05]
df_final = df[correlated_features]

# Define features and target variable
X = df_final.drop('Churn', axis=1)
y = df_final['Churn']
X.columns = [col.replace(' ', '_') for col in X.columns]

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=10)
X_train, y_train = SMOTE(random_state=10).fit_resample(X_train, y_train)

# Initialize and train the Gradient Boosting model
gb_classifier = GradientBoostingClassifier(n_estimators=200, learning_rate=0.1, max_depth=8)
gb_classifier.fit(X_train, y_train)

# Make predictions
y_pred = gb_classifier.predict(X_test)

# Evaluate model performance
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
class_report = classification_report(y_test, y_pred)

print("Accuracy:", accuracy)
print("Confusion Matrix:\n", conf_matrix)
print("Classification Report:\n", class_report)

# Perform cross-validation
cross_val_scores = cross_val_score(gb_classifier, X, y, cv=5, scoring='accuracy')
print("Cross-validation scores:", cross_val_scores)
print("Average cross-validation score:", cross_val_scores.mean())

# Hyperparameter tuning using GridSearchCV
param_grid = {
    'n_estimators': [50, 100, 200],
    'learning_rate': [0.1, 0.01, 0.001],
    'max_depth': [3, 5, 8]
}

grid_search = GridSearchCV(GradientBoostingClassifier(), param_grid, cv=5, scoring='accuracy')
grid_search.fit(X_train, y_train)

best_params_gb = grid_search.best_params_
best_score_gb = grid_search.best_score_

print("Best Parameters for Gradient Boosting:", best_params_gb)
print("Best Cross-validation Accuracy for Gradient Boosting:", best_score_gb)


# Save the final model
model_filename = 'final_gradient_boosting_model.pkl'
with open(model_filename, 'wb') as file:
    pickle.dump(gb_classifier, file)

print(f'Model saved to {model_filename}')

# Load the model to verify
with open(model_filename, 'rb') as file:
    loaded_gb_model = pickle.load(file)

# Verify the loaded model's performance
y_pred_loaded = loaded_gb_model.predict(X_test)
assert (y_pred_loaded == y_pred).all()