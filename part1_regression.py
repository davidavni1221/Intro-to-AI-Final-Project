import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from ucimlrepo import fetch_ucirepo
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, SGDRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import PolynomialFeatures
from sklearn.neighbors import KNeighborsRegressor

# --- 1. Data Loading & Preprocessing ---
print("Fetching dataset...")
auto_mpg = fetch_ucirepo(id=9)
X = auto_mpg.data.features
y = auto_mpg.data.targets

# Combine for easy EDA
df = pd.concat([X, y], axis=1)

# Handle missing horsepower values using the median
df['horsepower'] = df['horsepower'].fillna(df['horsepower'].median())

# --- 2. Exploratory Data Analysis (EDA) & Correlation Matrix ---
plt.figure(figsize=(10, 8))
correlation_matrix = df.corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title("Feature Correlation Matrix")
plt.savefig('correlation_matrix.png', dpi=300, bbox_inches='tight')
plt.show()

# --- 3. Data Splitting (Train: 70%, Validation: 15%, Test: 15%) ---
X_features = df.drop('mpg', axis=1)
y_target = df['mpg']

# First split: Separate out the Test set (15%)
X_temp, X_test, y_temp, y_test = train_test_split(X_features, y_target, test_size=0.15, random_state=42)
# Second split: Divide the remaining 85% into Train and Validation
# To get 15% of original for validation: 0.15 / 0.85 ≈ 0.1764
X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.1764, random_state=42)

# --- 4. Data Normalization (Fitted ONLY on Train to prevent Data Leakage) ---
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

print(f"Train size: {X_train_scaled.shape[0]}, Val size: {X_val_scaled.shape[0]}, Test size: {X_test_scaled.shape[0]}")

# --- 5. Linear Regression Baseline (Evaluated on Validation Set) ---
lr_model = LinearRegression()
lr_model.fit(X_train_scaled, y_train)
y_val_pred_lr = lr_model.predict(X_val_scaled)

mse_lr = mean_squared_error(y_val, y_val_pred_lr)
print(f"Linear Regression Validation MSE: {mse_lr:.2f}")

plt.figure(figsize=(8, 6))
plt.scatter(y_val, y_val_pred_lr, alpha=0.6, color='royalblue')
plt.plot([y_val.min(), y_val.max()], [y_val.min(), y_val.max()], color='firebrick', linestyle='--', lw=2)
plt.xlabel('Actual MPG (Validation)')
plt.ylabel('Predicted MPG')
plt.title('Baseline: Actual vs Predicted (Validation Set)')
plt.savefig('linear_baseline.png', dpi=300, bbox_inches='tight')
plt.show()

# --- 6. Polynomial Regression (Complexity Tuning) ---
degrees = range(1, 6)
train_errors, val_errors = [], []

for deg in degrees:
    poly = PolynomialFeatures(degree=deg)
    X_train_poly = poly.fit_transform(X_train_scaled)
    X_val_poly = poly.transform(X_val_scaled)

    model = LinearRegression()
    model.fit(X_train_poly, y_train)

    train_errors.append(mean_squared_error(y_train, model.predict(X_train_poly)))
    val_errors.append(mean_squared_error(y_val, model.predict(X_val_poly)))

plt.figure(figsize=(10, 6))
plt.plot(degrees, train_errors, label='Train Error', marker='o')
plt.plot(degrees, val_errors, label='Validation Error', marker='o')
plt.xlabel('Polynomial Degree')
plt.ylabel('MSE')
plt.title('Model Complexity Curve (Polynomial Regression)')
plt.legend()
plt.savefig('complexity_curve.png', dpi=300, bbox_inches='tight')
plt.show()

# --- 7. KNN Regression (Hyperparameter Tuning) ---
k_values = range(1, 21)
knn_train_errors, knn_val_errors = [], []

for k in k_values:
    knn = KNeighborsRegressor(n_neighbors=k)
    knn.fit(X_train_scaled, y_train)
    knn_train_errors.append(mean_squared_error(y_train, knn.predict(X_train_scaled)))
    knn_val_errors.append(mean_squared_error(y_val, knn.predict(X_val_scaled)))

plt.figure(figsize=(10, 6))
plt.plot(k_values, knn_train_errors, label='Train Error', marker='o')
plt.plot(k_values, knn_val_errors, label='Validation Error', marker='o')
plt.xticks(k_values)
plt.xlabel('Number of Neighbors (k)')
plt.ylabel('MSE')
plt.title('KNN Regression: k vs Validation Error')
plt.legend()
plt.savefig('knn_k_effect.png', dpi=300, bbox_inches='tight')
plt.show()

best_k = k_values[np.argmin(knn_val_errors)]
print(f"Optimal K value: {best_k}")


# --- 8. Optimization Behavior: SGD vs Mini-Batch GD ---
def train_optimization(batch_size, label):
    model = SGDRegressor(learning_rate='constant', eta0=0.01, random_state=42)
    val_mse_history = []

    epochs = 50
    for epoch in range(epochs):
        # Shuffle data for each epoch
        indices = np.random.permutation(len(X_train_scaled))
        X_shuffled = X_train_scaled[indices]
        y_shuffled = y_train.iloc[indices].values if isinstance(y_train, pd.Series) else y_train[indices]

        for i in range(0, len(X_train_scaled), batch_size):
            X_batch = X_shuffled[i:i + batch_size]
            y_batch = y_shuffled[i:i + batch_size]
            model.partial_fit(X_batch, y_batch)

        # Record validation error after each epoch
        val_mse_history.append(mean_squared_error(y_val, model.predict(X_val_scaled)))

    return val_mse_history


sgd_history = train_optimization(batch_size=1, label="Stochastic GD (Batch=1)")
minibatch_history = train_optimization(batch_size=32, label="Mini-Batch GD (Batch=32)")

plt.figure(figsize=(10, 6))
plt.plot(sgd_history, label='Stochastic GD (Batch Size = 1)', alpha=0.7)
plt.plot(minibatch_history, label='Mini-Batch GD (Batch Size = 32)', alpha=0.7)
plt.xlabel('Epochs')
plt.ylabel('Validation MSE')
plt.title('Optimization Strategy Comparison: SGD vs Mini-Batch GD')
plt.legend()
plt.savefig('optimization_curve.png', dpi=300, bbox_inches='tight')
plt.show()

# --- 9. FINAL EVALUATION ON TEST SET ---
# We use the best model found (e.g., KNN with best_k) and evaluate it ONCE on the test set.
final_model = KNeighborsRegressor(n_neighbors=best_k)
final_model.fit(X_train_scaled, y_train)  # Train on Train set
y_test_pred = final_model.predict(X_test_scaled)  # Test on Test set

final_mse = mean_squared_error(y_test, y_test_pred)
final_r2 = r2_score(y_test, y_test_pred)

print(f"\n--- Final Model Performance on HOLD-OUT TEST SET ---")
print(f"Final Model: KNN (k={best_k})")
print(f"Test MSE: {final_mse:.2f}")
print(f"Test R2 Score: {final_r2:.2f}")