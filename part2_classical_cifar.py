import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
import torchvision

# --- 1. Data Loading (Automatic Download via PyTorch) ---
print("Downloading and loading CIFAR-10 dataset...")
# This will download the data to a folder called 'data' in your project
train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True)
test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True)

# Extract data and labels (PyTorch stores images as numpy arrays of shape N x 32 x 32 x 3)
X_train_raw = train_dataset.data
y_train_raw = np.array(train_dataset.targets)
X_test_raw = test_dataset.data
y_test_raw = np.array(test_dataset.targets)

# --- 2. Preprocessing & Sub-sampling ---
# Flatten the images from 32x32x3 into a 1D array of 3072 features, and normalize to [0, 1]
X_train_scaled = X_train_raw.reshape(X_train_raw.shape[0], -1) / 255.0
X_test_scaled = X_test_raw.reshape(X_test_raw.shape[0], -1) / 255.0

# Take a subset of 6000 samples to split into 5000 Train and 1000 Validation (to save computation time)
subset_size = 6000
X_subset = X_train_scaled[:subset_size]
y_subset = y_train_raw[:subset_size]

X_train_sub, X_val_sub, y_train_sub, y_val_sub = train_test_split(
    X_subset, y_subset, test_size=1000, random_state=42, stratify=y_subset
)
print(f"Subset Train Size: {X_train_sub.shape[0]}, Subset Val Size: {X_val_sub.shape[0]}")

# --- 3. Hyperparameter Tuning: K-Nearest Neighbors ---
print("Tuning KNN...")
k_values = [1, 3, 5, 7, 9]
knn_val_acc = []

for k in k_values:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train_sub, y_train_sub)
    knn_val_acc.append(accuracy_score(y_val_sub, knn.predict(X_val_sub)))

best_k = k_values[np.argmax(knn_val_acc)]

plt.figure(figsize=(8, 5))
plt.plot(k_values, knn_val_acc, marker='o', color='green')
plt.title('KNN Hyperparameter Tuning')
plt.xlabel('Number of Neighbors (k)')
plt.ylabel('Validation Accuracy')
plt.xticks(k_values)
plt.grid(True)
plt.savefig('knn_tuning.png', dpi=300, bbox_inches='tight')
plt.show()

# --- 4. Hyperparameter Tuning: Linear SVM ---
# Required to be LINEAR by instructions
print("Tuning Linear SVM (This might take a minute)...")
c_values = [0.01, 0.1, 1.0, 10.0]
svm_val_acc = []

for c in c_values:
    svm = SVC(kernel='linear', C=c, random_state=42)
    svm.fit(X_train_sub, y_train_sub)
    svm_val_acc.append(accuracy_score(y_val_sub, svm.predict(X_val_sub)))

best_c_svm = c_values[np.argmax(svm_val_acc)]

plt.figure(figsize=(8, 5))
plt.plot([str(c) for c in c_values], svm_val_acc, marker='o', color='blue')
plt.title('Linear SVM Hyperparameter Tuning')
plt.xlabel('Regularization Strength (C)')
plt.ylabel('Validation Accuracy')
plt.grid(True)
plt.savefig('svm_tuning.png', dpi=300, bbox_inches='tight')
plt.show()

# --- 5. Hyperparameter Tuning: Logistic Regression ---
print("Tuning Logistic Regression...")
lr_val_acc = []
for c in c_values:
    lr = LogisticRegression(C=c, solver='lbfgs', max_iter=1000, random_state=42)
    lr.fit(X_train_sub, y_train_sub)
    lr_val_acc.append(accuracy_score(y_val_sub, lr.predict(X_val_sub)))

best_c_lr = c_values[np.argmax(lr_val_acc)]

# --- 6. Final Evaluation on TEST SET with the best models ---
print("\n--- Final Test Set Evaluation ---")
# 1. Best KNN
final_knn = KNeighborsClassifier(n_neighbors=best_k)
final_knn.fit(X_train_sub, y_train_sub)
print(f"Final KNN (k={best_k}) Test Accuracy: {accuracy_score(y_test_raw, final_knn.predict(X_test_scaled)):.4f}")

# 2. Best LR
final_lr = LogisticRegression(C=best_c_lr, solver='lbfgs', max_iter=1000, random_state=42)
final_lr.fit(X_train_sub, y_train_sub)
print(f"Final Logistic Reg (C={best_c_lr}) Test Accuracy: {accuracy_score(y_test_raw, final_lr.predict(X_test_scaled)):.4f}")

# 3. Best Linear SVM
final_svm = SVC(kernel='linear', C=best_c_svm, random_state=42)
final_svm.fit(X_train_sub, y_train_sub)
y_pred_svm_test = final_svm.predict(X_test_scaled)
print(f"Final Linear SVM (C={best_c_svm}) Test Accuracy: {accuracy_score(y_test_raw, y_pred_svm_test):.4f}")

# --- 7. Error Analysis (Confusion Matrix for Best Model) ---
classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
plt.figure(figsize=(10, 8))
cm = confusion_matrix(y_test_raw, y_pred_svm_test)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
plt.xlabel('Predicted Label')
plt.ylabel('Actual Label')
plt.title(f'Confusion Matrix - Linear SVM (C={best_c_svm}) on Test Set')
plt.savefig('confusion_matrix_cifar.png', dpi=300, bbox_inches='tight')
plt.show()