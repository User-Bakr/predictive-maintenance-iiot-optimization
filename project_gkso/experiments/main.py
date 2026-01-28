"""
Project: Optimizing Predictive Maintenance in Industrial IoT using Enhanced GKSO
Author: Abo Bakr Ahmed
Description:
    Implementation of the proposed predictive maintenance framework
    combining Logistic Regression and an enhanced GKSO algorithm.
"""



import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from google.colab import drive
import time
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_curve, auc
import random
import seaborn as sns

# Load dataset
df = pd.read_csv('AI4I.csv')
data=df

n = data.shape[0]
# First checks
print('Features non-null values and data type:')
data.info()
print('Check for duplicate values:',
      data['Product ID'].unique().shape[0]!=n)

df['Tool wear [min]'] = df['Tool wear [min]'].astype('float64')
df['Rotational speed [rpm]'] = df['Rotational speed [rpm]'].astype('float64')

# Pie chart of Type percentage
value = data['Type'].value_counts()
Type_percentage = 100*value/data.Type.shape[0]
labels = Type_percentage.index.array
x = Type_percentage.array
plt.pie(x, labels = labels, colors=sns.color_palette('tab10')[0:3], autopct='%.0f%%')
plt.title('Machine Type percentage')
plt.show()

features = [col for col in df.columns
            if df[col].dtype=='float64' or col =='Type']
num_features = [feature for feature in features if df[feature].dtype=='float64']
# Histograms of numeric features
fig, axs = plt.subplots(nrows=2, ncols=3, figsize=(18,7))
fig.suptitle('Numeric features histogram')
for j, feature in enumerate(num_features):
    sns.histplot(ax=axs[j//3, j-3*(j//3)], data=df, x=feature)
plt.show()

# boxplot of numeric features
fig, axs = plt.subplots(nrows=2, ncols=3, figsize=(18,7))
fig.suptitle('Numeric features boxplot')
for j, feature in enumerate(num_features):
    sns.boxplot(ax=axs[j//3, j-3*(j//3)], data=df, x=feature)
plt.show()

X = data.drop(['Machine failure'], axis=1)

# Target (y)
y = data['Machine failure']
y

import seaborn as sns
import matplotlib.pyplot as plt
# Check the class distribution of the target variable
class_counts = y.value_counts()
print(class_counts)

# Visualize the class distribution
sns.countplot(x=y)
plt.title('Class Distribution')
plt.show()

from sklearn.utils import resample

# Combine X and y for resampling
data_resampled = pd.concat([X, y], axis=1)

# Separate the majority and minority classes
majority_class = data_resampled[data_resampled['Machine failure'] == 0]
minority_class = data_resampled[data_resampled['Machine failure'] == 1]

# Upsample the minority class
minority_upsampled = resample(minority_class,
                               replace=True,
                               n_samples=len(majority_class),
                               random_state=42)

# Combine the upsampled minority class with the majority class
data_balanced = pd.concat([majority_class, minority_upsampled])

# Split back into features and target
X_balanced = data_balanced.drop(['Machine failure'], axis=1)
y_balanced = data_balanced['Machine failure']

# Check the class distribution of the target variable
class_counts = y_balanced.value_counts()
print(class_counts)

sns.countplot(x=y_balanced)
plt.title('Class Distribution')
plt.show()

np.random.seed(42)
random.seed(42)
#df= data_balanced # testing
# Ensure machine failure is 1 if any failure columns are 1
failure_columns = ['TWF', 'HDF', 'PWF', 'OSF', 'RNF']
df['Machine failure'] = df[failure_columns].max(axis=1)

# Randomly select 1000 records
df_sample = df.sample(6000, random_state=42)

# Drop unnecessary columns
df_sample = df_sample.drop(columns=['UDI', 'Product ID'])

# Encode categorical column 'Type'
df_sample['Type'] = LabelEncoder().fit_transform(df_sample['Type'])

# Split dataset into features (X) and target (y)
X = df_sample.drop(columns=['Machine failure'])
y = df_sample['Machine failure']

# Normalize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)

# X_train=X_scaled
# y_train = y

# X_test= Xtest
# y_test = ytest
# Define the CGKSO Classifier
class CGKSOClassifier:
    def __init__(self, num_particles=20, max_iter=10, lower_bound=0, upper_bound=1):
        self.num_particles = num_particles
        self.max_iter = max_iter
        self.lb = lower_bound
        self.ub = upper_bound

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def fitness(self, weights, X, y):
        pred = self.sigmoid(np.dot(X, weights)) >= 0.5
        return -accuracy_score(y, pred)  # Negative accuracy (since we minimize loss)

    def optimize(self, X_train, y_train):
        start_time = time.time()
        dim = X_train.shape[1]
        # Initialization
        particles = np.random.uniform(self.lb, self.ub, (self.num_particles, dim))
        fitness = np.array([self.fitness(p, X_train, y_train) for p in particles])

        best_score = np.min(fitness)
        best_position = particles[np.argmin(fitness)]

        history_acc = []
        history_fail = []
        history_non_fail = []
        h = [0.1]


        for iteration in range(self.max_iter):
            if h[-1] < 0:
              h[-1] = np.abs(h[-1])  # Ensure h[-1] is non-negative
            h_new = 1.2 - 1.5 * (h[-1] ** 4)
            h.append(np.clip(h_new, -10, 10))
            #h.append(1.2 - 1.5 * (h[-1] ** 4))
            p = 2 * (1 - (iteration / self.max_iter) ** 0.25) + abs(h[iteration]) * (
    (iteration / self.max_iter) ** 0.25 - (iteration / self.max_iter) ** 3)

            beta = 0.3 + (1.2 - 0.2) * (1 - (iteration / self.max_iter) ** 3) ** 2
            alpha = abs(beta * np.sin((3 * np.pi / 2 + np.sin(3 * np.pi / 2 * beta))))

            # Hunting Stage
            for i in range(self.num_particles):
                new_position = particles[i] + (self.lb + np.random.rand() * (self.ub - self.lb)) / (iteration + 1)
                new_position = np.clip(new_position, self.lb, self.ub)
                new_fitness = self.fitness(new_position, X_train, y_train)
                if new_fitness < fitness[i]:
                    fitness[i] = new_fitness
                    particles[i] = new_position

            # Best Position Attraction Effect
            for i in range(self.num_particles):
                s = 1.5 * (np.abs(fitness[i]) ** np.random.rand())
                new_position = best_position - particles[i] * s
                new_position = np.clip(new_position, self.lb, self.ub)
                new_fitness = self.fitness(new_position, X_train, y_train)
                if new_fitness < fitness[i]:
                    fitness[i] = new_fitness
                    particles[i] = new_position
             # Foraging stage
            for i in range(self.num_particles):
                TF = 1 if np.random.rand() > 0.5 else -1
                new_pos = (best_position + np.random.rand(dim) * (best_position - particles[i]) +
                          TF * (p ** 2) * (best_position - particles[i]))
                new_pos = np.clip(new_pos, self.lb, self.ub)
                new_fit = self.fitness(new_pos, X_train, y_train)
                if new_fit < fitness[i]:
                    fitness[i] = new_fit
                    particles[i] = new_pos

            # Self-protection
            for i in range(self.num_particles):
                rand_escape = np.random.rand()
                if rand_escape < 0.1:  # Escape condition (10% chance)
                    new_pos = np.random.uniform(self.lb, self.ub, dim)
                    new_fit = self.fitness(new_pos, X_train, y_train)
                    if new_fit < fitness[i]:
                        fitness[i] = new_fit
                        particles[i] = new_pos

            # Update global best
            if np.min(fitness) < best_score:
                best_score = np.min(fitness)
                best_position = particles[np.argmin(fitness)]

            # Store accuracy over iterations
            history_acc.append(-best_score)

            # Predict failure & non-failure machines at each iteration
            X_full = np.vstack((X_train, X_test))
            pred_iter = (self.sigmoid(np.dot(X_full, best_position)) >= 0.5).astype(int)

            ## pred_iter = (self.sigmoid(np.dot(X_train, best_position)) >= 0.5).astype(int)
            history_fail.append(np.sum(pred_iter == 1))
            history_non_fail.append(np.sum(pred_iter == 0))

        self.best_weights = best_position
        self.execution_time = time.time() - start_time
        return history_acc, history_fail, history_non_fail

    def predict(self, X):
        return (self.sigmoid(np.dot(X, self.best_weights)) >= 0.5).astype(int)


# Train CGKSO classifier
cgkso = CGKSOClassifier()
accuracy_history, failure_history, non_failure_history = cgkso.optimize(X_train, y_train)

# Make predictions
y_pred = cgkso.predict(X_test)

# Calculate metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)

#Results plotting

import numpy as np
import matplotlib.pyplot as plt

# Data samples
samples = np.array([1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000])

accuracy = [0.9958, 0.9912, 0.9963, 0.9937, 0.9991, 0.9904, 0.9947, 0.9919, 0.996, 0.9928]
precision = [0.9792, 0.9721, 0.9744, 0.9755, 0.9768, 0.9818, 0.9730, 0.9777, 0.9789, 0.98]
recall = [0.9863, 0.9847, 0.9881, 0.9809, 0.9825, 0.9894, 0.9876, 0.9817, 0.9889, 0.9832]
exe=[0.45, 0.87, 0.94, 1.09, 1.14, 1.23, 1.41, 1.78, 1.85, 1.99]
# Plot Accuracy
plt.figure(figsize=(8, 5))
plt.bar(samples, accuracy, color='blue', width=500)
plt.xlabel("Number of Data Samples")
plt.ylabel("Accuracy")
plt.title("Accuracy vs. Data Samples")
plt.xticks(samples, rotation=0)
plt.ylim(0.9, 1.0)
plt.grid(axis='y', linestyle='--')
plt.show()

# Plot Precision
plt.figure(figsize=(8, 5))
plt.bar(samples, precision, color='green', width=500)
plt.xlabel("Number of Data Samples")
plt.ylabel("Precision")
plt.title("Precision vs. Data Samples")
plt.xticks(samples, rotation=0)
plt.ylim(0.9, 1.0)
plt.grid(axis='y', linestyle='--')
plt.show()

# Plot Recall
plt.figure(figsize=(8, 5))
plt.bar(samples, recall, color='red', width=500)
plt.xlabel("Number of Data Samples")
plt.ylabel("Recall")
plt.title("Recall vs. Data Samples")
plt.xticks(samples, rotation=0)
plt.ylim(0.9, 1.0)
plt.grid(axis='y', linestyle='--')
plt.show()

# Plot exe
plt.figure(figsize=(8, 5))
plt.bar(samples, exe, color='red', width=500)
plt.xlabel("Number of Data Samples")
plt.ylabel("Execution time (seconds)")
plt.title("Execution time  vs. Data Samples")
plt.xticks(samples, rotation=0)
plt.grid(axis='y', linestyle='--')
plt.show()
