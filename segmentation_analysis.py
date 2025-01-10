import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier, plot_tree
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
file_path = 'mcdonalds.csv'  # Replace with your dataset path
data = pd.read_csv(file_path)

# Step 1: Data Preprocessing
# Encode categorical variables (Yes/No to 1/0, Gender to numeric)
binary_columns = ['yummy', 'convenient', 'spicy', 'fattening', 'greasy', 'fast',
                  'cheap', 'tasty', 'expensive', 'healthy', 'disgusting']
for col in binary_columns:
    data[col] = data[col].map({'Yes': 1, 'No': 0})

# Encode Gender
data['Gender'] = LabelEncoder().fit_transform(data['Gender'])

# Encode VisitFrequency ordinally
visit_mapping = {
    'Once a week': 5,
    'Every two weeks': 4,
    'Once a month': 3,
    'Every three months': 2,
    'Once a year or less': 1
}
data['VisitFrequency'] = data['VisitFrequency'].map(visit_mapping)

# Drop rows with missing values if any
data.dropna(inplace=True)

# Normalize data for clustering
scaler = StandardScaler()
data_scaled = scaler.fit_transform(data[binary_columns + ['Age', 'VisitFrequency']])

# Step 2: Apply K-means Clustering
kmeans = KMeans(n_clusters=3, random_state=42)
data['Segment'] = kmeans.fit_predict(data_scaled)

# Visualize cluster sizes
sns.countplot(x='Segment', data=data, palette='Set2')
plt.title("Segment Distribution")
plt.xlabel("Segment")
plt.ylabel("Count")
plt.show()

# Summary statistics for each segment
segment_summary = data.groupby('Segment').mean()
print(segment_summary)

# Step 3: Visualizations
# Heatmap of segment characteristics
plt.figure(figsize=(12, 6))
sns.heatmap(segment_summary[binary_columns + ['Age', 'VisitFrequency']], annot=True, fmt=".2f", cmap="coolwarm")
plt.title("Segment Characteristics Heatmap")
plt.xlabel("Attributes")
plt.ylabel("Segment")
plt.show()

# Age distribution by segment
plt.figure(figsize=(10, 5))
sns.boxplot(x='Segment', y='Age', data=data, palette='Set2')
plt.title("Age Distribution by Segment")
plt.xlabel("Segment")
plt.ylabel("Age")
plt.show()

# Visit frequency distribution by segment
plt.figure(figsize=(10, 5))
sns.boxplot(x='Segment', y='VisitFrequency', data=data, palette='Set3')
plt.title("Visit Frequency Distribution by Segment")
plt.xlabel("Segment")
plt.ylabel("Visit Frequency (1: Less Frequent, 5: Most Frequent)")
plt.show()

# Step 4: Predictive Modeling (Decision Tree)
# Define features and target
X = data[binary_columns + ['Age', 'VisitFrequency']]
y = data['Segment']

# Train a decision tree
tree = DecisionTreeClassifier(max_depth=4, random_state=42)
tree.fit(X, y)

# Plot the decision tree
plt.figure(figsize=(16, 10))
plot_tree(tree, feature_names=X.columns, class_names=[str(i) for i in tree.classes_], filled=True, rounded=True)
plt.title("Decision Tree for Segment Prediction")
plt.show()

# Step 5: Customizing Marketing Strategies
# Define strategies for each segment based on characteristics
segment_strategies = {
    0: "Focus on maintaining positive perceptions of taste and value. Promote 'fun and tasty' campaigns to younger, frequent visitors.",
    1: "Address concerns about cost and perceived negatives. Introduce budget-friendly options and emphasize quality improvements for older demographics.",
    2: "Highlight premium offerings and upscale experiences. Target younger audiences seeking high-value items with a focus on luxury and exclusivity."
}

# Display strategies for each segment
for segment, strategy in segment_strategies.items():
    print(f"Segment {segment} Strategy:\n{strategy}\n")
