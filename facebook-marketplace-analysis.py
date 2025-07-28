import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# Load Facebook Marketplace data
dataset = pd.read_csv('Facebook_Marketplace_data.csv')

# Data preprocessing - check standard deviations
std_list = dataset.iloc[:,3:].std()
print("Standard Deviations:")
print(std_list)

# Substituting Missing Values
x = dataset.iloc[:,3:].values
imputer = SimpleImputer(missing_values=np.nan, strategy='median') # using median due to high standard deviation
imputer.fit(x)
x = imputer.transform(x)
print("\nImputed data shape:", x.shape)

# Q1: Correlate Time of Publishing and Number of Reactions
dataset['status_published'] = pd.to_datetime(dataset['status_published'])
dataset['status_published'] = dataset['status_published'].dt.hour # Extract only the hour
print("\nExtracted hours from timestamp:")
print(dataset['status_published'].head())

# Correlation analysis between time and reactions
a = dataset.iloc[:,2:4]
correlation_matrix = a.corr()

# Visualize correlation
plt.figure(figsize=(8, 6))
sns.heatmap(correlation_matrix, annot=True)
plt.title('Correlation: Time vs Reactions')
plt.show()

# Analyze hourly reactions
hourly_reactions = dataset.groupby('status_published')['num_reactions'].mean().sort_values(ascending=False)
print("\nHourly Reactions Analysis:")
print(hourly_reactions)

# Q2: Correlation between Reactions and Comments/Shares
reactions = dataset.iloc[:,3].values
comments = dataset.iloc[:,4].values
shares = dataset.iloc[:,5].values

# Build regressor for reactions vs comments relation
regressor = LinearRegression()
reactions_reshaped = reactions.reshape(-1,1)
comments_reshaped = comments.reshape(-1,1)
shares_reshaped = shares.reshape(-1,1)
regressor.fit(reactions_reshaped, comments_reshaped)

# Visualize reactions vs comments
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.scatter(reactions, comments, color='red', alpha=0.6)
plt.plot(reactions, regressor.predict(reactions_reshaped), color='blue')
plt.xlabel('Reactions')
plt.ylabel('Comments')
plt.title('Reactions vs Comments Relation')

# Build regressor for reactions vs shares relation
regressor1 = LinearRegression()
regressor1.fit(reactions_reshaped, shares_reshaped)

# Visualize reactions vs shares
plt.subplot(1, 2, 2)
plt.scatter(reactions, shares, alpha=0.6)
plt.plot(reactions, regressor1.predict(reactions_reshaped), color='red')
plt.xlabel('Reactions')
plt.ylabel('Shares')
plt.title('Reactions vs Shares Relation')
plt.tight_layout()
plt.show()

# Overall correlation matrix
z = dataset.iloc[:,3:]
correlation_matrix = z.corr()

plt.figure(figsize=(12, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
plt.title('Overall Correlation Matrix')
plt.show()

# Q3: K-Means Clustering
# Prepare data for clustering
newdataset = dataset.drop(['status_published', 'Column1', 'Column2', 'Column3', 'Column4','status_id'], axis=1)
x = newdataset.iloc[:,:].values

# One Hot Encoding for Categorical Data
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [0])], remainder='passthrough')
x = np.array(ct.fit_transform(x))

# Q4: Finding Optimum Number of Clusters by Elbow Method
wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, init='k-means++', random_state=42)
    kmeans.fit(x)
    wcss.append(kmeans.inertia_)

# Plot elbow curve
plt.figure(figsize=(10, 6))
plt.plot(range(1, 11), wcss, marker='o')
plt.title('The Elbow Method')
plt.xlabel('Number of Clusters')
plt.ylabel('WCSS')
plt.grid(True)
plt.show()

# Apply K-means with optimal clusters (3 clusters based on elbow method)
kmeans = KMeans(n_clusters=3, init='k-means++', random_state=42)
y_kmeans = kmeans.fit_predict(x)

# Using Silhouette Score to Calculate Accuracy
silhouette_avg = silhouette_score(x, y_kmeans)
print(f"\nThe average silhouette score is: {silhouette_avg}")

# Q5: Different type of posts count
post_type_counts = dataset['status_type'].value_counts()
print("\nPost Type Distribution:")
print(post_type_counts)

# Q6: Averages of Reactions, Comments, Shares by Post Type
average_metrics = dataset.groupby('status_type')[['num_reactions', 'num_comments', 'num_shares']].mean()
print("\nAverage Metrics by Post Type:")
print(average_metrics)

# Visualization of post type performance
plt.figure(figsize=(12, 8))

# Plot 1: Post type distribution
plt.subplot(2, 2, 1)
post_type_counts.plot(kind='bar')
plt.title('Post Type Distribution')
plt.xlabel('Post Type')
plt.ylabel('Count')
plt.xticks(rotation=45)

# Plot 2: Average reactions by post type
plt.subplot(2, 2, 2)
average_metrics['num_reactions'].plot(kind='bar', color='orange')
plt.title('Average Reactions by Post Type')
plt.xlabel('Post Type')
plt.ylabel('Average Reactions')
plt.xticks(rotation=45)

# Plot 3: Average comments by post type
plt.subplot(2, 2, 3)
average_metrics['num_comments'].plot(kind='bar', color='green')
plt.title('Average Comments by Post Type')
plt.xlabel('Post Type')
plt.ylabel('Average Comments')
plt.xticks(rotation=45)

# Plot 4: Average shares by post type
plt.subplot(2, 2, 4)
average_metrics['num_shares'].plot(kind='bar', color='red')
plt.title('Average Shares by Post Type')
plt.xlabel('Post Type')
plt.ylabel('Average Shares')
plt.xticks(rotation=45)

plt.tight_layout()
plt.show()

print("\n=== ANALYSIS COMPLETE ===")
print(f"Dataset analyzed: {len(dataset)} Facebook Marketplace posts")
print(f"Peak engagement time: {hourly_reactions.index[0]}:00 hours")
print(f"Best performing post type: {average_metrics['num_reactions'].idxmax()}")
print(f"Clustering silhouette score: {silhouette_avg:.3f}")
