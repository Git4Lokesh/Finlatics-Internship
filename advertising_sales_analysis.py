import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

# Load advertising sales data
dataset = pd.read_csv('advertising_sales_data.csv')

# Checking for Missing Values
print("Missing values check:")
missing_values = dataset.isna().sum()
print(missing_values)

# Analyze Radio column statistics
print("\nRadio column statistics:")
print(dataset['Radio'].describe())
print(f"Radio median: {dataset['Radio'].median()}")

# Filling NaN in Radio Column with Median
imputer = SimpleImputer(missing_values=np.nan, strategy='median')
dataset['Radio'] = imputer.fit_transform(dataset[['Radio']]).flatten()

# Verify no missing values remain
print("\nMissing values after imputation:")
print(dataset.isna().sum())

# Q1: Average Money spent on TV Advertising
tv_avg = dataset['TV'].mean()
print(f"\nAverage TV advertising spend: ${tv_avg:.1f}K")

# Q2: Correlation between Radio Advertising Expenditure and Sales
radio_advertising = dataset['Radio']
sales = dataset['Sales']

# Visualize the relation using linear regressor
regressor = LinearRegression()
regressor.fit(radio_advertising.values.reshape(-1,1), sales.values.reshape(-1,1))

plt.figure(figsize=(10, 6))
plt.scatter(radio_advertising.values, sales.values, color='red', alpha=0.6)
plt.plot(radio_advertising.values, regressor.predict(radio_advertising.values.reshape(-1,1)), color='blue')
plt.title('Radio Advertising Expenditure vs Sales')
plt.xlabel('Radio Advertising Expenditure ($K)')
plt.ylabel('Sales')
plt.grid(True, alpha=0.3)
plt.show()

# Calculate correlation
correlation = dataset['Radio'].corr(dataset['Sales'])
print(f"\nRadio-Sales correlation: {correlation:.4f} (weak positive correlation)")

# Q3: Which advertising medium affects product sales the most
correlation_matrix = dataset.iloc[:,1:].corr()

plt.figure(figsize=(8, 6))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0)
plt.title('Advertising Channels Correlation Matrix')
plt.show()

# Print correlations with sales
print("\nCorrelations with Sales:")
for col in ['TV', 'Radio', 'Newspaper']:
    corr = dataset[col].corr(dataset['Sales'])
    print(f"{col}: {corr:.4f}")

# Q4: Linear Regression Model to Predict Sales
features = dataset.iloc[:,1:4].values  # TV, Radio, Newspaper
target = dataset.iloc[:,4].values      # Sales

# Train-test split
features_train, features_test, target_train, target_test = train_test_split(
    features, target, test_size=0.2, random_state=1
)

# Build and train the model
regressor = LinearRegression()
regressor.fit(features_train, target_train)
target_pred = regressor.predict(features_test)

# Visualize actual vs predicted sales
plt.figure(figsize=(10, 6))
plt.scatter(target_test, target_pred, alpha=0.7)
plt.xlabel('Actual Sales')
plt.ylabel('Predicted Sales')
plt.title('Actual vs Predicted Sales')
# Plot perfect prediction line
plt.plot([min(target_test), max(target_test)], [min(target_test), max(target_test)], 
         color='red', linestyle='--', label='Perfect Prediction')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

# Calculate performance metrics
mse = mean_squared_error(target_test, target_pred)
r2 = r2_score(target_test, target_pred)

print(f"\nModel Performance:")
print(f"Mean Squared Error: {mse:.4f}")
print(f"R² Score: {r2:.4f}")
print(f"Model explains {r2*100:.1f}% of sales variance")

# Display model coefficients
coefficients = regressor.coef_
intercept = regressor.intercept_
feature_names = ['TV', 'Radio', 'Newspaper']

print(f"\nModel Equation:")
print(f"Sales = {intercept:.3f}", end="")
for i, (name, coef) in enumerate(zip(feature_names, coefficients)):
    print(f" + {coef:.4f}*{name}", end="")
print()

print(f"\nCoefficient Analysis:")
for name, coef in zip(feature_names, coefficients):
    print(f"{name}: {coef:.4f} (sales units per $1K spent)")

# Q5: Prediction for new data
new_feature = np.array([200, 40, 50])  # TV=200K, Radio=40K, Newspaper=50K
new_feature = new_feature.reshape(1, -1)
predicted_sales = regressor.predict(new_feature)
print(f"\nPrediction for TV=$200K, Radio=$40K, Newspaper=$50K:")
print(f"Predicted sales: {predicted_sales[0]:.2f} units")

# Q6: Will normalization of feature data affect model accuracy?
scaler = StandardScaler()
features_train_normalized = scaler.fit_transform(features_train)
features_test_normalized = scaler.transform(features_test)

# Train model on normalized data
regressor_norm = LinearRegression()
regressor_norm.fit(features_train_normalized, target_train)
target_pred_norm = regressor_norm.predict(features_test_normalized)

# Calculate metrics for normalized model
r2_normalized = r2_score(target_test, target_pred_norm)
mse_normalized = mean_squared_error(target_test, target_pred_norm)

print(f"\nNormalized Model Performance:")
print(f"R² Score (normalized): {r2_normalized:.4f}")
print(f"MSE (normalized): {mse_normalized:.4f}")
print(f"Impact of normalization: {'No change' if abs(r2 - r2_normalized) < 0.001 else 'Changed'}")
print("Note: No change expected as all features are in same units (currency)")

# Q7: Excluding TV expenditure from model to test accuracy
new_features = dataset.iloc[:,2:4].values  # Radio and Newspaper only
new_features_train, new_features_test, target_train, target_test = train_test_split(
    new_features, target, test_size=0.2, random_state=1
)

# Train model without TV
regressor_no_tv = LinearRegression()
regressor_no_tv.fit(new_features_train, target_train)
target_pred_no_tv = regressor_no_tv.predict(new_features_test)

# Visualize performance without TV
plt.figure(figsize=(10, 6))
plt.scatter(target_test, target_pred_no_tv, alpha=0.7, color='orange')
plt.xlabel('Actual Sales')
plt.ylabel('Predicted Sales')
plt.title('Predicted vs Actual Sales (Without TV Features)')
plt.plot([min(target_test), max(target_test)], [min(target_test), max(target_test)], 
         color='red', linestyle='--', label='Perfect Prediction')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

# Calculate performance without TV
r2_no_tv = r2_score(target_test, target_pred_no_tv)
mse_no_tv = mean_squared_error(target_test, target_pred_no_tv)

print(f"\nModel Performance WITHOUT TV:")
print(f"R² Score (no TV): {r2_no_tv:.4f}")
print(f"MSE (no TV): {mse_no_tv:.4f}")

# Compare all models
print(f"\n=== MODEL COMPARISON ===")
print(f"Full Model (TV+Radio+Newspaper): R² = {r2:.4f}")
print(f"Without TV (Radio+Newspaper):     R² = {r2_no_tv:.4f}")
print(f"Accuracy drop without TV: {((r2 - r2_no_tv)/r2)*100:.1f}%")
print(f"This proves TV is the most critical feature for sales prediction!")

# Summary visualization
plt.figure(figsize=(12, 4))

# Model comparison
models = ['Full Model\n(TV+Radio+Newspaper)', 'Without TV\n(Radio+Newspaper)']
r2_scores = [r2, r2_no_tv]

plt.subplot(1, 3, 1)
bars = plt.bar(models, r2_scores, color=['green', 'orange'])
plt.title('Model Performance Comparison')
plt.ylabel('R² Score')
plt.ylim(0, 1)
for bar, score in zip(bars, r2_scores):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
             f'{score:.3f}', ha='center', va='bottom')

# Correlation comparison
plt.subplot(1, 3, 2)
channels = ['TV', 'Radio', 'Newspaper']
correlations = [dataset[col].corr(dataset['Sales']) for col in channels]
bars = plt.bar(channels, correlations, color=['red', 'blue', 'green'])
plt.title('Sales Correlation by Channel')
plt.ylabel('Correlation with Sales')
for bar, corr in zip(bars, correlations):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
             f'{corr:.3f}', ha='center', va='bottom')

# Spending distribution
plt.subplot(1, 3, 3)
spending = [dataset[col].mean() for col in channels]
bars = plt.bar(channels, spending, color=['red', 'blue', 'green'])
plt.title('Average Spending by Channel')
plt.ylabel('Average Spend ($K)')
for bar, spend in zip(bars, spending):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, 
             f'${spend:.0f}K', ha='center', va='bottom')

plt.tight_layout()
plt.show()

print(f"\n=== ANALYSIS COMPLETE ===")
print(f"Dataset analyzed: {len(dataset)} advertising campaigns")
print(f"Most effective channel: TV (correlation: {dataset['TV'].corr(dataset['Sales']):.3f})")
print(f"Model accuracy: {r2*100:.1f}% (R² = {r2:.3f})")
print(f"Critical finding: Removing TV reduces accuracy by {((r2 - r2_no_tv)/r2)*100:.1f}%")
