# Import libraries
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import datetime
import seaborn as sns
import sklearn
import plotly.express as px
from plotly import tools
from plotly.subplots import make_subplots
import plotly.figure_factory as ff
import plotly.graph_objs as go
from matplotlib import colors
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering
from sklearn import metrics


df = pd.read_csv('/Users/amirahnurazman/Desktop/Master/SEM_2:23/WQD 7005/Assignment/Dataset/1. marketing_campaign.csv')
print(df)
print(df.shape)
print(df.info())
print(df.columns)
print(df.describe())

# Check missing values
total_null = df.isnull().sum()
percent_null = (total_null/df.isnull().count())*100
print(total_null, percent_null)

# Drop missing values in Income
df_clean = df.dropna(subset=['Income'])
print(df_clean)

# Check for duplicated values
df_clean = df_clean.drop_duplicates()

# Drop unwanted attributes
df_clean = df_clean.drop(columns=['Z_CostContact', 'Z_Revenue'])

# Change Dt_Customer to datetime
df_clean['Dt_Customer'] = pd.to_datetime(df_clean['Dt_Customer'], dayfirst=True)
# Set display options to show all columns
# pd.set_option('display.max_columns', None)
print(df_clean.describe())

# Univariate Analysis
## Categorical Data
cat_col=df_clean[['Year_Birth', 'Education', 'Marital_Status', 'Kidhome', 'Teenhome',
                  'AcceptedCmp1', 'AcceptedCmp2', 'AcceptedCmp3', 'AcceptedCmp4', 'AcceptedCmp5',
                  'Complain', 'Response']]

def unique_val(column):
    unique_values = df_clean[column].value_counts()
    print(f"\nThere are {len(unique_values)} unique values in {column}: \n{unique_values}")
    return unique_values

for column in cat_col.columns:
    unique_val(column)

a = 4  # number of rows
b = 3  # number of columns
c = 1  # initialize plot counter

fig = plt.figure(figsize=(15, 8))

for i in cat_col:
    plt.subplot(a, b, c)
    plt.title('{}'.format(i))
    sns.countplot(data=df_clean, x=i)
    c = c + 1

plt.tight_layout()
plt.show()

## Numerical Data
numerical_col=df_clean[['Income', 'Recency', 'MntWines', 'MntFruits', 'MntMeatProducts', 'MntFishProducts',
                       'MntSweetProducts', 'MntGoldProds', 'NumDealsPurchases', 'NumWebPurchases',
                        'NumCatalogPurchases', 'NumStorePurchases', 'NumWebVisitsMonth']]

total_numerical_col = len(numerical_col.columns)
numerical_num_cols = 3
numerical_num_rows = (total_numerical_col + numerical_num_cols - 1) // numerical_num_cols

fig = plt.figure(figsize=(15, 8))

for i, col in enumerate(numerical_col.columns):
    plt.subplot(numerical_num_rows, numerical_num_cols, i + 1)
    plt.hist(numerical_col[col])
    plt.title(col, fontsize=10)
    plt.xticks(rotation=90)

plt.tight_layout()
plt.show()

fig = plt.figure(figsize=(15, 8))

for i, col in enumerate(numerical_col.columns):
    plt.subplot(numerical_num_rows, numerical_num_cols, i + 1)
    plt.boxplot(numerical_col[col])
    plt.title(col, fontsize=10)
    plt.xticks(rotation=90)

plt.tight_layout()
plt.show()

# Data Consistency
def marital_status(status):
    if 'Married' in status or 'Together' in status:
        return 'Partner'
    else:
        return 'Single'

df_clean['Marital_Status'] = df_clean['Marital_Status'].apply(marital_status)

def education_status(education):
    if 'Basic' in education:
        return 'School'
    elif 'Graduation' in education or '2n Cycle' in education:
        return 'Undergraduate'
    else:
        return 'Postgraduate'


df_clean['Education'] = df_clean['Education'].apply(education_status)

# Feature Engineering
df_clean['Age'] = 2024 - df_clean['Year_Birth']

import datetime
df_clean['Dt_Customer'] = df_clean['Dt_Customer'].dt.date

df_clean['DaysJoin'] = (datetime.date.today() - df_clean['Dt_Customer']).apply(lambda x: x.days)

cmp_cols = ['AcceptedCmp1', 'AcceptedCmp2', 'AcceptedCmp3', 'AcceptedCmp4', 'AcceptedCmp5']
df_clean['TotalAcceptedCmp'] = df_clean[cmp_cols].sum(axis=1)

expense_cols = ['MntWines', 'MntFruits', 'MntMeatProducts',
                'MntFishProducts', 'MntSweetProducts', 'MntGoldProds']
df_clean['TotalExpense'] = df_clean[expense_cols].sum(axis=1)

df_clean['Marital_Status_Num'] = df_clean['Marital_Status'].map({'Partner': 2, 'Single': 1})

df_clean['TotalKids'] = df_clean['Kidhome'] + df_clean['Teenhome']

df_clean['FamSize'] = df_clean['Marital_Status_Num'] + df_clean['TotalKids']

col_delete = ['ID', 'Year_Birth', 'Dt_Customer', 'Marital_Status', 'Kidhome', 'Teenhome']
df_clean = df_clean.drop(col_delete, axis=1)

pd.set_option('display.max_columns', None)
print(df_clean.describe())

# Remove incorrect value in Age
df_clean = df_clean[df_clean['Age'] <= 100]
print(df_clean.describe())
print(df_clean.info())

## Update numerical column
new_numerical = df_clean[['Age', 'DaysJoin', 'TotalAcceptedCmp', 'TotalExpense', 'TotalKids', 'FamSize']]
new_numerical_col = pd.concat((numerical_col, new_numerical), axis=1, join='inner')
print(new_numerical_col.shape)

total_new_numerical_col = len(new_numerical_col.columns)
new_numerical_num_cols = 3
new_numerical_num_rows = (total_new_numerical_col + new_numerical_num_cols - 1) // new_numerical_num_cols

fig = plt.figure(figsize=(15, 12))

for i, col in enumerate(new_numerical_col.columns):
    plt.subplot(new_numerical_num_rows,new_numerical_num_cols, i + 1)
    plt.hist(new_numerical_col[col])
    plt.title(col, fontsize=10)

plt.tight_layout()
plt.show()

fig = plt.figure(figsize=(15, 12))

for i, col in enumerate(new_numerical_col.columns):
    plt.subplot(new_numerical_num_rows, new_numerical_num_cols, i + 1)
    plt.boxplot(new_numerical_col[col])
    plt.title(col, fontsize=10)

plt.tight_layout()
plt.show()

## Update categorical column
new_cat_col = df_clean.drop(columns=new_numerical_col.columns)

a = 3  # number of rows
b = 3  # number of columns
c = 1  # initialize plot counter

fig = plt.figure(figsize=(15, 8))

for i in new_cat_col:
    plt.subplot(a, b, c)
    plt.title('{}'.format(i))
    sns.countplot(data=df_clean, x=i)
    c = c + 1

plt.tight_layout()
plt.show()

# Correlation Matrix
accept = df_clean[[col for col in df_clean.columns if col != 'Education']]

corr_num = accept.corr()
plt.figure(figsize=(15, 9))
sns.heatmap(corr_num, annot=True, fmt=".2f", cmap='coolwarm', linewidths=0.5)
plt.title('Correlation Matrix Heatmap with Accepts', fontsize=16)
plt.show()

# Remove highly correlated features
high_corr = ['MntWines', 'MntFruits', 'MntMeatProducts', 'MntFishProducts', 'MntSweetProducts',
             'MntGoldProds', 'AcceptedCmp1', 'AcceptedCmp2', 'AcceptedCmp3', 'AcceptedCmp4',
             'AcceptedCmp5', 'TotalKids', 'Marital_Status_Num']

data = df_clean.drop(high_corr, axis=1)
print(data.head())
print(data.describe())
print(data.info())

# Bivariate Analysis
# Pair plot
data_pair = data.select_dtypes(exclude='object')

import plotly.express as px
fig = px.scatter_matrix(data_pair)
fig.show()

# Remove obvious outliers
high_outliers = data[(data['Income'] >= 666666) |
                     (data['NumWebPurchases'] > 20) |
                     (data['NumCatalogPurchases'] > 20)]
print(high_outliers)

data = data.drop(high_outliers.index)
print(data.describe())

# Distribution after remove obvious outliers
total_col = len(data.select_dtypes(exclude='object').columns)
cols = 3
rows = (total_col + cols - 1) // cols

fig = plt.figure(figsize=(15, 12))

for i, col in enumerate(data.select_dtypes(exclude='object').columns):
    plt.subplot(rows, cols, i + 1)
    plt.hist(data[col])
    plt.title(col, fontsize=10)

plt.tight_layout()
plt.show()

fig = plt.figure(figsize=(15, 12))

for i, col in enumerate(data.select_dtypes(exclude='object').columns):
    plt.subplot(rows, cols, i + 1)
    plt.boxplot(data[col])
    plt.title(col, fontsize=10)

plt.tight_layout()
plt.show()

fig = px.scatter_matrix(data.select_dtypes(exclude='object'))
fig.show()

# Function to detect outliers using IQR method
def detect_outliers(data, col):
    Q1 = data[col].quantile(0.25)
    Q3 = data[col].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    outliers = data[(data[col] < lower_bound) | (data[col] > upper_bound)]
    return outliers

# Dictionary to store outliers
outliers = {}

for col in data.select_dtypes(exclude='object').columns:
    outliers[col] = detect_outliers(data, col)

for col, outlier_data in outliers.items():
    print(f"Outliers in column '{col}': {len(outlier_data)}")

# Feature Scaling OneHotEncoder with RobustScaler
df_transform = data.copy()

binary_cols = ['Complain', 'Response']
data_cat_cols = df_transform.select_dtypes(include='object').columns.tolist()
data_num_cols = df_transform.select_dtypes(exclude='object').columns.difference(binary_cols).tolist()

from sklearn.preprocessing import LabelEncoder, OneHotEncoder, RobustScaler, StandardScaler
from sklearn.compose import ColumnTransformer

label_encoder = LabelEncoder()
for col in data_cat_cols:
    df_transform[col] = label_encoder.fit_transform(df_transform[col])

transformer = ColumnTransformer(transformers=[
    ('num', RobustScaler(), data_num_cols)
])

df_transformed = transformer.fit_transform(df_transform)

df_transformed = pd.DataFrame(df_transformed, columns=data_num_cols)
# Add the binary and encoded categorical columns back
df_final = pd.concat([df_transformed.reset_index(drop=True), df_transform[binary_cols + data_cat_cols].reset_index(drop=True)], axis=1)

print(df_final.head())
print(df_final.info())
print(df_final.describe())
print(df_final.isnull().sum())

# Distribution after scaled
total_col = len(df_final.select_dtypes(exclude='object').columns)
cols = 3
rows = (total_col + cols - 1) // cols

fig = plt.figure(figsize=(15, 12))

for i, col in enumerate(df_final.select_dtypes(exclude='object').columns):
    plt.subplot(rows, cols, i + 1)
    plt.hist(df_final[col])
    plt.title(col, fontsize=10)

plt.tight_layout()
plt.show()

fig = plt.figure(figsize=(15, 12))

for i, col in enumerate(df_final.select_dtypes(exclude='object').columns):
    plt.subplot(rows, cols, i + 1)
    plt.boxplot(df_final[col])
    plt.title(col, fontsize=10)

plt.tight_layout()
plt.show()

# Dictionary to store outliers in the scaled data
outliers_scaled = {}

for col in df_final.select_dtypes(exclude='object').columns:
    outliers_scaled[col] = detect_outliers(df_final, col)

for col, outlier_data in outliers_scaled.items():
    print(f"Outliers in scaled column '{col}': {len(outlier_data)}")

# KMeans Clustering
# Elbow method to find optimal number of clusters with Label Encoded data

inertia_le = []

for k in range(2, 11):
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=20)
    kmeans.fit(df_final)
    inertia_le.append(kmeans.inertia_)

# Plotting the Elbow graph
from kneed import KneeLocator

# Determine the best_k automatically using KneeLocator
kneedle_le = KneeLocator(range(2, 11), inertia_le, curve='convex', direction='decreasing')
bestK = kneedle_le.elbow

plt.plot(range(2, 11), inertia_le, marker='o')
plt.axvline(x=bestK, color='r', linestyle='--', label=f'Optimal k = {bestK}')
plt.title('Elbow method (Label Encoded Data)')
plt.xlabel('Number of clusters')
plt.ylabel('Inertia')
plt.show()

# Visualized KMeans
kmeans = KMeans(n_clusters=5, random_state=42, n_init=20)
kmeans.fit(df_final)
df_final['Cluster'] = kmeans.labels_

# Validation
from sklearn.metrics import silhouette_score

# Silhouette Analysis
silhouettes = []
ks_silhouette = list(range(2, 12))

for n_cluster in ks_silhouette:
    kmeans = KMeans(n_clusters=n_cluster, random_state=42, n_init=20)
    kmeans.fit(df_final)
    labels = kmeans.labels_
    sil_coeff = silhouette_score(df_final, labels, metric='euclidean')
    print(f"For n_clusters={n_cluster}, The Silhouette Coefficient is {sil_coeff:.4f}")
    silhouettes.append(sil_coeff)

# Plotting Silhouette Scores
plt.subplot(212)
plt.scatter(ks_silhouette, silhouettes, marker='x', color='red', label='Silhouette Coefficients')
plt.plot(ks_silhouette, silhouettes, label='Silhouette Score Trend')
plt.title('Silhouette Analysis (Label Encoded Data)')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('Silhouette Score')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()


# PCA to reduce dimensions to 3 for plotting
pca = PCA(n_components=2)
pca_result = pca.fit_transform(df_final.drop('Cluster', axis=1))

# Plot the clusters
plt.figure(figsize=(10, 8))
plt.scatter(pca_result[:, 0], pca_result[:, 1], c=df_final['Cluster'], cmap='viridis', marker='o')
plt.title('KMeans Clustering Visualization (k=5)')
plt.xlabel('PCA Component 1')
plt.ylabel('PCA Component 2')

plt.colorbar(label='Cluster Label')
plt.show()


# Clustering Analysis
df_transform['Cluster'] = df_final['Cluster'].values

# Distribution of clusters
plt.figure(figsize=(15, 8))
ax = sns.countplot(x=df_transform['Cluster'])
ax.bar_label(ax.containers[0], fontsize=10)
plt.title("Distribution of the Clusters", fontsize=16)
plt.xlabel("Cluster", fontsize=14)
plt.ylabel("Count", fontsize=14)
plt.show()

# Income vs. TotalExpense by Cluster
plt.figure(figsize=(15, 8))
sns.scatterplot(data=df_transform, x='Income', y='TotalExpense', hue='Cluster', palette='viridis', s=100)
plt.title('Income vs. TotalExpense by Cluster', fontsize=16)
plt.xlabel('Income', fontsize=14)
plt.ylabel('TotalExpense', fontsize=14)
plt.legend(title='Cluster')
plt.show()

# TotalExpense by Cluster
plt.figure(figsize=(15, 8))
sns.boxplot(data=df_transform, x='Cluster', y='TotalExpense', palette='viridis')
plt.title('TotalExpense by Cluster', fontsize=16)
plt.xlabel('Cluster', fontsize=14)
plt.ylabel('TotalExpense', fontsize=14)
plt.show()

# Average TotalExpense by Cluster
plt.figure(figsize=(15, 8))
ax = sns.barplot(data=df_transform, x='Cluster', y='TotalExpense', palette='viridis', estimator=np.mean, ci=None)
for container in ax.containers:
    ax.bar_label(container, label_type='edge', fontsize=12, padding=6)

plt.title('Average TotalExpense by Cluster', fontsize=16)
plt.xlabel('Cluster', fontsize=14)
plt.ylabel('Average TotalExpense', fontsize=14)
plt.show()

# Average TotalAcceptedCmp by Cluster
plt.figure(figsize=(15, 8))
ax = sns.barplot(data=df_transform, x='Cluster', y='TotalAcceptedCmp', palette='viridis', estimator=np.mean, ci=None)
for container in ax.containers:
    ax.bar_label(container, label_type='edge', fontsize=12, padding=6)

plt.title('Average TotalAcceptedCmp by Cluster', fontsize=16)
plt.xlabel('Cluster', fontsize=14)
plt.ylabel('Average TotalAcceptedCmp', fontsize=14)
plt.show()

# FamSize by Cluster
plt.figure(figsize=(15, 8))
ax = sns.barplot(data=df_transform, x='Cluster', y='FamSize', palette='viridis', ci=None)
for container in ax.containers:
    ax.bar_label(container, label_type='edge', fontsize=12, padding=6)

plt.title('Average FamSize by Cluster', fontsize=16)
plt.xlabel('Cluster', fontsize=14)
plt.ylabel('Average FamSize', fontsize=14)
plt.show()

# TotalExpense vs. FamSize by Cluster
plt.figure(figsize=(15, 8))
ax = sns.barplot(data=df_transform, x='FamSize', y='TotalExpense', hue='Cluster',
                 palette='viridis', dodge=True, estimator=np.sum, ci=None)

for container in ax.containers:
    ax.bar_label(container, label_type='edge', fontsize=8, padding=6)

plt.title('TotalExpense vs. FamSize by Cluster', fontsize=16)
plt.xlabel('FamSize', fontsize=14)
plt.ylabel('TotalExpense', fontsize=14)
plt.legend(title='Cluster')
plt.show()

# Age Distribution by Cluster
plt.figure(figsize=(15, 8))
sns.boxplot(data=df_transform, x='Cluster', y='Age', palette='viridis')
plt.title('Age Distribution by Cluster', fontsize=16)
plt.xlabel('Cluster', fontsize=14)
plt.ylabel('Age', fontsize=14)
plt.show()

# Average NumWebVisitsMonth by Cluster
avg_visits = df_transform.groupby('Cluster')['NumWebVisitsMonth'].mean().reset_index()
avg_visits.columns = ['Cluster', 'AvgWebVisits']

plt.figure(figsize=(15, 8))
ax = sns.barplot(data=avg_visits, x='Cluster', y='AvgWebVisits', hue='Cluster', palette='viridis',
                 dodge=True, estimator=np.sum, ci=None)
for container in ax.containers:
    ax.bar_label(container, label_type='edge', fontsize=12, padding=6)

plt.title('Average Web Visits per Cluster', fontsize=16)
plt.xlabel('Cluster', fontsize=14)
plt.ylabel('AvgWebVisits', fontsize=14)
plt.legend(title='Cluster')
plt.show()

# NumWebVisitsMonth vs. TotalExpense by Cluster
plt.figure(figsize=(15, 8))
sns.barplot(data=df_transform, x='NumWebVisitsMonth', y='TotalExpense', hue='Cluster',
                 palette='viridis', dodge=True, estimator=np.sum, ci=None)

plt.title('NumWebVisitsMonth vs. TotalExpense by Cluster', fontsize=16)
plt.xlabel('NumWebVisitsMonth', fontsize=14)
plt.ylabel('TotalExpense', fontsize=14)
plt.legend(title='Cluster')
plt.show()

# Recency by Cluster
plt.figure(figsize=(15, 8))
ax = sns.barplot(data=df_transform, x='Cluster', y='Recency', palette='viridis', estimator=np.mean, ci=None)
for container in ax.containers:
    ax.bar_label(container, label_type='edge', fontsize=12, padding=6)

plt.title('Average Recency by Cluster', fontsize=16)
plt.xlabel('Cluster', fontsize=14)
plt.ylabel('Average Recency (days)', fontsize=14)
plt.show()

# Number of Purchases by Cluster
purchase_cols = ['NumWebPurchases', 'NumStorePurchases', 'NumCatalogPurchases']
df_melted = df_transform.melt(id_vars=['Cluster', 'TotalExpense'],
                              value_vars=purchase_cols, var_name='PurchaseChannel', value_name='ChannelCount')

plt.figure(figsize=(15, 8))
ax = sns.barplot(data=df_melted, x='Cluster', y='ChannelCount', hue='PurchaseChannel', palette='viridis', ci=None)

for container in ax.containers:
    ax.bar_label(container, fmt='%.2f', label_type='edge', fontsize=12, padding=6)

plt.title('Number of Purchases by Cluster and Channel', fontsize=16)
plt.xlabel('Cluster', fontsize=14)
plt.ylabel('Average Number of Purchases', fontsize=14)
plt.legend(title='Purchase Channel')
plt.show()






