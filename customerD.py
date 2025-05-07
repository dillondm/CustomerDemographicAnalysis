import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

df = pd.read_csv('Mall_Customers.csv')
df.drop_duplicates(inplace=True)
print(df.head())
print(df.info())
print(df.isnull().sum())

sns.countplot(x='Genre', data=df)
plt.title("Gender Distribution")
plt.show()

sns.histplot(df['Age'], bins=10, kde=True)
plt.title('Age Distribution')
plt.show()

sns.scatterplot(x='Annual Income (k$)', y='Spending Score (1-100)', hue='Genre', data=df)
plt.title('Income vs Spending Score')
plt.show()

sns.boxplot(x='Genre', y='Spending Score (1-100)', data=df)
plt.title('Spending Score by Gender')
plt.show()



X = df[['Annual Income (k$)', 'Spending Score (1-100)']]
kmeans = KMeans(n_clusters=5, random_state=0)
df['Segment'] = kmeans.fit_predict(X)

sns.scatterplot(x='Annual Income (k$)', y='Spending Score (1-100)', hue='Segment', palette='Set1', data=df)
plt.title('Customer Segments')
plt.show()

