# -*- coding: utf-8 -*-
"""
@author: renata.fernandes
"""
# CHAPTER 1 - Clustering for dataset exploration

# Import necessary modules
import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, Normalizer


#Clustering 
grain_df = pd.read_csv('Dados/Grains/seeds.csv', sep=',')
grain_df['varieties'] = grain_df['varieties'].replace([1,2,3],["Canadian wheat","Kama wheat", "Rosa wheat"])
                                        
# Create arrays for the features and the response variable
varieties = grain_df['varieties'].values
points = grain_df.drop('varieties', axis=1).values

# Create a KMeans instance with 3 clusters: model
model =KMeans(n_clusters=3)

# Fit model to points
model.fit(points)

# Determine the cluster labels of new_points: labels
labels = model.predict(points)

# Print cluster labels of new_points
print('Cluster labels of new_points: {}'.format(labels))

#Inspect your clustering

# Assign the columns of new_points: xs and ys
xs = points[:,0]
ys = points[:,1]

# Make a scatter plot of xs and ys, using labels to define the colors
plt.scatter(xs,ys,c=labels, alpha=0.5)
# Assign the cluster centers: centroids
centroids = model.cluster_centers_

# Assign the columns of centroids: centroids_x, centroids_y
centroids_x = centroids[:,0]
centroids_y = centroids[:,1]

# Make a scatter plot of centroid
plt.scatter(centroids_x, centroids_y, marker='D', s=50)
plt.title('Seeds Clusters - KMeans')
plt.xlabel('f1')
plt.ylabel('f2')
plt.show()


#How many clusters of grain?
ks = range(1, 6)
inertias = []

for k in ks:
    # Create a KMeans instance with k clusters: model
    model = KMeans(n_clusters=k)
    
    # Fit model to samples
    model.fit(points)
    
    # Append the inertia to the list of inertias
    inertias.append(model.inertia_)
    
# Plot ks vs inertias
plt.plot(ks, inertias, '-o')
plt.title('Seeds Clusters - ks vs inertias')
plt.xlabel('number of clusters, k')
plt.ylabel('inertia')
plt.xticks(ks)
plt.show()



#Evaluating the grain clustering
# Create a KMeans model with 3 clusters: model
model = KMeans(n_clusters=3)

# Use fit_predict to fit model and obtain cluster labels: labels
labels = model.fit_predict(points)

# Create a DataFrame with labels and varieties as columns: df
grain_df_ct = pd.DataFrame({'labels': labels, 'varieties': varieties})

# Create crosstab: ct
ct = pd.crosstab(grain_df_ct['labels'], grain_df_ct['varieties'])

# Display ct
print(ct)


#Scaling fish data for clustering

fish_df = pd.read_csv('Dados/fish.csv', sep=',')

# Create arrays for the features and the response variable
species = fish_df['species'].values
samples = fish_df.drop('species', axis=1).values

# Create scaler: scaler
scaler = StandardScaler()

# Create KMeans instance: kmeans
kmeans = KMeans(n_clusters=4)

# Create pipeline: pipeline
pipeline = make_pipeline(scaler, kmeans)

#Clustering the fish data

# Fit the pipeline to samples
pipeline.fit(samples)

# Calculate the cluster labels: labels
labels = pipeline.predict(samples)

# Create a DataFrame with labels and species as columns: df
fish_df_ct = pd.DataFrame({'labels':labels,'species':species})
ct = pd.crosstab(fish_df_ct['labels'], fish_df_ct['species'])
# Display ct
print(ct)


#Clustering stocks using KMeans
stocks_df = pd.read_csv('Dados/company-stock-movements-2010-2015-incl.csv', sep=',')

# Create arrays for the features and the response variable
stock_companies = stocks_df['company'].values
stock_movements = stocks_df.drop('company', axis=1).values

# Create a normalizer: normalizer
normalizer = Normalizer()

# Create a KMeans model with 10 clusters: kmeans
kmeans = KMeans(n_clusters=10)

# Make a pipeline chaining normalizer and kmeans: pipeline
pipeline = make_pipeline(normalizer,kmeans)


# Fit pipeline to the daily price movements
pipeline.fit(stock_movements)


#Which stocks move together?

# Predict the cluster labels: labels
labels = pipeline.predict(stock_movements)

# Create a DataFrame aligning labels and companies: df
df = pd.DataFrame({'labels': labels, 'companies': stock_companies})

# Display df sorted by cluster label
print(df.sort_values('labels'))

