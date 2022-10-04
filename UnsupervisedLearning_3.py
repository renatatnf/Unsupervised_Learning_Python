# -*- coding: utf-8 -*-
"""
@author: renata.fernandes
"""
# CHAPTER 3 - Decorrelating your data and dimension reduction

# Perform the necessary imports
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from scipy.sparse import csr_matrix

#Correlated data in nature
grain_df = pd.read_csv('Dados/Grains/seeds.csv', sep=',')

# Assign the 0th column of grains: width
width = grain_df['f5'] 

# Assign the 1st column of grains: length
length = grain_df['f4']

# Scatter plot width vs length
plt.scatter(width, length)
plt.axis('equal')
plt.show()

# Calculate the Pearson correlation
correlation, pvalue = pearsonr(width,length)

# Display the correlation
print(correlation)


#Decorrelating the grain measurements with PCA

# Create PCA instance: model
model = PCA()

# Apply the fit_transform method of model to grains: pca_features
grains = grain_df.to_numpy()[:,[3,4]]
pca_features = model.fit_transform(grains)

# Assign 0th column of pca_features: xs
xs = pca_features[:,0]

# Assign 1st column of pca_features: ys
ys = pca_features[:,1]

# Scatter plot xs vs ys
plt.scatter(xs, ys)
plt.axis('equal')
plt.show()

# Calculate the Pearson correlation of xs and ys
correlation, pvalue = pearsonr(xs, ys)

# Display the correlation
print(correlation)



#The first principal component
# Make a scatter plot of the untransformed points
plt.scatter(grains[:,0], grains[:,1])

# Create a PCA instance: model
model = PCA()

# Fit model to points
model.fit(grains)

# Get the mean of the grain samples: mean
mean = model.mean_

# Get the first principal component: first_pc
first_pc = model.components_[0,:]

# Plot first_pc as an arrow, starting at mean
plt.arrow(mean[0], mean[1], first_pc[0], first_pc[1], color='red', width=0.01)

# Keep axes on same scale
plt.axis('equal')
plt.show()


#Variance of the PCA features

fish_df = pd.read_csv('Dados/fish.csv', sep=',')
fish_array_features =  fish_df.to_numpy()[:,1:]

# Create scaler: scaler
scaler = StandardScaler()

# Create a PCA instance: pca
pca = PCA()

# Create pipeline: pipeline
pipeline = make_pipeline(scaler, pca)

# Fit the pipeline to 'samples'
pipeline.fit(fish_array_features)

# Plot the explained variances
features = range(pca.n_components_)
plt.bar(features, pca.explained_variance_)
plt.xlabel('PCA feature')
plt.ylabel('variance')
plt.xticks(features)
plt.show()



#Dimension reduction of the fish measurements

# Create a PCA model with 2 components: pca
pca = PCA(n_components=2)

# Fit the PCA instance to the scaled samples
scaled_samples = scaler.fit_transform(fish_array_features)
pca.fit(scaled_samples)

# Transform the scaled samples: pca_features
pca_features = pca.transform(scaled_samples)

# Print the shape of pca_features
print(pca_features.shape)



#A tf-idf word-frequency array

# Create a TfidfVectorizer: tfidf
tfidf = TfidfVectorizer() 

# Apply fit_transform to document: csr_mat
documents = ['cats say meow', 'dogs say woof', 'dogs chase cats']
csr_mat = tfidf.fit_transform(documents)

# Print result of toarray() method
print(csr_mat.toarray())

# Get the words: words
words = tfidf.get_feature_names_out()

# Print words
print(words)


#Clustering Wikipedia part I
#TruncatedSVD is able to perform PCA on sparse arrays in csr_matrix format, 
# such as word-frequency arrays.

# Create a TruncatedSVD instance: svd
svd = TruncatedSVD(n_components=50)

# Create a KMeans instance: kmeans
kmeans = KMeans(n_clusters=6)

# Create a pipeline: pipeline
pipeline = make_pipeline(svd,kmeans)


#Clustering Wikipedia part II
articles_df = pd.read_csv('Dados/Wikipedia articles/wikipedia-vectors.csv', sep=',', index_col=0)
articles = csr_matrix(articles_df.transpose())
titles = list(articles_df.columns)

# Fit the pipeline to articles
pipeline.fit(articles)

# Calculate the cluster labels: labels
labels = pipeline.predict(articles)

# Create a DataFrame aligning labels and titles: df
df = pd.DataFrame({'label': labels, 'article': titles})

# Display df sorted by cluster label
print(df.sort_values('label'))

