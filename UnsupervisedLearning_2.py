# -*- coding: utf-8 -*-
"""
@author: renata.fernandes
"""

# CHAPTER 2 -  Visualization with hierarchical clustering and t-SNE

# Import necessary modules
import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, Normalizer,normalize
from scipy.cluster.hierarchy import linkage, dendrogram


#Hierarchical clustering of the grain data

grain_df = pd.read_csv('Dados/Grains/seeds.csv', sep=',')
grain_df['varieties'] = grain_df['varieties'].replace([1,2,3],["Canadian wheat","Kama wheat", "Rosa wheat"])
# Create arrays for the features and the response variable
varieties = grain_df['varieties'].values
samples = grain_df.drop('varieties', axis=1).values

# Calculate the linkage: mergings
mergings = linkage(samples,method='complete')

# Plot the dendrogram, using varieties as labels
dendrogram(mergings,
            labels=varieties,
            leaf_rotation=90,
            leaf_font_size=6,
)
plt.show()


#Hierarchies of stocks

stocks_df = pd.read_csv('Dados/company-stock-movements-2010-2015-incl.csv', sep=',')
# Create arrays for the features and the response variable
stock_companies = stocks_df['company'].values
stock_movements = stocks_df.drop('company', axis=1).values

# Normalize the movements: normalized_movements
normalized_movements = normalize(stock_movements)

# Calculate the linkage: mergings
mergings = linkage(normalized_movements,method='complete')

# Plot the dendrogram
dendrogram(mergings, labels=stock_companies, leaf_rotation=90, leaf_font_size=6)
plt.show()


#Different linkage, different hierarchical clustering!

eurovision_df = pd.read_csv('Dados/eurovision-2016.csv', sep=',')
# Create arrays for the features and the response variable
country_names = eurovision_df['From country'].values
samples = eurovision_df.drop(['From country', 'To country', 'Jury Points', 'Televote Points'], axis=1).values

# Calculate the linkage: mergings
mergings = linkage(samples, method='single')

# Plot the dendrogram
dendrogram(mergings, labels=country_names, leaf_rotation=90, leaf_font_size=6)
plt.show()


# #Extracting the cluster labels
# # Perform the necessary imports
# import pandas as pd
# from scipy.cluster.hierarchy import fcluster

# # Use fcluster to extract labels: labels
# labels = fcluster(mergings, 6, criterion='distance')

# # Create a DataFrame with labels and varieties as columns: df
# df = pd.DataFrame({'labels': labels, 'varieties': varieties})

# # Create crosstab: ct
# ct = pd.crosstab(df['labels'], df['varieties'])

# # Display ct
# print(ct)


# #Extracting the cluster labels
# # Import TSNE
# from sklearn.manifold import TSNE

# # Create a TSNE instance: model
# model = TSNE(learning_rate=200)

# # Apply fit_transform to samples: tsne_features
# tsne_features = model.fit_transform(samples)

# # Select the 0th feature: xs
# xs = tsne_features[:,0]

# # Select the 1st feature: ys
# ys = tsne_features[:,1]

# # Scatter plot, coloring by variety_numbers
# plt.scatter(xs, ys, c=variety_numbers)
# plt.show()



# #A t-SNE map of the stock market
# # Import TSNE
# from sklearn.manifold import TSNE

# # Create a TSNE instance: model
# model = TSNE(learning_rate=50)

# # Apply fit_transform to normalized_movements: tsne_features
# tsne_features = model.fit_transform(normalized_movements)

# # Select the 0th feature: xs
# xs = tsne_features[:,0]

# # Select the 1th feature: ys
# ys = tsne_features[:,1]

# # Scatter plot
# plt.scatter(xs, ys, alpha=0.5)

# # Annotate the points
# for x, y, company in zip(xs, ys, companies):
#     plt.annotate(company, (x, y), fontsize=5, alpha=0.75)
# plt.show()


