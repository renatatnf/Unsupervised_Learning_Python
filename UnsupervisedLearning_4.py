# -*- coding: utf-8 -*-
"""
@author: renata.fernandes
"""

#CHAPTER 4 - Discovering interpretable features
# Perform the necessary imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import NMF,  PCA
from scipy.sparse import csr_matrix
from sklearn.preprocessing import normalize, Normalizer, MaxAbsScaler
from sklearn.pipeline import make_pipeline

#NMF applied to Wikipedia articles
articles_df = pd.read_csv('Dados/Wikipedia articles/wikipedia-vectors.csv', sep=',', index_col=0)
articles = csr_matrix(articles_df.transpose())
titles = list(articles_df.columns)

# Create an NMF instance: model
model = NMF(n_components=6)

# Fit the model to articles
model.fit(articles)

# Transform the articles: nmf_features
nmf_features = model.transform(articles)

# Print the NMF features
print(nmf_features.round(2))


#NMF features of the Wikipedia articles

# Create a pandas DataFrame: df
df = pd.DataFrame(nmf_features,index=titles)

# Print the row for 'Anne Hathaway'
print(df.loc['Anne Hathaway',:])

# Print the row for 'Denzel Washington'
print(df.loc['Denzel Washington',:])



#NMF reconstructs samples
NMF_Components = np.array([[1,0.5,0],[0.2,0.1,2.1]])
featuresValuesSample= np.array([[2,1]])
np.dot(featuresValuesSample,NMF_Components)


#NMF learns topics of documents

# Create a DataFrame: components_df
f = open('Dados/Wikipedia articles/wikipedia-vocabulary-utf8.txt', 'r')
#print(f.read())
words = f.read().splitlines()

components_df = pd.DataFrame(model.components_, columns=words)

# Print the shape of the DataFrame
print(components_df.shape)

# Select row 3: component
component = components_df.iloc[3,:]

# Print result of nlargest
print(component.nlargest())


#Which articles are similar to 'Cristiano Ronaldo'?

# Normalize the NMF features: norm_features
norm_features = normalize(nmf_features)

# Create a DataFrame: df
df = pd.DataFrame(norm_features, index=titles)

# Select the row corresponding to 'Cristiano Ronaldo': article
article = df.loc['Cristiano Ronaldo']

# Compute the dot products: similarities
similarities = df.dot(article)

# Display those with the largest cosine similarity
print(similarities.nlargest())





#Explore the LED digits dataset
lcd_df = pd.read_csv('Dados/lcd-digits.csv', sep=',',header = None)
lcd_df_samples =  lcd_df.to_numpy()
# Select the 0th row: digit
digit = lcd_df_samples[0,:]

# Print digit
print(digit)

# Reshape digit to a 13x8 array: bitmap
bitmap = digit.reshape(13,8)

# Print bitmap
print(bitmap)

# Use plt.imshow to display bitmap
plt.imshow(bitmap, cmap='gray', interpolation='nearest')
plt.colorbar()
plt.show()



#NMF learns the parts of images
def show_as_image(sample):
    bitmap = sample.reshape((13, 8))
    plt.figure()
    plt.imshow(bitmap, cmap='gray', interpolation='nearest')
    plt.colorbar()
    plt.show()

# Create an NMF model: model
model = NMF(n_components=7)

# Apply fit_transform to samples: features
features = model.fit_transform(lcd_df_samples)

# Call show_as_image on each component
for component in model.components_:
    show_as_image(component)

# Assign the 0th row of features: digit_features
digit_features = features[0,:]

# Print digit_features
print(digit_features)



#PCA doesn't learn parts

# Create a PCA instance: model
model = PCA(n_components=7)

# Apply fit_transform to samples: features
features = model.fit_transform(lcd_df_samples)

# Call show_as_image on each component
for component in model.components_:
    show_as_image(component)
    
    

# #Recommend musical artists part 1
# artists_names_df = pd.read_csv('Dados/Musical artists/artists.csv', sep=',', header = None)
# artists_df = pd.read_csv('Dados/Musical artists/scrobbler-small-sample.csv', sep=',')

# # Create a MaxAbsScaler: scaler
# scaler = MaxAbsScaler()

# # Create an NMF model: nmf
# nmf = NMF(n_components=20)

# # Create a Normalizer: normalizer
# normalizer = Normalizer()

# # Create a pipeline: pipeline
# pipeline = make_pipeline(scaler,nmf,normalizer)

# # Apply fit_transform to artists: norm_features
# norm_features = pipeline.fit_transform(artists_df)


# #Recommend musical artists part 2

# # Create a DataFrame: df
# df = pd.DataFrame(norm_features, index=artists_names_df[:])

# # Select row of 'Bruce Springsteen': artist
# artist = df.loc['Bruce Springsteen']

# # Compute cosine similarities: similarities
# similarities = df.dot(artist)

# # Display those with highest cosine similarity
# print(similarities.nlargest())
