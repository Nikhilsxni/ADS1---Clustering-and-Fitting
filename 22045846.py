# -*- coding: utf-8 -*-
"""
Created on Fri Jan 05 17:18:31 2024

@author: Nikhil Soni
StudentID: 22045846
"""

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# importing Data

df = pd.read_csv("CO2_emissions.csv")

# Clustering using KMeans
features = df[['CO2_Liquid_Fuel', 'CO2_Solid_Fuel',
               'CO2_Gaseous_Fuel', 'Overall_CO2']]
scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)

kmeans = KMeans(n_clusters=3, random_state=42)
df['Cluster'] = kmeans.fit_predict(features_scaled)

# PCA for dimensionality reduction
pca = PCA(n_components=2)
principal_components = pca.fit_transform(features_scaled)
df[['PC1', 'PC2']] = principal_components

# Plotting
plt.figure(figsize=(12, 8))

# Plot CO2 emissions from Liquid Fuel Consumption
plt.subplot(2, 2, 1)
scatter = plt.scatter(df['PC1'], df['PC2'],
                      c=df['Cluster'], cmap='viridis', edgecolors='k')
plt.title('CO2 Emissions from Liquid Fuel Consumption')
plt.xlabel('CO2 Emission Concentrate')
plt.ylabel('Liquid Fuel Consumption in kt')

# Add legend
legend_labels = ['0', '1', '2']
plt.legend(handles=scatter.legend_elements()[
           0], labels=legend_labels, title='Clusters')

# Plot CO2 emissions from Solid Fuel Consumption 
plt.figure(figsize=(8, 6))

plt.plot(df['Country'], df['CO2_Solid_Fuel'],
         marker='o', color='skyblue', label='Solid Fuel')
plt.title('CO2 Emissions from Solid Fuel Consumptions')
plt.xlabel('Country')
plt.ylabel('CO2 Emissions')
plt.legend()
plt.grid(True)


# plot CO2 emissions from gaseous fuel consumption
plt.figure(figsize=(12, 8))
for country in df['Country']:
    country_data = df[df['Country'] == country]
    plt.plot(country_data['Year'],
             country_data['CO2_Gaseous_Fuel'], marker='o', label=country)

plt.title('CO2 Emissions from Gaseous Fuel Consumption')
plt.xlabel('Year')
plt.ylabel('CO2 Emissions (Million Metric Tons)')
plt.legend()
plt.grid(True)


# Bar graph for overall CO2 consumption
plt.figure(figsize=(10, 6))
df.plot(x='Country', y='Overall_CO2', kind='bar', color='orange', legend=False)
plt.title('Overall CO2 Emisiions')
plt.xlabel('In 2016')
plt.ylabel('Overall CO2 Emissions (Million Metric Tons)')

plt.tight_layout()
plt.show()
