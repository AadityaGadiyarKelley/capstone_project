#!/usr/bin/env python
# coding: utf-8

# In[50]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score
from sklearn.metrics import r2_score, mean_squared_error
import warnings
warnings.filterwarnings('ignore')

# Load the data
train_data_one = pd.read_csv('train_FD001.txt', sep=' ', header=None)
train_data_two = pd.read_csv('train_FD002.txt', sep=' ', header=None)
train_data_three = pd.read_csv('train_FD003.txt', sep=' ', header=None)
train_data_four = pd.read_csv('train_FD004.txt', sep=' ', header=None)

# Drop the empty columns
train_data_one = train_data_one.drop(columns=[26, 27])
train_data_two = train_data_two.drop(columns=[26, 27])
train_data_three = train_data_three.drop(columns=[26, 27])
train_data_four = train_data_four.drop(columns=[26, 27])

print(train_data_one.shape)
train_data_one.head()
print(train_data_two.shape)
train_data_two.head()
print(train_data_three.shape)
train_data_three.head()
print(train_data_four.shape)
train_data_four.head()


# In[53]:


#Giving column names
column_names = ['engine_id', 'cycle', 'setting_1', 'setting_2', 'setting_3'] + [f'sensor_{i}' for i in range(1, 22)]
train_data_one.columns = column_names
train_data_two.columns = column_names
train_data_three.columns = column_names
train_data_four.columns = column_names

print(train_data_four.shape)
train_data_four.head()


# In[54]:


# Basic engine stats
print("FD001")
print(f"Number of unique engines: {train_data_one['engine_id'].nunique()}")
print(f"Engine run lengths:")
print(train_data_one.groupby('engine_id')['cycle'].max().describe())
print("FD002")
print(f"Number of unique engines: {train_data_two['engine_id'].nunique()}")
print(f"Engine run lengths:")
print(train_data_two.groupby('engine_id')['cycle'].max().describe())
print("FD003")
print(f"Number of unique engines: {train_data_three['engine_id'].nunique()}")
print(f"Engine run lengths:")
print(train_data_three.groupby('engine_id')['cycle'].max().describe())
print("FD004")
print(f"Number of unique engines: {train_data_four['engine_id'].nunique()}")
print(f"Engine run lengths:")
print(train_data_four.groupby('engine_id')['cycle'].max().describe())


# In[55]:


datasets = {
    'FD001': train_data_one,
    'FD002': train_data_two, 
    'FD003': train_data_three,
    'FD004': train_data_four
}

# Collect variations
results = {}
for dataset_name, df in datasets.items():
    variations = {}
    for col in df.columns:
        if 'sensor' in col.lower():
            variations[col] = df[col].std()
    results[dataset_name] = variations

# Convert to DataFrame and display
chart_df = pd.DataFrame(results)
print(chart_df.round(6))

# Simple heatmap
plt.figure(figsize=(10, 12))
sns.heatmap(chart_df, annot=True, fmt='.4f', cmap='viridis')
plt.title('Sensor Standard Deviation Across C-MAPSS Datasets')
plt.show()


# In[61]:


import matplotlib.pyplot as plt

engine_1_data_001 = train_data_one[train_data_one['engine_id'] == 1]
engine_1_data_002 = train_data_two[train_data_two['engine_id'] == 1]
engine_1_data_003 = train_data_three[train_data_three['engine_id'] == 1]
engine_1_data_004 = train_data_four[train_data_four['engine_id'] == 1]

# Plot sensor_4 for engine 1 over its entire lifetime (FD001)
plt.figure(figsize=(10, 4))
plt.plot(engine_1_data_001['cycle'], engine_1_data_001['sensor_4'])
plt.title('Sensor 4 - Engine 1 Over Time (FD001)')
plt.xlabel('Cycle')
plt.ylabel('Sensor 4 Value')
plt.grid(True)
plt.show()


# In[62]:


# Plot sensor_4 for engine 1 over its entire lifetime (FD002)
plt.figure(figsize=(10, 4))
plt.plot(engine_1_data_002['cycle'], engine_1_data_002['sensor_4'])
plt.title('Sensor 4 - Engine 1 Over Time (FD002)')
plt.xlabel('Cycle')
plt.ylabel('Sensor 4 Value')
plt.grid(True)
plt.show()


# In[59]:


# Plot sensor_4 for engine 1 over its entire lifetime (FD003)
plt.figure(figsize=(10, 4))
plt.plot(engine_1_data_003['cycle'], engine_1_data_003['sensor_4'])
plt.title('Sensor 4 - Engine 1 Over Time (FD003)')
plt.xlabel('Cycle')
plt.ylabel('Sensor 4 Value')
plt.grid(True)
plt.show()


# In[58]:


# Plot sensor_4 for engine 1 over its entire lifetime (FD003)
plt.figure(figsize=(10, 4))
plt.plot(engine_1_data_004['cycle'], engine_1_data_004['sensor_4'])
plt.title('Sensor 4 - Engine 1 Over Time (FD004)')
plt.xlabel('Cycle')
plt.ylabel('Sensor 4 Value')
plt.grid(True)
plt.show()


# In[56]:


datasets = {
    'FD001': train_data_one,    # Your FD001 variable
    'FD002': train_data_two,    # Your FD002 variable  
    'FD003': train_data_three,  # Your FD003 variable
    'FD004': train_data_four    # Your FD004 variable
}

fig, axes = plt.subplots(2, 2, figsize=(15, 10))
axes = axes.flatten()

for i, (dataset_name, train_df) in enumerate(datasets.items()):
    
    # Get failure cycles for each engine
    failure_cycles = train_df.groupby('engine_id')['cycle'].max()
    
    # Create histogram
    ax = axes[i]
    bins = range(0, int(failure_cycles.max()) + 25, 25)
    counts, bin_edges, patches = ax.hist(failure_cycles, bins=bins, alpha=0.7, edgecolor='black')
    
    ax.set_title(f'{dataset_name} - Engine Failure Distribution ({len(failure_cycles)} engines)')
    ax.set_xlabel('Failure Cycle')
    ax.set_ylabel('Number of Engines')
    ax.grid(True, alpha=0.3)
    
    # Add count labels on bars
    for count, edge in zip(counts, bin_edges[:-1]):
        if count > 0:
            ax.text(edge + 12.5, count + 0.5, str(int(count)), ha='center', va='bottom')

plt.tight_layout()
plt.show()


# In[ ]:




