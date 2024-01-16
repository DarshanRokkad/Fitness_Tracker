import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from DataTransformation import LowPassFilter, PrincipalComponentAnalysis
from TemporalAbstraction import NumericalAbstraction
from FrequencyAbstraction import FourierTransformation
from sklearn.cluster import KMeans


# --------------------------------------------------------------
# Load data
# --------------------------------------------------------------

df = pd.read_pickle('../../data/interim/02_outlier_removed_chauvenet.pkl')

predictor_columns = df.columns[:6]

plt.style.use('fivethirtyeight')
plt.rcParams['figure.figsize'] = (20, 5)
plt.rcParams['figure.dpi'] = 100
plt.rcParams['lines.linewidth'] = 2

# --------------------------------------------------------------
# Dealing with missing values (imputation)
# --------------------------------------------------------------

df.info()

for col in predictor_columns:
    df[col] = df[col].interpolate()


# --------------------------------------------------------------
# Calculating set duration
# --------------------------------------------------------------

df[df['set'] == 1]['acc_y'].plot()
# observation: we can see some pattern in the plot so we can smoothen the curve to get the pattern

duration = df[df['set'] == 1].index[-1] - df[df['set'] == 1].index[0]
duration.seconds

for s in df['set'].unique():
    start = df[df['set'] == s].index[0]
    stop = df[df['set'] == s].index[-1]
    
    duration = stop - start
    df.loc[df['set'] == s, 'duration'] = duration.seconds

duration_df = df.groupby(['category'])['duration'].mean()

duration_df.iloc[0] / 5
duration_df.iloc[1] / 10

# --------------------------------------------------------------
# Butterworth lowpass filter
# --------------------------------------------------------------

df_lowpass = df.copy()
LowPass = LowPassFilter()

sf = 1000 / 200     # 200ms for 1 repetation
cutoff = 1.3        # twick this parameter to get the best curve
df_lowpass = LowPass.low_pass_filter(df_lowpass, 'acc_y', sf, cutoff, order = 5)

# let's visualize some samples
subset = df_lowpass[df_lowpass['set'] == 5]
print(subset['label'][0])
fig, ax = plt.subplots(nrows = 2, sharex = True, figsize = (20, 10))
ax[0].plot(subset['acc_y'].reset_index(drop = True), label = 'raw data')
ax[1].plot(subset['acc_y_lowpass'].reset_index(drop = True), label = 'betterworth filter')
ax[0].legend()
ax[1].legend()

# apply for all the columns and overwrite the original values
for col in predictor_columns:
    df_lowpass = LowPass.low_pass_filter(df_lowpass, col, sf, cutoff, order = 5)
    df_lowpass[col] = df_lowpass[col + '_lowpass']
    df_lowpass.drop(columns = [col + '_lowpass'], inplace = True)
    

# --------------------------------------------------------------
# Principal component analysis PCA
# --------------------------------------------------------------

df_pca = df.copy()
pca = PrincipalComponentAnalysis()

pc_values = pca.determine_pc_explained_variance(df_pca, predictor_columns)

# elow method is used to obtain the optimal pricipal components i.e features
plt.figure(figsize = (10, 10))
plt.plot(range(1, len(predictor_columns) + 1), pc_values)
plt.xlabel('pricipal components analysics')
plt.ylabel('explained variance')
plt.show()

df_pca = pca.apply_pca(df_pca, predictor_columns, 3)

# let's visualize some samples
subset = df_pca[df_pca['set'] == 3]
subset[['pca_1', 'pca_2', 'pca_3']].plot()


# --------------------------------------------------------------
# Sum of squares attributes
# --------------------------------------------------------------

df_squared = df_pca.copy()

acc_r = np.sqrt(df['acc_x'] ** 2 + df['acc_y'] ** 2 + df['acc_z'] ** 2)
gyr_r = np.sqrt(df['gyr_x'] ** 2 + df['gyr_y'] ** 2 + df['gyr_z'] ** 2)

df_squared['acc_r'] = acc_r
df_squared['gyr_r'] = gyr_r

# let's visualize some samples
subset = df_squared[df_squared['set'] == 3]
subset[['acc_r', 'gyr_r']].plot(subplots = True)

# --------------------------------------------------------------
# Temporal abstraction
# --------------------------------------------------------------

df_temporal = df_squared.copy()
NumAbs = NumericalAbstraction()

predictor_columns = list(predictor_columns) + ['acc_r', 'gyr_r']

ws = int(1000 / 200)

for col in predictor_columns:
    df_temporal = NumAbs.abstract_numerical(df_temporal, [col], ws, 'mean')
    df_temporal = NumAbs.abstract_numerical(df_temporal, [col], ws, 'std')
# **observations** : here, in some window we will get 2 exercise so taking mean of 2 exercise is not good for our problem
# So, we will loop over each set of exercise and take window for only that set

df_temporal_list = []
for s in df['set'].unique():
    subset = df_temporal[df_temporal['set'] == s]
    subset = NumAbs.abstract_numerical(subset, predictor_columns, ws, 'mean')
    subset = NumAbs.abstract_numerical(subset, predictor_columns, ws, 'std')
    df_temporal_list.append(subset)

df_temporal = pd.concat(df_temporal_list)

# let's visualize some samples
subset[['acc_y', 'acc_y_temp_mean_ws_5', 'acc_y_temp_std_ws_5']].plot()
subset[['gyr_y', 'gyr_y_temp_mean_ws_5', 'gyr_y_temp_std_ws_5']].plot()


# --------------------------------------------------------------
# Frequency features
# --------------------------------------------------------------

df_freq = df_temporal.copy().reset_index()
FreqAbs = FourierTransformation()

ws = int(2800 / 200)
sr = int(1000 / 200)
df_freq = FreqAbs.abstract_frequency(df_freq, ['acc_y'], ws, sr)

# let's visualize some samples
subset = df_freq[df_freq['set'] == 15]
subset[['acc_y']].plot()
subset[
    [
        'acc_y_max_freq',
        'acc_y_freq_weighted',
        'acc_y_pse',
        'acc_y_freq_1.429_Hz_ws_14',
        'acc_y_freq_2.5_Hz_ws_14'
    ]
].plot()

# for each subset
df_freq_list =[]
for s in df_freq['set'].unique():
    subset = df_freq[df_freq['set'] == s].reset_index(drop = True).copy()
    subset = FreqAbs.abstract_frequency(subset, predictor_columns, ws, sr)
    df_freq_list.append(subset)
    
df_freq = pd.concat(df_freq_list).set_index('epoch (ms)', drop = True)


# --------------------------------------------------------------
# Dealing with overlapping windows
# --------------------------------------------------------------

df_freq.dropna(inplace = True)
df_freq = df_freq.iloc[::2]


# --------------------------------------------------------------
# Clustering
# --------------------------------------------------------------

df_cluster = df_freq.copy()

cluster_columns = ['acc_x', 'acc_y', 'acc_z']
subset = df_cluster[cluster_columns]


k_values = range(2, 10)
interias = []

# let's find the wcss value for elbow curve
for k in k_values:
    kmeans = KMeans(n_clusters = k, n_init = 20, random_state = 0)
    cluster_labels = kmeans.fit_predict(subset)
    interias.append(kmeans.inertia_)
    
# let's see elbow curve and decide the optimal k value
plt.figure(figsize = (10, 10))
plt.plot(k_values, interias)
plt.xlabel('k values')
plt.ylabel('WCSS')
plt.show()
# **observations** : We can choose 5 as the optimal k value


# add the cluster column
kmeans = KMeans(n_clusters = 5, n_init = 20, random_state = 0)
df_cluster['cluster'] = kmeans.fit_predict(subset)

# plot the cluster by cluster group
fig = plt.figure(figsize = (15, 15))
ax = fig.add_subplot(projection = '3d')
for c in df_cluster['cluster'].unique():
    subset = df_cluster[df_cluster['cluster'] == c]
    ax.scatter(subset['acc_x'], subset['acc_y'], subset['acc_z'], label = c)
ax.set_xlabel('x-axis')
ax.set_ylabel('y-axis')
ax.set_zlabel('z-axis')
plt.legend()
plt.show()

# plot the cluster by label
fig = plt.figure(figsize = (15, 15))
ax = fig.add_subplot(projection = '3d')
for c in df_cluster['label'].unique():
    subset = df_cluster[df_cluster['label'] == c]
    ax.scatter(subset['acc_x'], subset['acc_y'], subset['acc_z'], label = c)
ax.set_xlabel('x-axis')
ax.set_ylabel('y-axis')
ax.set_zlabel('z-axis')
plt.legend()
plt.show()
# **observation** : we can see that some cluster are close to each other so they are merged in above plot but actual group we go here.
# bench and overhead press are close to each other as there moment is also same they give similar reading


# --------------------------------------------------------------
# Export dataset
# --------------------------------------------------------------
df_cluster.to_pickle('../../data/interim/03_data_features.pkl')