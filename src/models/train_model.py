import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from LearningAlgorithms import ClassificationAlgorithms
import seaborn as sns
import itertools
from sklearn.metrics import accuracy_score, confusion_matrix


# Plot settings
plt.style.use("fivethirtyeight")
plt.rcParams["figure.figsize"] = (20, 5)
plt.rcParams["figure.dpi"] = 100
plt.rcParams["lines.linewidth"] = 2

df = pd.read_pickle('../../data/interim/03_data_features.pkl')


# --------------------------------------------------------------
# Create a training and test set
# --------------------------------------------------------------

df = df.drop(columns = ['participant', 'set', 'category'])

# separating independent and dependent feature
x = df.drop(columns = ['label'])
y = df['label']

# train test split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 42, stratify = y)


# let's see some visualization
fig, ax = plt.subplots(figsize = (10, 5))
df['label'].value_counts().plot(kind = 'bar', ax = ax, color = 'lightblue', label = 'Total')
y_train.value_counts().plot(kind = 'bar', ax = ax, color = 'dodgerblue', label = 'Train')
y_test.value_counts().plot(kind = 'bar', ax = ax, color = 'royalblue', label = 'Test')
plt.legend()
plt.show()


# --------------------------------------------------------------
# Split feature subsets
# --------------------------------------------------------------

basic_features = ['acc_x', 'acc_y', 'acc_z', 'gyr_x', 'gyr_y', 'gyr_z']
square_features = ['acc_r', 'gyr_r']
pca_features = ['pca_1', 'pca_2', 'pca_3']
time_features = [f for f in df.columns if '_temp_' in f]
freq_features = [f for f in df.columns if ('_freq' in f) or ('_pse' in f)]
cluster_features = ['cluster']

print(f'Basic features : {len(basic_features)}')
print(f'Square features : {len(square_features)}')
print(f'Pca features : {len(pca_features)}')
print(f'Time features : {len(time_features)}')
print(f'Frequency features : {len(freq_features)}')
print(f'Cluster feature : {len(cluster_features)}')

feature_set_1 = list(set(basic_features))
feature_set_2 = list(set(basic_features + square_features + pca_features))
feature_set_3 = list(set(feature_set_2 + time_features))
feature_set_4 = list(set(feature_set_3 + freq_features + cluster_features))


# --------------------------------------------------------------
# Perform forward feature selection using simple decision tree
# --------------------------------------------------------------

learner = ClassificationAlgorithms()

max_features = 10 
selected_frequency, ordered_frequency, ordered_scores = learner.forward_selection(
    max_features, x_train, y_train
)

# let's store the features to avoid running again when the kernel is crashed
selected_features = [
    'acc_x_freq_0.0_Hz_ws_14',
    'duration',
    'acc_y_freq_0.0_Hz_ws_14',
    'acc_z_temp_mean_ws_5',
    'acc_z_freq_1.786_Hz_ws_14',
    'gyr_x_max_freq',
    'gyr_r_freq_2.5_Hz_ws_14',
    'gyr_x_freq_weighted',
    'gyr_x_freq_0.357_Hz_ws_14',
    'acc_z'
]

# visualize the accuracy of the model with respect to number of features 
plt.figure(figsize = (10, 5))
plt.plot(np.arange(1, max_features+1, 1), ordered_scores)
plt.xlabel('Number of features')
plt.ylabel('Accuracy')
plt.show()


# --------------------------------------------------------------
# Grid search for best hyperparameters and model selection
# --------------------------------------------------------------

possible_feature_sets = [
    feature_set_1,
    feature_set_2,
    feature_set_3, 
    feature_set_4,
    selected_frequency
]

feature_names = [
    'Feature Set 1',
    'Feature Set 2',
    'Feature Set 3',
    'Feature Set 4',
    'Selected Feature'
]

iterations = 1
score_df = pd.DataFrame()

for i, f in zip(range(len(possible_feature_sets)), feature_names):
    print("Feature set:", i)
    selected_train_x = x_train[possible_feature_sets[i]]
    selected_test_x = x_test[possible_feature_sets[i]]

    # First run non deterministic classifiers to average their score.
    performance_test_nn = 0
    performance_test_rf = 0

    for it in range(0, iterations):
        print("\tTraining neural network,", it)
        (
            class_train_y,
            class_test_y,
            class_train_prob_y,
            class_test_prob_y,
        ) = learner.feedforward_neural_network(
            selected_train_x,
            y_train,
            selected_test_x,
            gridsearch=False,
        )
        performance_test_nn += accuracy_score(y_test, class_test_y)

        print("\tTraining random forest,", it)
        (
            class_train_y,
            class_test_y,
            class_train_prob_y,
            class_test_prob_y,
        ) = learner.random_forest(
            selected_train_x, y_train, selected_test_x, gridsearch=True
        )
        performance_test_rf += accuracy_score(y_test, class_test_y)

    performance_test_nn = performance_test_nn / iterations
    performance_test_rf = performance_test_rf / iterations

    # And we run our deterministic classifiers:
    print("\tTraining KNN")
    (
        class_train_y,
        class_test_y,
        class_train_prob_y,
        class_test_prob_y,
    ) = learner.k_nearest_neighbor(
        selected_train_x, y_train, selected_test_x, gridsearch=True
    )
    performance_test_knn = accuracy_score(y_test, class_test_y)

    print("\tTraining decision tree")
    (
        class_train_y,
        class_test_y,
        class_train_prob_y,
        class_test_prob_y,
    ) = learner.decision_tree(
        selected_train_x, y_train, selected_test_x, gridsearch=True
    )
    performance_test_dt = accuracy_score(y_test, class_test_y)

    print("\tTraining naive bayes")
    (
        class_train_y,
        class_test_y,
        class_train_prob_y,
        class_test_prob_y,
    ) = learner.naive_bayes(selected_train_x, y_train, selected_test_x)

    performance_test_nb = accuracy_score(y_test, class_test_y)

    # Save results to dataframe
    models = ["NN", "RF", "KNN", "DT", "NB"]
    new_scores = pd.DataFrame(
        {
            "model": models,
            "feature_set": f,
            "accuracy": [
                performance_test_nn,
                performance_test_rf,
                performance_test_knn,
                performance_test_dt,
                performance_test_nb,
            ],
        }
    )
    score_df = pd.concat([score_df, new_scores])


# --------------------------------------------------------------
# Create a grouped bar plot to compare the results
# --------------------------------------------------------------

score_df.sort_values(by = 'accuracy', ascending = False)

# let's visualize the accuracy of models
plt.figure(figsize = (10, 10))
sns.barplot(x = 'model', y = 'accuracy', hue = 'feature_set', data = score_df)
plt.xlabel('Model')
plt.ylabel('Accuracy')
plt.ylim(0.7, 1)
plt.legend()
plt.show()


# --------------------------------------------------------------
# Select best model and evaluate results
# --------------------------------------------------------------

class_train_y, class_test_y, class_train_prob_y, class_test_prob_y  = learner.random_forest(x_train[feature_set_4], y_train, x_test[feature_set_4], gridsearch = True)

accuracy_score = accuracy_score(y_test, class_test_y)

classes = class_test_prob_y.columns
cm = confusion_matrix(y_test, class_test_y, labels = classes)

# create confusion matrix for cm
plt.figure(figsize=(10, 10))
plt.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
plt.title("Confusion matrix")
plt.colorbar()
tick_marks = np.arange(len(classes))
plt.xticks(tick_marks, classes, rotation=45)
plt.yticks(tick_marks, classes)
thresh = cm.max() / 2.0
for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
    plt.text(
        j,
        i,
        format(cm[i, j]),
        horizontalalignment="center",
        color="white" if cm[i, j] > thresh else "black",
    )
plt.ylabel("True label")
plt.xlabel("Predicted label")
plt.grid(False)
plt.show()


# --------------------------------------------------------------
# Select train and test data based on participant
# --------------------------------------------------------------

df = pd.read_pickle('../../data/interim/03_data_features.pkl')

participant_df = df.drop(columns = ['set', 'category'], axis = 1)

x_train = participant_df[participant_df['participant'] != 'A'].drop('label', axis = 1)
x_train = x_train.drop(['participant'], axis = 1)
x_test = participant_df[participant_df['participant'] == 'A'].drop('label', axis = 1)
x_test = x_test.drop(['participant'], axis = 1)

y_train = participant_df[participant_df['participant'] != 'A']['label']
y_test = participant_df[participant_df['participant'] == 'A']['label']

# let's visualize the distribution of the split
fig, ax = plt.subplots(figsize = (10, 5))
df['label'].value_counts().plot(
    kind = 'bar', ax = ax, color = 'lightblue', label = 'Train'
)
y_train.value_counts().plot(kind = 'bar', ax = ax, color = 'dodgerblue', label = 'Train')
y_test.value_counts().plot(kind = 'bar', ax = ax, color = 'royalblue', label = 'Test')
plt.legend()
plt.show()


# --------------------------------------------------------------
# Use best model again and evaluate results
# --------------------------------------------------------------

class_train_y, class_test_y, class_train_prob_y, class_test_prob_y  = learner.random_forest(x_train[feature_set_4], y_train, x_test[feature_set_4], gridsearch = True)

accuracy_score = accuracy_score(y_test, class_test_y)

classes = class_test_prob_y.columns
cm = confusion_matrix(y_test, class_test_y, labels = classes)

# create confusion matrix for cm
plt.figure(figsize=(10, 10))
plt.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
plt.title("Confusion matrix")
plt.colorbar()
tick_marks = np.arange(len(classes))
plt.xticks(tick_marks, classes, rotation=45)
plt.yticks(tick_marks, classes)
thresh = cm.max() / 2.0
for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
    plt.text(
        j,
        i,
        format(cm[i, j]),
        horizontalalignment="center",
        color="white" if cm[i, j] > thresh else "black",
    )
plt.ylabel("True label")
plt.xlabel("Predicted label")
plt.grid(False)
plt.show()


# --------------------------------------------------------------
# Try with more complex model with the selected features
# --------------------------------------------------------------

class_train_y, class_test_y, class_train_prob_y, class_test_prob_y  = learner.feedforward_neural_network(x_train[selected_frequency], y_train, x_test[selected_frequency], gridsearch = True)

accuracy_score = accuracy_score(y_test, class_test_y)
classes = class_test_prob_y.columns
cm = confusion_matrix(y_test, class_test_y, labels = classes)

# create confusion matrix for cm
plt.figure(figsize=(10, 10))
plt.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
plt.title("Confusion matrix")
plt.colorbar()
tick_marks = np.arange(len(classes))
plt.xticks(tick_marks, classes, rotation=45)
plt.yticks(tick_marks, classes)
thresh = cm.max() / 2.0
for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
    plt.text(
        j,
        i,
        format(cm[i, j]),
        horizontalalignment="center",
        color="white" if cm[i, j] > thresh else "black",
    )
plt.ylabel("True label")
plt.xlabel("Predicted label")
plt.grid(False)
plt.show()