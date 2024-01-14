import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import os

# --------------------------------------------------------------
# Load data
# --------------------------------------------------------------

df = pd.read_pickle('../../data/interim/01_processed_data.pkl')


# --------------------------------------------------------------
# Plot single columns
# --------------------------------------------------------------

set_df = df[ df['set'] == 1 ]

plt.plot(set_df['acc_y'])
# **observation**: unable to see the number of samples
plt.plot(set_df.reset_index()['acc_y'])
# **observation**: we can see there are 80 samples


# --------------------------------------------------------------
# Plot all exercises
# --------------------------------------------------------------

temp_df = df.reset_index()

for exercise in temp_df['label'].unique():
    subset_df = temp_df[ temp_df['label'] == exercise ]
    fig, ax = plt.subplots()
    plt.plot(subset_df['acc_y'], label = exercise)
    plt.legend()
    plt.show()

# seeing in details by taking 100 samples
for exercise in temp_df['label'].unique():
    subset_df = temp_df[ temp_df['label'] == exercise ]
    fig, ax = plt.subplots()
    plt.plot(subset_df[:100]['acc_y'], label = exercise)
    plt.legend()
    plt.show()


# --------------------------------------------------------------
# Adjust plot settings
# --------------------------------------------------------------

mpl.style.use('seaborn-v0_8-deep')
mpl.rcParams['figure.figsize'] = (20, 5)
mpl.rcParams['figure.dpi'] = 100 # to extract high resolution


# --------------------------------------------------------------
# Compare medium vs. heavy sets
# --------------------------------------------------------------

category_df = temp_df.sort_values('participant').sort_values('set').reset_index().query("label == 'squat'").query("participant == 'A'")

fig, ax = plt.subplots()
category_df.groupby(['category'])['acc_y'].plot()
ax.set_label('acc_y')
ax.set_label('samples')
plt.legend()
# **observations** : When the person A is doing the squat exercise with heavy weights have less frequency and with medium weights have more frequency in accerator y-axis
# a person can lift up and down medium weight fast then heavy weights 


# --------------------------------------------------------------
# Compare participants
# --------------------------------------------------------------

participant_df = temp_df.query('label == "bench"').sort_values('participant').reset_index()

fig, ax = plt.subplots()
participant_df.groupby('participant')['acc_y'].plot()
ax.set_xlabel('participant')
ax.set_ylabel('acc_y')
plt.legend()

# --------------------------------------------------------------
# Plot multiple axis
# --------------------------------------------------------------

exercise = 'squat'
participant = 'A'
all_axis_df = df.query(f'label == "{exercise}"').query(f'participant == "{participant}"').reset_index()

fig, ax = plt.subplots()
all_axis_df[['acc_x', 'acc_y', 'acc_z']].plot(ax = ax)
ax.set_xlabel('samples')
ax.set_ylabel('accerator reading')
plt.legend()

# --------------------------------------------------------------
# Create a loop to plot all combinations per sensor
# --------------------------------------------------------------

exercises = df['label'].unique()
participants = df['participant'].sort_values().unique()

# for accerator 
for exercise in exercises:
    for participant in participants:
        all_axis_df = (
            df.query(f'label == "{exercise}"')
            .query(f'participant == "{participant}"')
            .reset_index()
        )
        
        if len(all_axis_df) > 0 :    # to avoid the empty plots
            fig, ax = plt.subplots()
            all_axis_df[['acc_x', 'acc_y', 'acc_z']].plot(ax = ax)
            ax.set_xlabel('samples')
            ax.set_ylabel('accerator reading')
            plt.title(f"{exercise} exercise done by participant {participant}")
            plt.legend()   

# for gyroscope 
for exercise in exercises:
    for participant in participants:
        all_axis_df = (
            df.query(f'label == "{exercise}"')
            .query(f'participant == "{participant}"')
            .reset_index()
        )
        
        if len(all_axis_df) > 0 :    # to avoid the empty plots
            fig, ax = plt.subplots()
            all_axis_df[['gyr_x', 'gyr_y', 'gyr_z']].plot(ax = ax)
            ax.set_xlabel('samples')
            ax.set_ylabel('gyroscope reading')
            plt.title(f"{exercise} exercise done by participant {participant}")
            plt.legend()   


# --------------------------------------------------------------
# Combine plots in one figure
# --------------------------------------------------------------

exercise = 'row'
participant = 'A'
combined_plot_df = (
    df.query(f'label == "{exercise}"')
    .query(f'participant == "{participant}"')
    .reset_index()
)

fig, ax = plt.subplots(nrows = 2, sharex = True, figsize = (20, 10))
combined_plot_df[['acc_x', 'acc_y', 'acc_z']].plot(ax = ax[0])
combined_plot_df[['gyr_x', 'gyr_y', 'gyr_z']].plot(ax = ax[1])

ax[1].set_xlabel('samples')
ax[0].set_ylabel('accerator reading')
ax[1].set_ylabel('gyroscope reading')

# --------------------------------------------------------------
# Loop over all combinations and export for both sensors
# --------------------------------------------------------------

exercises = df['label'].unique()
participants = df['participant'].sort_values().unique()

path = '../../reports/figures/'
if not os.path.exists(path):
    os.makedirs(path)
    
for exercise in exercises:
    for participant in participants:
        combined_plot_df = (
            df.query(f'label == "{exercise}"')
            .query(f'participant == "{participant}"')
            .reset_index()
        )
        
        if len(combined_plot_df) > 0 :    # to avoid the empty plots
            fig, ax = plt.subplots(nrows = 2, sharex = True, figsize = (20, 10))
            combined_plot_df[['acc_x', 'acc_y', 'acc_z']].plot(ax = ax[0])
            combined_plot_df[['gyr_x', 'gyr_y', 'gyr_z']].plot(ax = ax[1])

            ax[1].set_xlabel('samples')
            ax[0].set_ylabel('accerator reading')
            ax[1].set_ylabel('gyroscope reading')             
            
            plt.savefig(f'../../reports/figures/{exercise.title()}({participant}).png')
