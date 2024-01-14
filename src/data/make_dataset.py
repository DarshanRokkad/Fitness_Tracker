import pandas as pd
from glob import glob

# --------------------------------------------------------------
# Read single CSV file
# --------------------------------------------------------------
file = pd.read_csv('../../data/raw/MetaMotion/A-bench-heavy2-rpe8_MetaWear_2019-01-11T16.10.08.270_C42732BE255C_Accelerometer_12.500Hz_1.4.4.csv')

# --------------------------------------------------------------
# List all data in data/raw/MetaMotion
# --------------------------------------------------------------
files = glob("../../data/raw/MetaMotion/*.csv")
files = [f.replace('\\','/') for f in files]
file = files[0]

# --------------------------------------------------------------
# Extract features from filename
# --------------------------------------------------------------

file_name = file.split('/')[-1]

participant = file_name.split('-')[0]
label = file_name.split('-')[1]
category = file_name.split('-')[2].rstrip('2')

df = pd.read_csv(files[0])
df['participant'] = participant
df['label'] = label
df['category'] = category

# --------------------------------------------------------------
# Read all files
# --------------------------------------------------------------

acc_df = pd.DataFrame()
gyr_df = pd.DataFrame()

acc_set = 1
gyr_set = 1

for file in files:    
    file_name = file.split('/')[-1]
    
    participant = file_name.split('-')[0]
    label = file_name.split('-')[1]
    category = file_name.split('-')[2].rstrip('_MetaWear_2019').rstrip('123')
    
    df = pd.read_csv(file)
    df['participant'] = participant
    df['label'] = label
    df['category'] = category
    
    if 'Accelerometer' in file_name:
        df['set'] = acc_set
        acc_set += 1
        acc_df = pd.concat([acc_df, df], axis = 0)
    
    if 'Gyroscope' in file_name:
        df['set'] = gyr_set
        gyr_set += 1
        gyr_df = pd.concat([gyr_df, df], axis = 0)
        
acc_df[acc_df['set'] == 1]

# --------------------------------------------------------------
# Working with datetimes
# --------------------------------------------------------------
# epoch is present in milli seconds of unix time so convert it into date time and make as index 

acc_df.info()

pd.to_datetime(acc_df['epoch (ms)'], unit = 'ms')
pd.to_datetime(acc_df['epoch (ms)'], unit = 'ms').dt.day

acc_df.index = pd.to_datetime(acc_df['epoch (ms)'], unit = 'ms')
gyr_df.index = pd.to_datetime(gyr_df['epoch (ms)'], unit = 'ms')

# dropping unwanted datetime column
acc_df.drop(columns = ['epoch (ms)', 'time (01:00)', 'elapsed (s)'], axis = 1, inplace = True)
gyr_df.drop(columns = ['epoch (ms)', 'time (01:00)', 'elapsed (s)'], axis = 1, inplace = True)

# --------------------------------------------------------------
# Turn into function
# --------------------------------------------------------------

files = [f.replace('\\','/') for f in glob("../../data/raw/MetaMotion/*.csv")]

def read_data_from_files(files):
    '''
    takes the csv file names list and 
    returns the accerator dataframe and gyroscope dataframe 
    '''
    acc_df = pd.DataFrame()
    gyr_df = pd.DataFrame()

    acc_set = 1
    gyr_set = 1

    # Read all files and concatenate them with respective dataframe
    for file in files:    
        file_name = file.split('/')[-1]
        
        # Extract features from filename
        participant = file_name.split('-')[0]
        label = file_name.split('-')[1]
        category = file_name.split('-')[2].rstrip('_MetaWear_2019').rstrip('123')
        
        df = pd.read_csv(file)
        df['participant'] = participant
        df['label'] = label
        df['category'] = category
        
        if 'Accelerometer' in file_name:
            df['set'] = acc_set
            acc_set += 1
            acc_df = pd.concat([acc_df, df], axis = 0)
        
        if 'Gyroscope' in file_name:
            df['set'] = gyr_set
            gyr_set += 1
            gyr_df = pd.concat([gyr_df, df], axis = 0)
    
    # Working with datetimes
    acc_df.index = pd.to_datetime(acc_df['epoch (ms)'], unit = 'ms')
    gyr_df.index = pd.to_datetime(gyr_df['epoch (ms)'], unit = 'ms')

    # dropping unwanted datetime column
    acc_df.drop(columns = ['epoch (ms)', 'time (01:00)', 'elapsed (s)'], axis = 1, inplace = True)
    gyr_df.drop(columns = ['epoch (ms)', 'time (01:00)', 'elapsed (s)'], axis = 1, inplace = True)
    
    return acc_df, gyr_df

acc_df, gyr_df = read_data_from_files(files)

# --------------------------------------------------------------
# Merging datasets
# --------------------------------------------------------------


# --------------------------------------------------------------
# Resample data (frequency conversion)
# --------------------------------------------------------------

# Accelerometer:    12.500HZ
# Gyroscope:        25.000Hz


# --------------------------------------------------------------
# Export dataset
# --------------------------------------------------------------