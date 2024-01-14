import pandas as pd
from glob import glob


# --------------------------------------------------------------
# Create accerator and gyroscope dataframe
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
# Merging datasets - to get the accelerator and gyroscope value at the time stamp
# --------------------------------------------------------------

df_merged = pd.concat([acc_df.iloc[:, :3], gyr_df], axis = 1)

df_merged.columns = [
    'acc_x', 
    'acc_y', 
    'acc_z', 
    'gyr_x', 
    'gyr_y', 
    'gyr_z',
    'participant',
    'label',
    'category',
    'set'
]


# --------------------------------------------------------------
# Resample data (frequency conversion) - to make accelerator and gyroscope frequency match
# --------------------------------------------------------------
# Accelerometer:    12.500HZ
# Gyroscope:        25.000Hz

sampling = {
    'acc_x' : 'mean',
    'acc_y' : 'mean',
    'acc_z' : 'mean',
    'gyr_x' : 'mean',
    'gyr_y' : 'mean',
    'gyr_z' : 'mean',
    'participant' : 'last',
    'label' : 'last',
    'category' : 'last',
    'set' : 'last'
}
# resamping is done to take same frequency reading for both accelerator and gyroscope
df_merged[:1000].resample(rule = '200ms').apply(sampling)

# grouping dataframe by the day to apply resampling for a particular day
days = [g for n, g in df_merged.groupby(pd.Grouper(freq = 'D'))]
df_resampled = pd.concat([df.resample(rule = '200ms').apply(sampling).dropna() for df in days])

df_resampled.info()
df_resampled['set'] = df_resampled['set'].astype(int)

# --------------------------------------------------------------
# Export dataset
# --------------------------------------------------------------

df_resampled.to_csv('../../data/interim/01_processed_data.csv')
df_resampled.to_pickle('../../data/interim/01_processed_data.pkl')