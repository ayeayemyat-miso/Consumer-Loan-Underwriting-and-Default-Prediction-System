import pandas as pd
print('Starting script...')
print('Loading data...')
df = pd.read_csv('data/loan_data_engineered.csv', nrows=1000)
print(f'Loaded {len(df)} rows')
print('Data loaded successfully!')
print(f'Columns: {df.columns.tolist()[:5]}...')
