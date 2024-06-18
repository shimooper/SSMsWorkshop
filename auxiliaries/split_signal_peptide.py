import pandas as pd
from sklearn.model_selection import train_test_split

TRAIN_PATH = r'C:\repos\SSMsWorkshop\benchmarks\signal_peptide\train.csv'

train_df = pd.read_csv(TRAIN_PATH)
train_split_df, valid_split_df = train_test_split(train_df, test_size=0.15, random_state=42, stratify=train_df['label'])

train_split_df.to_csv(r'C:\repos\SSMsWorkshop\benchmarks\signal_peptide\train_split.csv', index=False)
valid_split_df.to_csv(r'C:\repos\SSMsWorkshop\benchmarks\signal_peptide\valid_split.csv', index=False)
