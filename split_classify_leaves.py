import pandas as pd
import os
from sklearn.model_selection import train_test_split

# Read the train.csv file
csv_path = os.path.join('datasets', 'classify-leaves', 'train.csv')
df = pd.read_csv(csv_path)

print(f"Total samples: {len(df)}")
print(f"Number of classes: {df['label'].nunique()}")

# Split the dataset into train and test sets (80% train, 20% test)
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df['label'])

print(f"Training samples: {len(train_df)}")
print(f"Testing samples: {len(test_df)}")

# Save the split datasets
train_df.to_csv(os.path.join('datasets', 'classify-leaves', 'train_split.csv'), index=False)
test_df.to_csv(os.path.join('datasets', 'classify-leaves', 'test_split.csv'), index=False)

print("Dataset split complete!")
print("\nClass distribution in training set:")
print(train_df['label'].value_counts())
print("\nClass distribution in testing set:")
print(test_df['label'].value_counts())