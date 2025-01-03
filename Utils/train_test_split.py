import pandas as pd
from sklearn.model_selection import train_test_split

# Load the data
data = pd.read_csv('../Data/happiness_train_complete.csv', encoding='gb2312')

train, test = train_test_split(data, test_size=0.1, random_state=42)

label = test[['id', 'happiness']]


# Save the data
train.to_csv('../Data/train.csv', index=False)
test.to_csv('../Data/test.csv', index=False)


