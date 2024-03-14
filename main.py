import pandas as pd
# How to read a CSV file as a data frame
df = pd.read_csv('data.csv')

# How to explore the data 
#print(df)
#print(df.describe())

# How to convert the data into numpy
myNumpyArray = df.to_numpy()
#print(myNumpyArray)

# Data preprocessing

df = df.drop_duplicates().dropna()
#df = df.drop_duplicates().fillna(-1000)

#print(df)

#print(df.Feature1.to_numpy())

#print(df["Feature1"])

#print(df[df.Feature1 > 2.5])

#print(df.loc[:,['Feature1','Feature2']])
#print(df.loc[0:3,['Feature1','Feature2']])

# Write a Python function that takes all the data as a single dataframe
# It isolates and returns the last column as a numpy array called labels.
# It extracts all other columns as a numpy array called features. 
# Return X_train, X_test, y_train, y_test, class names. 
# It takes the size of the split as 
# parameter. 

def train_test_split_dataframe(data, test_proportion = 0.5):
    shuffled = data.sample(frac=1)
    sizeTraining = round(data.shape[0]* (1-test_proportion))
    X_train = shuffled[:sizeTraining].to_numpy()
    X_test = shuffled[sizeTraining:].to_numpy()
    y_train = X_train["Label"].to_numpy()
    y_test = X_test["Label"].to_numpy()
    return X_train, X_test, y_train, y_test

X_train, X_test, y_train, y_test = train_test_split_dataframe(df)

pass