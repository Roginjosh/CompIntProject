import os
import pandas as pd
from sklearn.model_selection import train_test_split

# This is a data preparation file that has already been run, and the prepared 
# data is currently in the git repository. It doesn't need to be run again.


birds = pd.read_csv("./birds/updatedBirds.csv")

# just doing a standard 70/30 split for train and test data
# worth noting that this makes sure that there is an even 
# distribution of each label in each dataset
trainset, testset = train_test_split(
    birds, test_size=0.3, stratify=birds["label"], random_state=1505
)

trainPath = os.path.join(".", "birds", "train.csv")
testPath = os.path.join(".", "birds", "test.csv")

trainset.to_csv(trainPath, index=False)
testset.to_csv(testPath, index=False)
