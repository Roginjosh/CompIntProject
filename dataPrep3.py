import os
import pandas as pd
from sklearn.model_selection import train_test_split

birds = pd.read_csv("./birds/updatedBirds.csv")

trainset, testset = train_test_split(
    birds, test_size=0.3, stratify=birds["label"], random_state=1505
)

trainPath = os.path.join(".", "birds", "train.csv")
testPath = os.path.join(".", "birds", "test.csv")

trainset.to_csv(trainPath, index=False)
testset.to_csv(testPath, index=False)
