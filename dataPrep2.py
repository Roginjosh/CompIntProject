import pandas as pd
import os


def buildNewPath(row):
    original = row["filepaths"]
    [_, bird, num] = original.split("/")
    newPath = os.path.join(".", "birds", bird + num.split("j")[0] + ".jpg")
    return newPath


birds = pd.read_csv("./birds/birds.csv")

birds["filepaths"] = birds.apply(buildNewPath, axis=1)

birds["filepaths"] = birds["filepaths"].str.replace("Blue_Tit", "BlueTit", case=False)

# print(birds.shape[0])
birds = birds[birds["filepaths"].apply(os.path.exists)]
# print(birds.shape[0])
birds = birds.drop_duplicates(subset="filepaths")
# print(birds.shape[0])
birds.to_csv(os.path.join(".", "birds", "updatedBirds.csv"), index=False)
