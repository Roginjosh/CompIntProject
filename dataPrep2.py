import pandas as pd
import os

# This is a data preparation file that has already been run, and the prepared 
# data is currently in the git repository. It doesn't need to be run again.

# simple function that takes the filepath row, splits it into a list of each 
# component of it's path, and then joins the desirable part's of its path into 
# a new file path, one to match the updated file location of the images moved 
# in the previous data preparation step
def buildNewPath(row):
    original = row["filepaths"]
    [_, bird, num] = original.split("/")
    newPath = os.path.join(".", "birds", bird + num.split("j")[0] + ".jpg")
    return newPath


birds = pd.read_csv("./birds/birds.csv")

birds["filepaths"] = birds.apply(buildNewPath, axis=1) # apply above function

# rename all Blue_Tit to BlueTit
birds["filepaths"] = birds["filepaths"].str.replace("Blue_Tit", "BlueTit", case=False)

# print(birds.shape[0])
birds = birds[birds["filepaths"].apply(os.path.exists)] # get rid of any records that don't have a corresponding image file
# print(birds.shape[0])
birds = birds.drop_duplicates(subset="filepaths") # get rid of duplicate records
# print(birds.shape[0])
birds.to_csv(os.path.join(".", "birds", "updatedBirds.csv"), index=False) # save to a new csv, updatedBirds.csv
