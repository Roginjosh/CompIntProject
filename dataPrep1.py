import os

# This is a data preparation file that has already been run, and the prepared 
# data is currently in the git repository. It doesn't need to be run again.


root = ".\\birds\\withBackground" # this is where the original images are stored from the download
birdTypes = []

# walk thru the "root" directory and get a list of subdirectories
for dirpath, dirnames, _ in os.walk(root):
    for dirname in dirnames:
        birdTypes.append(dirname)

# for each "subdirectory," walk through it, and copy each file in it to the upper level directory
# with a concatenated name of it's folder and it's name
for name in birdTypes:
    filenames = [f for _, _, f in os.walk(os.path.join(root, name))]
    files = filenames[0]
    for file in files:
        imgPath = os.path.join(root, name, file)
        goalPath = os.path.join(".\\birds", name + file)
        with open(imgPath, "rb") as src:
            with open(goalPath, "wb") as dst:
                dst.write(src.read())
