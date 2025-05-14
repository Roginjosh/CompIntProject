import os

root = ".\\birds\\withBackground"
birdTypes = []
for dirpath, dirnames, _ in os.walk(root):
    for dirname in dirnames:
        birdTypes.append(dirname)

for name in birdTypes:
    filenames = [f for _, _, f in os.walk(os.path.join(root, name))]
    files = filenames[0]
    for file in files:
        imgPath = os.path.join(root, name, file)
        goalPath = os.path.join('.\\birds', name+file)
        with open(imgPath, 'rb') as src:
            with open(goalPath, 'wb') as dst:
                dst.write(src.read())



