# Project README
## *Josh Rogge*
### How to Load the Environment
*I used **Git Bash** on **Windows 10**.*

### To set up the Conda environment, run:
`$ conda env create -f environment.yaml`

`$ conda activate CIProject`
### File Structure

This project has a few "types" of files.
1. dataPrepX.py files
    - These files are used for data preprocessing. In order to run the training and evaluating, these do not need to be run by you, as the data will already be prepared and  uploaded to github. They are mostly here to see the steps followed for data preparation.
2. training_***.py files
    - These files are what I used to train each type of architecture that I tested out. Right now there is just *basic and *ResNet18. They should be able to be run fresh from downloading from GitHub. If you run these, it will slightly change the accuracies that are mentioned elsewhere in the code, but likely not by very much.
3. eval_***.py files
    - These files are what I used to evaluate each network, and check how accurate it is. Quick reminder that the statistical average is about 5% accuarcy, since there are 20 relatively evenly distributed classes.

4. *.pth files
    - These files are how the freshly trained networks are saved. A fresh download from github will pull the most recent network saved to these files, which is what most of the documentation is based on. Running one of the **type 2** files will overwrite this. The **type 3** files will load their networks for evaluation from these files.

5. Misc Files
    - *.json, these files are used to make sure that the index to label relationship is held constant between testing and evaluation **This is most certainly the hardest lesson learned.**
    - environment.yaml, this file is what you use to load the python environment that I used to develop this project, it relies on conda to manage dependencies so that someone doesn't have to go thru and manually install every single library that I use in any possible way.
    - myClasses.py, this file is just used for code readability. I store my custom dataset class and my basic CNN classes here.
    - README.md this file is the catchall basic explanations for everything.

### Running the Evaluation
If you just want to check the results, you can run the evaluation scripts directly:

- eval_basic.py evaluates the model trained in training_basic.py

    - The basic model currently achieves about 35% accuracy, but this may vary slightly if you retrain.
- eval_ResNet18.py evaluates the model trained in training_ResNet18.py
    - The ResNet18 model currently achieves about 66% accuracy, with similar variance on retrain.

### My Basic Workflow

**1. Download the Dataset**

Dataset source:
20 UK Garden Birds – Kaggle:
https://www.kaggle.com/datasets/davemahony/20-uk-garden-birds?select=withBackground

The download has the following structure:
``` 
birds/
└── withBackground/
│   ├── Blackbird/
│   │   ├── (1).jpg
│   │   ├── (2).jpg
│   │   └── ...
│   ├── Bluetit/
│   ├── Carrion_Crow/
│   └── ...
```
**2. Process Data to fit in Pytorch's Dataset Class**

PyTorch’s Dataset class expects:

- All images in a single flat folder

- A corresponding annotations CSV

To meet this format, I used dataPrep1.py, which transforms the folder structure into:

```
birds/
└── Blackbird(1).jpg
└── Blackbird(2).jpg
└── ...
└── Bluetit(1).jpg
└── Bluetit(2).jpg
└── ...
└── ....
```

I then used dataPrep2.py, which updates the annotations file to match the "transformed" data.

It has three purposes:

    1. Update the filepaths column
    2. Get rid of records that don't correspond to an existing file.
    3. Get rid of duplicate records.


**3. Split Data for Training and Testing**

I needed to split the full dataset into a training set and a testing set. **Originally, I just used an index split in each of the training and eval files. What ended up happening is i trained on classes 0 thru 13 and tested on 14 thru 19. This gave me an accuracy of 0%**
To fix this, I implemented dataPrep3.py, which uses sklearn's built in dataset splitting function to pseudorandomly split the train and test sets, while making sure that both sets have proper representation of each class in each set.

**4. Create a Custom Dataset Class**
This class can be found in myClasses.py, called CustomImageDataset. This was done following the tutorial of how to implement datasets on pytorch's website at:

https://docs.pytorch.org/tutorials/beginner/basics/data_tutorial.html

specific modification details can be found in the file itself.

**5. Create a custom basic CNN network**
This class can befound in myClasses.py, called Net. This was done closely following the tutorial of how to implement datasets on pytorch's website at:

https://docs.pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html

This tutorial is a CNN that classifies 32 by 32 images, so a few small modifications were made to fit my usecase. Specific modification details can be found in the file itself.

**6. Train the Basic CNN on the Train Split of Data**
This step is done in the training_basic.py file. It is also heavily based on the CIFAR10 tutorial linked above. A few modifications were also made based on the code I am working on in my graduate research, such as the project bar and elapsed time code. Specific details of the code are contained within the file itself.

**7. Evaluate the Basic CNN's Performance on the Test Split of Data**
This step is done in the eval_basic.py file. It is also heavily based on the CIFAR10 tutorial linked above. The same modifications made above are made here. Specific details of the code are contained within the file itself. I ended up getting an accuracy of 35% with this simple network. Considering the statistical average to be just about 5%, this is a huge improvement. As someone who would not consider them an avid "birder", I can get about 30% on the images after studying them for about 30 minutes. The performance is not what I would consider excellent, but I also did not devote an extraordinary amount of time tuning this simple network for this project.

**8. Train the ResNet18 Model on the Train Set of Data**
This step is done in training_ResNet18.py. After viewing the performance and training time of the basic CNN, I wanted to see how much I could improve my results just by swapping out my basic CNN for a pretrained ResNet18 Model, which is known for how effective it is on images.

**9. Evaluate the ResNet18 Model on the Test Set of Data**
This step is done in eval_ResNet18.py. This file basically just swaps a few lines out of eval_basic.py to check and see the improvement that resnet had over a basic CNN. I ended up getting an accuracy of 66% with this simple network. Considering the statistical average to be just about 5%, and the basic CNN just doing 35%, this is a huge improvement. This specific performance was not excellent, but while messing around with the model I was able to get it's accuracy range to jump between about 63% to upwards of 78%, which I certainly would consider pretty good.