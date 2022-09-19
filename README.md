# Highway Networks

Kaan Yarali ky2446


Highway networks are novel neural network architectures
which enable the training of extremely deep networks.

* __figures__: Figures used in preparing the project report.
* __models__: Models that performed the best results and presented in the report are also saved in here. These models can be laoded and run. 
* __history__:  While saving the above-mentioned models, the loss values at each epoch saved as numpy array.
* __dataset.py__: This module downloads the dataset and installs it. (from tensorflow-tfds)
* __highwayFC.py__: This module implements the custom fully connected highway layer.
* __highwayclass.py__: This module implements the highway network using the custom highway layer and tensorflow sequential model API.
* __plain.py__: This module implements the plain network using tensorflow sequential model API.
* __convolutionalHighwaylayer.py__: This module implements the custom convolutional highway layer.
* __utilsProject.py__: This module contains the utility functions used throughout the project.
* __main.ipynb__: This module shows the example usage of the fully connected highway networks,the plain networks and the functions implemented in the utilsProject.py.

The detailed report can be found in [here](https://github.com/ecbme4040/e4040-2021spring-project-HIGH-ky2446/blob/main/E4040.2021Spring.HIGH.report.ky2446.pdf)

# Running Instructions

This project is implemented in the conda TF2020 enviroment. There are two important packages to be installed in that environment before running.

```
conda install seaborn

pip install tensorflow-datasets
```

# Organization of this directory
```
.
├── E4040.2021Spring.HIGH.report.ky2446.pdf
├── README.md
├── figures
│   ├── 100layerhighwayVSplain.png
│   ├── 10layerhighwayVSplain.png
│   ├── 20layerhighwayVSplain.png
│   ├── 50layerhighwayVSplain.png
│   ├── activationgraphs
│   │   ├── mean.png
│   │   └── variance.png
│   ├── heatmaps
│   │   ├── cifar100
│   │   │   ├── cifar100meantransformgateoutput.png
│   │   │   ├── cifar100outputheatmap.png
│   │   │   ├── cifar100transformbiasheatmap.png
│   │   │   └── cifar100transformgateoutputheatmap.png
│   │   └── mnist
│   │       ├── MNISTmeantransformgateoutput.png
│   │       ├── mnistoutputheatmap.png
│   │       ├── mnisttransformbiasheatmap.png
│   │       └── mnisttransformgateoutputheatmap.png
│   └── lesiongraphs
│       ├── lesionedhighwaycifar100.png
│       └── lesionedhighwaymnist.png
├── history
│   └── ecbm4040ProjectHistory.zip
├── main.ipynb
├── models
│   └── ecbm4040ProjectModels.zip
├── requirements.txt
└── utils
    ├── convolutionalHighwaylayer.py
    ├── dataset.py
    ├── highwayFC.py
    ├── highwayclass.py
    ├── plain.py
    └── utilsProject.py
```
