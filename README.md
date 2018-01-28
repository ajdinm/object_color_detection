# object_color_detection


Detection of dominant color with segmentation and neural networks is project developed for Intelligent Control for course at Faculty of Electrical Engineering, Sarajevo. This approach extracts forground by using thresholding algorithm, then converts forground pixels to HSV color space and counts most common H value (H_dom). Afterwards, (H_dom, 100, 50) is converted back to RGB and sent to pretrained NN. Neural network outputs vector which is uniquely mapped to color name (yellow, red, green, blue).

This project is developed using Python2.7, and includes [requirements file.](https://github.com/ajdinm/object_color_detection/blob/master/requirements.txt) which provides all necessary dependencies of this project. For more information, [documentation](https://pip.pypa.io/en/stable/user_guide/#requirements-files) is available.

# Repository structure
This repository is contains two folders, `data` and `src`. The former contains training and validation dataset for NN that maps RGB to color name, while the latter contains full source code. On this directory level, `dataset` folder should also be present in order to perfom detection of domiant color of the image. This folder should contain one folder for each color and inside should be images for testing. [Here](https://drive.google.com/open?id=1JrNTNRDrUuAWvPsfzFuq1-5RsWTLqcmt) is folder used for this project, which includes images from [Berkeley's BigBIRD dataset](http://rll.berkeley.edu/bigbird/). 

Inside `src` directory, two files (Python scripts) are of interest: `classifier.py` and `main.py`. The former contain code for training, validation, saving and loading NN classifiers. It assumes `data/colors` folder, which is provided along this repo. New colors and datapoints can be added as long as they follow naming and format conventions. The latter (`main.py`) contains code that builds upon `classifier.py` to provide core functionality of the project. Center focus is on `get_dominant(img)` function, which for supplied image performs thresholding and gets most common hue value of the forgrounds. Functions like `predict(filename)` and `predict_imgs()` build upon this function to provide higher level abstractions.

# Execution
Once all requirements are installed and one is positioned inside `src` folder, following command runs project example:

```
python main.py

```

# Links
[Dataset](https://drive.google.com/open?id=1unD2RfsPvZgDvCtZRKHFmHMxGVG41rwa) 
[Paper](https://drive.google.com/open?id=1YfD8BPDGXeKL_RLEyROfzrzohUerjfNb) (in Bosnian)
