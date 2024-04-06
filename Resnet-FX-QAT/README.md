# How to Quantization Aware Training (QAT): PyTorch ResNet Coding tutorial

This is the finished code associated with the YouTube tutorial at:

**TODO**

This code is built from the code for the FX Graph mode tutorial, located in `Resnet-FX-Graph-Mode-Quant`. 
However, we modularize some stuff, and build up some of the functions a bit more.


### Prerequisites:
#### Installing PyTorch:
To run this code, you need to have PyTorch installed in your environment. If you do not have PyTorch installed, please follow this [official guide](https://pytorch.org/get-started/locally/).

I created this code with PyTorch Version: 2.1.1. In case you have any versioning issues, you can revert to that version.

#### Printing the FX graph:
To run `fx_model.graph.print_tabular()`, one needs to have `tabulate` installed. To do, activate your (e.g. conda) environment and run
```
pip install tabulate
```

#### Printing the FX graph:
For this tutorial, I downloaded some images form google search, one example each for a handful of the ImageNet classes. 
You can add whatever ImageNet class examples you want, but be make sure to you name the images the same as the class names, e.g. `hen.jpg` for classname `hen`. 
Or, feel free to generalise the code so that isn't a constraint!


### Running this code:
Once you have PyTorch installed, first navigate to a directory you will be working from. As you follow the next steps, your final file structure will look like this: `your-directory/Resnet-FX-QAT`.

Next, from `your-directory`, clone the `Quantization-Tutorials` repo. This repo contains different tutorials, but they are all interlinked. Feel no need to do any of the others! I just structured it this way because the tutorials share a lot of code and it might help people to see different parts in one place.

You can also `git init` and then `git pull/fetch`, depending on what you prefer.

To clone the repo, run:
```
git clone git@github.com:OscarSavolainenDR/Quantization-Tutorials.git .
```

If you did the cloning in place with the `.` at the end, your folder structure should look like `your-folder/Resnet-FX-QAT`, with various other folders for other tutorials.

Next, cd into the Resnet FX QAT tutorial:
```
cd Resnet-FX-QAT
```
Then, just run `python main.py` from your command line! However I would obviously recommend that you follow along with the tutorial, so that you learn how it all works and get your hands dirty.

Let me know if there are any issues!
