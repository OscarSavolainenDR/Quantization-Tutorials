# Cross Layer Equalization (CLE): PyTorch ResNet Coding tutorial

This is the finished code associated with the YouTube tutorial at:

[![Cross Layer Equalization: Everything You Need to Know](https://ytcards.demolab.com/?id=3eATdsWmHyI&title=Cross+Layer+Equalization%3A+Everything+You+Need+to+Know&lang=en&timestamp=1715768680&background_color=%230d1117&title_color=%23ffffff&stats_color=%23dedede&max_title_lines=1&width=250&border_radius=5 "Cross Layer Equalization: Everything You Need to Know")](https://www.youtube.com/watch?v=3eATdsWmHyI)

This code is built from the code for the QAT tutorial, located in `Resnet-FX-QAT`.
We expand upon it to allow fusing of Conv and BN for float as well as quantized models.
We add capability to do CLE, including automating of the production of the list of to-be-CLE'd
layer pairs via a graph-tracing technique.

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

For this tutorial, I downloaded some images from google search, one example each for a handful of the ImageNet classes.
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

Next, cd into the Resnet FX CLE tutorial:

```
cd Resnet-FX-CLE
```

Then, just run `python main.py` from your command line! However I would obviously recommend that you follow along with the tutorial, so that you learn how it all works and get your hands dirty.

The code is also available as a Jupyter Notebook: `CLE_notebook.ipynb`, in case this is preferred.

Let me know if there are any issues!
