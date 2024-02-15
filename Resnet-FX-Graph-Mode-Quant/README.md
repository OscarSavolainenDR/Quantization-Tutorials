# How to do FX Graph Mode Quantization of a ResNet (Static)

This is the finished code associated with the Youtube tutorial at:

**TODO**


### Prerequisites:
To run this code, you need to have PyTorch installed in your environment. If you do not have PyTorch installed, please follow this [official guide](https://pytorch.org/get-started/locally/).

I created this code with PyTorch Version: 2.1.1. In case you have any versioning issues, you can revert to that version.

### Running this code:
Once you have PyTorch installed, first navigate to a directory you will be working from. As you follow the next steps, your final file structure will look like this: `your-directory/Resnet-FX-Graph-Mode-Quant`.

Next, from `your-directory`, clone the `Quantization-Tutorials` repo. This repo contains different tutorials, but they are all interlinked. Feel no need to do any of the others! I just structured it this way because the tutorials share a lot of code and it might help people to see different parts in one place.

You can also `git init` and then `git pull/fetch`, depending on what you prefer.

To clone the repo, run:
```
git clone git@github.com:OscarSavolainenDR/Quantization-Tutorials.git .
```

If you did the cloning in place with the `.` at the end, your folder structure should look like `your-folder/Resnet-FX-Graph-Mode-Quant`, with various other folders for other tutorials.

Next, cd into the Resnet FX Graph Mode Quantization tutorial:
```
cd Resnet-FX-Graph-Mode-Quant
```
Then, just run `python fx_quant_resnet.py` from your command line! However I would obviously recommend that you follow along with the tutorial, so that you learn how it all works and get your hands dirty.

If you want to compare the process to eager mode, just run  `python eager_quant_resnet.py`. For the tutorial on eager mode (which includes making architecture changes to the publicly available Resenet model to make it quantizable), cd into `your-folder/Resnet-Eager-Mode-Quant` and see the `README.md` there.


Let me know if there are any issues!
