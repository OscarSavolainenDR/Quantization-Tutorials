# How to Quantize a ResNet from Scratch

This is the finished code associated with the Youtube tutorial at: 

[![How to Quantize a Resnet to Int8 from scratch (Coding Tutorial)](https://ytcards.demolab.com/?id=8dLSVrnuBXM&title=How+to+Quantize+a+Resnet+to+Int8+from+scratch+(Coding+Tutorial)&lang=en&timestamp=1706473016&background_color=%230d1117&title_color=%23ffffff&stats_color=%23dedede&max_title_lines=1&width=250&border_radius=5 "How to Quantize a Resnet to Int8 from scratch (Coding Tutorial)")](https://www.youtube.com/watch?v=8dLSVrnuBXM)

### Prerequisites:
To run this code, you need to have PyTorch installed in your environment. If you do not have PyTorch installed, please follow this [official guide](https://pytorch.org/get-started/locally/).

I created this code with PyTorch Version: 2.1.1. In case you have any versioning issues, you can revert to that version.

### Running this code:
Once you have PyTorch installed, first navigate to a directory you will be working from. As you follow the next steps, your final file structure will look like this: `your-directory\Quantization-Tutorials\Resnet`.

Next, from `your-directory`, clone the `Quantization-Tutorials` repo. This contains different tutorials, but they are all interlinked. Feel no need to do any of the others! I just structured it this way because the tutorials share a lot of code and it might help people to see different parts in one place.

You can also `git init` and then `git pull/fetch`, depending on what you prefer.

To clone the repo, run
```
git clone git@github.com:OscarSavolainenDR/Quantization-Tutorials.git
```

Next, cd into the Resnet tutorial:
```
cd Resnet
```
Then, just run `python quant_resnet.py` from your command line! However I would obviously recommend that you follow along with the tutorial, so that you learn how it all works and get your hands dirty.

Let me know if there are any issues!
