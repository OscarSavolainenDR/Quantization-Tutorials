# How to do FX Graph Mode Quantization (PyTorch ResNet Coding tutorial)

In this tutorial series, we use Torch's FX Graph mode quantization to quantize a ResNet. In the first video, we look at the Directed Acyclic Graph (DAG), and see how the fusing, placement of quantstubs and FloatFunctionals all happen automatically. In the second, we look at some of the intricacies of how quantization interacts with the GraphModule. In the third and final video, we look at some more advanced techniques for manipulating and traversing the graph, and use these to discover an alternative to forward hooks, and for fusing BatchNorm layers into their preceding Convs.

[![How to do FX Graph Mode Quantization: FX Graph Mode Quantization Coding tutorial - Part 1/3](https://ytcards.demolab.com/?id=AHw5BOUfLU4&title=How+to+do+FX+Graph+Mode+Quantization%3A+FX+Graph+Mode+Quantization+Coding+tutorial+-+Part+1%2F3&lang=en&timestamp=1710264531&background_color=%230d1117&title_color=%23ffffff&stats_color=%23dedede&max_title_lines=1&width=250&border_radius=5 "How to do FX Graph Mode Quantization: FX Graph Mode Quantization Coding tutorial - Part 1/3")](https://www.youtube.com/watch?v=AHw5BOUfLU4)
[![How does Graph Mode Affect Quantization? FX Graph Mode Quantization Coding tutorial - Part 2/3](https://ytcards.demolab.com/?id=1S3jlGdGdjM&title=How+does+Graph+Mode+Affect+Quantization%3F+FX+Graph+Mode+Quantization+Coding+tutorial+-+Part+2%2F3&lang=en&timestamp=1710452876&background_color=%230d1117&title_color=%23ffffff&stats_color=%23dedede&max_title_lines=1&width=250&border_radius=5 "How does Graph Mode Affect Quantization? FX Graph Mode Quantization Coding tutorial - Part 2/3")](https://www.youtube.com/watch?v=1S3jlGdGdjM)
[![Advanced PyTorch Graph Manipulation: FX Graph Mode Quantization Coding tutorial - Part 3/3](https://ytcards.demolab.com/?id=azpsgB8y0A8&title=Advanced+PyTorch+Graph+Manipulation%3A+FX+Graph+Mode+Quantization+Coding+tutorial+-+Part+3%2F3&lang=en&timestamp=1711116192&background_color=%230d1117&title_color=%23ffffff&stats_color=%23dedede&max_title_lines=1&width=250&border_radius=5 "Advanced PyTorch Graph Manipulation: FX Graph Mode Quantization Coding tutorial - Part 3/3")](https://www.youtube.com/watch?v=azpsgB8y0A8)

This repo is the code associated with the videos.

### Prerequisites:
To run this code, you need to have PyTorch installed in your environment. If you do not have PyTorch installed, please follow this [official guide](https://pytorch.org/get-started/locally/).

I created this code with PyTorch Version: 2.1.1. In case you have any versioning issues, you can revert to that version.

To run `fx_model.graph.print_tabular()`, one needs to have `tabulate` installed. To do, activate your (e.g. conda) environment and run
```
pip install tabulate
```

To plot the graph as a tree in an SVG file, i.e. to run:
```
g = passes.graph_drawer.FxGraphDrawer(fx_model, 'resnet18-fx-model')
```

One needs to install GraphViz and have it on PATH (or as a local PATH variable).

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
Then, just run `python main.py` from your command line! However I would obviously recommend that you follow along with the tutorial, so that you learn how it all works and get your hands dirty.

For the tutorial on eager mode (which includes making architecture changes to the publicly available Resenet model to make it quantizable), cd into `your-folder/Resnet-Eager-Mode-Quant` and see the `README.md` there.


Let me know if there are any issues!
