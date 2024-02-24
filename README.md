# Quantization-Tutorials
A bunch of coding tutorials for my [Youtube videos on Neural Network Quantization](https://www.youtube.com/@NeuralNetworkQuantization).

# Resnet-Eager-Mode-Quant:

[![How to Quantize a ResNet from Scratch! Full Coding Tutorial (Eager Mode)](https://ytcards.demolab.com/?id=jNZ1rkIfwsM&title=How+to+Quantize+a+ResNet+from+Scratch!+Full+Coding+Tutorial+(Eager+Mode)&lang=en&timestamp=1706473016&background_color=%230d1117&title_color=%23ffffff&stats_color=%23dedede&max_title_lines=1&width=250&border_radius=5 "How to Quantize a ResNet from Scratch! Full Coding Tutorial (Eager Mode)")](https://www.youtube.com/watch?v=jNZ1rkIfwsM)


This is the first coding tutorial. We take the `torchvision` `ResNet` model and quantize it entirely from scratch with the PyTorch quantization library, using Eager Mode Quantization.

We discuss common issues one can run into, as well as some interesting but tricky bugs.

# Resnet-Eager-Mode-Dynamic-Quant:

**TODO**

In this tutorial, we do dynamic quantization on a ResNet model. We look at how dynamic quantization works, what the default settings are in PyTorch, and discuss how it differs to static quantization.

# Resnet-FX-Graph-Mode-Quant:

**TODO** 

In this tutorial, we use Torch's FX Graph mode quantization to quantize a ResNet. We look at the Directed Acyclic Graph (DAG), at how the fusing, placement of quantstubs and FloatFunctionals all happen automatically, and compare it to Eager mode, e.g. how the location of the requantization stpe will now be different.
