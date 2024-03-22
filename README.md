# Quantization-Tutorials
A bunch of coding tutorials for my [Youtube videos on Neural Network Quantization](https://www.youtube.com/@NeuralNetworkQuantization).

# Resnet-Eager-Mode-Quant:

[![How to Quantize a ResNet from Scratch! Full Coding Tutorial (Eager Mode)](https://ytcards.demolab.com/?id=jNZ1rkIfwsM&title=How+to+Quantize+a+ResNet+from+Scratch%21+Full+Coding+Tutorial+%28Eager+Mode%29%0D%0A&lang=en&timestamp=1706473016&background_color=%230d1117&title_color=%23ffffff&stats_color=%23dedede&max_title_lines=1&width=250&border_radius=5 "How to Quantize a ResNet from Scratch! Full Coding Tutorial (Eager Mode)")](https://www.youtube.com/watch?v=jNZ1rkIfwsM)

This is the first coding tutorial. We take the `torchvision` `ResNet` model and quantize it entirely from scratch with the PyTorch quantization library, using Eager Mode Quantization.

We discuss common issues one can run into, as well as some interesting but tricky bugs.

# Resnet-Eager-Mode-Dynamic-Quant:

**TODO**

In this tutorial, we do dynamic quantization on a ResNet model. We look at how dynamic quantization works, what the default settings are in PyTorch, and discuss how it differs to static quantization.


# How to do FX Graph Mode Quantization (PyTorch ResNet Coding tutorial)

In this tutorial series, we use Torch's FX Graph mode quantization to quantize a ResNet. In the first video, we look at the Directed Acyclic Graph (DAG), and see how the fusing, placement of quantstubs and FloatFunctionals all happen automatically. In the second, we look at some of the intricacies of how quantization interacts with the GraphModule. In the third and final video, we look at some more advanced techniques for manipulating and traversing the graph, and use these to discover an alternative to forward hooks, and for fusing BatchNorm layers into their preceding Convs.

[![How to do FX Graph Mode Quantization: FX Graph Mode Quantization Coding tutorial - Part 1/3](https://ytcards.demolab.com/?id=AHw5BOUfLU4&title=How+to+do+FX+Graph+Mode+Quantization%3A+FX+Graph+Mode+Quantization+Coding+tutorial+-+Part+1%2F3&lang=en&timestamp=1710264531&background_color=%230d1117&title_color=%23ffffff&stats_color=%23dedede&max_title_lines=1&width=250&border_radius=5 "How to do FX Graph Mode Quantization: FX Graph Mode Quantization Coding tutorial - Part 1/3")](https://www.youtube.com/watch?v=AHw5BOUfLU4)
[![How does Graph Mode Affect Quantization? FX Graph Mode Quantization Coding tutorial - Part 2/3](https://ytcards.demolab.com/?id=1S3jlGdGdjM&title=How+does+Graph+Mode+Affect+Quantization%3F+FX+Graph+Mode+Quantization+Coding+tutorial+-+Part+2%2F3&lang=en&timestamp=1710452876&background_color=%230d1117&title_color=%23ffffff&stats_color=%23dedede&max_title_lines=1&width=250&border_radius=5 "How does Graph Mode Affect Quantization? FX Graph Mode Quantization Coding tutorial - Part 2/3")](https://www.youtube.com/watch?v=1S3jlGdGdjM)
[![Advanced PyTorch Graph Manipulation: FX Graph Mode Quantization Coding tutorial - Part 3/3](https://ytcards.demolab.com/?id=azpsgB8y0A8&title=Advanced+PyTorch+Graph+Manipulation%3A+FX+Graph+Mode+Quantization+Coding+tutorial+-+Part+3%2F3&lang=en&timestamp=1711116192&background_color=%230d1117&title_color=%23ffffff&stats_color=%23dedede&max_title_lines=1&width=250&border_radius=5 "Advanced PyTorch Graph Manipulation: FX Graph Mode Quantization Coding tutorial - Part 3/3")](https://www.youtube.com/watch?v=azpsgB8y0A8)
