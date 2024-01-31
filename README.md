# Min Mamba
simple implementation of Mamba model with tensorflow
<img src="assets/mamba.png"/>
the original paper : [here](https://arxiv.org/abs/2312.00752) 


## What is Mamba ? 
Mamba at it's core is a recurrent neural network architecture, that outperforms Transformers with faster inference and improved handling of long sequences of length up to 1 million

## Why Mamba ? 
Mamba is 5x faster throughput than Transformers and scales linearly instead of quadratically with the length of the sequence.

## Why called Mamba ?
They call it Mamba because the build on work called S4 models to create “selective structured state space sequence models” , so it has alot of sss...like snake sound 🐍

The problem is that transformers do not scale that well to long sequence lengths. This is because the self attention mechanism is quadratic. Every word has to attend to every other word in the sentence n^2,
for example the sentence below there are 21 tokens, and 21*21=441 combinations the network has to compute through the keys, queries, and values matrices : 

<img src="assets/attention.jpeg"/>

while the core of Mamba model is RNN , lets see why?

because RNN is linear architecture :
<img src="assets/RNN.png"/>

RNN Issues :
1. Fast for generation but slow in training.
2. Collapse all the information down to a hidden space, and tend to forget information on longer sequences.

