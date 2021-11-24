# jaxtalk
A re-implementation of Karpathy's [neuraltalk](https://github.com/karpathy/neuraltalk) in jax. In an attempt to study jax and build something with it, I decided to implement neuraltalk, originally implemented in numpy. 

## The Model
The captioning model uses an LSTM, a Linear layer, and Embedding Layer, implemented from scratch using jax and trained using vanilla SGD. This was made possible by the automatic differentiation in jax. Also, with the jit compilation offered in jax, this implementation is very much faster than neuraltalk. 

For a batch size of 100, on a cpu, neuraltalk performs a train step in 2.07 seconds while jaxtalk does the same in 0.07s which is about 30 times faster.

## Usage
First install jax
```
pip install --upgrade pip
pip install --upgrade "jax[cpu]"
```

Obtain the dataset by following the instructions in neuraltalk [here](https://github.com/karpathy/neuraltalk#getting-started)

Train the captioning model, using the command below
```
python main.py
```

You specify the hyperparameters with command line switches. For a full list of the hyperparameters and their description run 
```
python main.py --help
```








