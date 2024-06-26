# heinsen_attention

Reference implementation of "[Softmax Attention with Constant Cost per Token](http://arxiv.org/abs/2404.05843)" (Heinsen, 2024).

We propose a simple modification to the conventional Softmax attention mechanism applied by Transformers: Instead of quantifying pairwise query-key similarity with scaled dot-products, we quantify it with the logarithms of scaled dot-products of exponentials:

$$\overset{\text{modified}}{\text{Attention}}(Q, K, V) := \displaystyle \text{Softmax}\left( \log \frac{\exp(Q) \exp(K)^T}{\exp(c)} \right) V,$$

where $c$ is a scaling constant. This simple modification [linearizes attention](https://arxiv.org/abs/2006.16236) with exponential kernel feature maps and makes it expressible as a composition of log-sums of exponentials, with a latent space of constant size, enabling application with constant time and space complexity per token.

Note that the feature function corresponding to an exponential kernel is infinite dimensional.


## Table of Contents

* [How Does it Work?](#how-does-it-work)

* [Frequently Asked Questions](#frequently-asked-questions)

* [Installation and Usage](#installation-and-usage)

* [Important Limitations](#important-limitations) <-- make sure to read them!

* [Replicating Published Results](#replicating-published-results)

* [Notes](#notes)

* [Citing](#citing)


## How Does it Work?

It's best to _see it in action_ with a toy example. First, we will show how to compute causal (autoregressive) Softmax attention with our modification using the familiar quadratic-cost formulation. Then, we will show how we linearize computation as a composition of log-sums of exponentials, obtaining the same results. Finally, we will split the sequence in chunks and compute attention sequentially, chunk by chunk, incurring constant cost per token, again obtaining the same results. 


### Our Toy Example

Start by importing all dependencies we will need:

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
```

Now, let's create toy queries `Q`, keys `K`, and values `V`. Our method requires computing the logarithm of `V`. If there are any negative values in `V`, their logarithms will be complex numbers, which are not uniformly well-supported in PyTorch. To avoid having to deal with them in our toy example, we will limit `V`'s elements to positive numbers. Also, we will keep the number of tokens `n_tok`, key features `d_key`, and value features `d_val` tiny so that when we print results, they can fit on a single screen:

```python
# Setup for our toy example:
n_tok = 10
d_key = 4
d_val = 4

Q = torch.randn(n_tok, d_key)
K = torch.randn(n_tok, d_key)

log_V = torch.randn(n_tok, d_val)  # real
V = torch.exp(log_V)               # positive only
```


### First, Causal Softmax Attention with Quadratic Cost

Here is a PyTorch module that computes our attention mechanism with its quadratic-cost formulation,

$$\text{Softmax} \left( \log \frac{\exp(Q) \exp(K)^T}{\exp(c)} \right) V,$$

using $c = c_1 + c_2$ as the scaling constant, with $c_1 = \max(Q)$ and $c_2 = \max(K)$:

```python
class QuadraticCostCausalAttention(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, Q, K, V):
        c1, c2 = (Q.detach().max(), K.detach().max())                        # scaling constants
        sims = torch.log((Q - c1).exp() @ (K - c2).exp().transpose(-2, -1))  # [n_tok, n_tok]
        mask = sims.new_ones(sims.shape[-2:], dtype=torch.bool).tril()       # [n_tok, n_tok]
        sims = sims.masked_fill(mask.logical_not(), float('-inf'))           # [n_tok, n_tok]
        Y = F.softmax(sims, dim=-1) @ V                                      # [n_tok, d_val]
        return Y
```

Try it:

```python
quadratic_attn = QuadraticCostCausalAttention()
Y1 = quadratic_attn(Q, K, V)
print(Y1)
```


### Second, Linearized Causal Softmax Attention

Here is a PyTorch module that computes the same output, using a linearized formulation that consists _entirely of log-sums of exponentials_. Note that the module accepts `log_V` instead of `V` as an input:

```python
class LinearizedCausalAttention(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, Q, K, log_V):
        Q, K, log_V = (Q.unsqueeze(-1), K.unsqueeze(-1), log_V.unsqueeze(-2))

        H_S = torch.logcumsumexp(K + log_V, dim=-3)  # [n_tok, d_key, d_val]  eq. (6) in paper
        H_Z = torch.logcumsumexp(K        , dim=-3)  # [n_tok, d_key, 1]      eq. (6)

        log_S = torch.logsumexp(Q + H_S, dim=-2)     # [n_tok, d_val]         eq. (5)
        log_Z = torch.logsumexp(Q + H_Z, dim=-2)     # [n_tok, d_val]         eq. (5)

        Y = torch.exp(log_S - log_Z)                 # [n_tok, d_val]         eq. (2)
        return Y
```

Try it:

```python
linearized_attn = LinearizedCausalAttention()
Y2 = linearized_attn(Q, K, log_V)
print(Y2)
```

You can confirm the results are the same as with the quadratic formulation:

```python
print('Do Y1 and Y2 match?', torch.allclose(Y1, Y2))
```


### Finally, Sequential Causal Softmax Attention with Constant Cost per Token

We now sequentialize the computation by caching our attention mechanism's latent state, which has a constant size, enabling us to apply attention over a stream of tokens that arrive in chunks, with constant time and space complexity per token:

```python
class SequentialCausalAttention(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, Q, K, log_V, using_prev_context=False):
        Q, K, log_V = (Q.unsqueeze(-1), K.unsqueeze(-1), log_V.unsqueeze(-2))

        H_S = torch.logcumsumexp(K + log_V, dim=-3)   # [n_tok, d_key, d_val] eq. (6) in paper
        H_Z = torch.logcumsumexp(K        , dim=-3)   # [n_tok, d_key, 1]     eq. (6)

        if using_prev_context:
            H_S = self.prev_H_S.logaddexp(H_S)        # [n_tok, d_key, d_val] use cache
            H_Z = self.prev_H_Z.logaddexp(H_Z)        # [n_tok, d_key, 1]     use cache

        self.prev_H_S = H_S[..., -1:, :, :].detach()  # [1, d_key, d_val]     cache end-state
        self.prev_H_Z = H_Z[..., -1:, :, :].detach()  # [1, d_key, 1]         cache end-state

        log_S = torch.logsumexp(Q + H_S, dim=-2)      # [n_tok, d_val]        eq. (5)
        log_Z = torch.logsumexp(Q + H_Z, dim=-2)      # [n_tok, 1]            eq. (5)

        Y = torch.exp(log_S - log_Z)                  # [n_tok, d_val]        eq. (2)
        return Y
```

Try it:

```python
# Split sequence into a stream of chunks:
chunk_len = 3
chunks = zip(
    Q.split(chunk_len, dim=-2),
    K.split(chunk_len, dim=-2),
    log_V.split(chunk_len, dim=-2),
)

# Instantiate the module:
sequential_attn = SequentialCausalAttention()

# Compute attention over the first chunk:
chunk = next(chunks)
print('Processing a chunk with {} tokens.'.format(chunk[0].size(-2)))
Y3 = [sequential_attn(*chunk)]  # saves latent state

# Compute attention over remaining chunks, using prev context for each one:
for chunk in chunks:
    print('Processing a chunk with {} tokens.'.format(chunk[0].size(-2)))
    Y3.append(sequential_attn(*chunk, using_prev_context=True))

print('---\nConcatenated:')
Y3 = torch.cat(Y3, dim=-2)
print(Y3)
```

You can confirm the results are the same as before:

```python
print('Do Y1 and Y3 match?', torch.allclose(Y1, Y3))
```

At each step, the above module is computing attention over all tokens in the input context! Remarkably, the stream of chunks could be _never-ending_! That's right: We can compute Softmax attention over input contexts of unlimited length!


### The Key Insight

Take a single query vector $\mathbf{q}$ and a single key vector $\mathbf{k}$ in $\mathbb{R}^{d}$.

$$\mathbf{q} = \begin{bmatrix} q_1 \\\ q_2 \\\ \vdots \\\ q_d \end{bmatrix}, \quad \mathbf{k} = \begin{bmatrix} k_1 \\\ k_2 \\\ \vdots \\\ k_d \end{bmatrix}.$$

The logarithm of the dot-product $\langle \cdot, \cdot \rangle$ of their exponentials is:

$$\begin{aligned}
    \log \langle \exp(\mathbf{q}), \exp(\mathbf{k}) \rangle
    & = \log ( e^{q_1} e^{k_1} + e^{q_2} e^{k_2} + \dots + e^{q_d} e^{k_d} ) \\
    & = \log \sum \left( \begin{bmatrix} e^{q_1} \\\ e^{q_2} \\\ \vdots \\\ e_{q^d} \end{bmatrix} \odot \begin{bmatrix} e^{k_1} \\\ e_{k^2} \\\ \vdots \\\ e_{k^d} \end{bmatrix} \right) \\
    & = \log \sum \left( \begin{bmatrix} e^{q_1} e^{k_1} \\\ e^{q_2} e^{k_2} \\\ \vdots \\\ e^{q_d} e^{k_d} \end{bmatrix} \right) \\
    & = \log \sum \left( \begin{bmatrix} e^{q_1 + k_1} \\\ e^{q_2 + k_2} \\\ \vdots \\\ e^{q_d + k_d} \end{bmatrix} \right) \\
    & = \log \sum \exp \left( \begin{bmatrix} q_1 \\\ q_2 \\\ \vdots \\\ q_d \end{bmatrix} + \begin{bmatrix} k_1 \\\ k_2 \\\ \vdots \\\ k_d \end{bmatrix} \right) \\
    & = \log\sum\exp ( \mathbf{q} + \mathbf{k} ) \\
    & = \text{LSE} ( \mathbf{q} + \mathbf{k} ), \\
\end{aligned}$$

where $\text{LSE}$ is shorthand for "Logarithm of a Sum of Exponentials."

Armed with this insight, we prove that our Softmax attention mechanism is expressible as a composition of log-sums of exponentials that is linearizable, with a latent space of constant size, enabling sequential application with constant time and space complexity per token. For details, please see our paper.


## Frequently Asked Questions

*Q: "Is this method a special case of ``linear attention'' as proposed by [Katharopoulos et al (2020)](https://arxiv.org/abs/2006.16236)?"*

A: Yes. The quadratic-cost formulation is expressible as a special case of linear attention. It's the special case that applies exponential kernel feature maps, whose corresponding feature function is infinite dimensional:

$$\text{Softmax}\left( \log \frac{\exp(Q) \exp(K)^T}{\exp(c)} \right) V = \begin{bmatrix} \displaystyle \frac{\exp(Q) \exp(K)^T}{\sum_{[n_K]} \exp(Q) \exp(K)^T} \end{bmatrix} V,$$

where $\sum_{[n_K]}$ sums over the dimension indexed by the number of keys. The gram matrix is symmetric and positive semi-definite, giving us a kernel (Mercer's theorem). Expressed in code:

```python
class NumericallyUnstableCausalAttention(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, Q, K, V):
        exp_sims = Q.exp() @ K.exp().transpose(-2, -1)                          # [n_tok, n_tok]
        mask = exp_sims.new_ones(exp_sims.shape[-2:], dtype=torch.bool).tril()  # [n_tok, n_tok]
        exp_sims = exp_sims.masked_fill(mask.logical_not(), 0.0)                # [n_tok, n_tok]
        Y = (exp_sims / exp_sims.sum(dim=-1, keepdim=True)) @ V                 # [n_tok, d_val]
        return Y
```

It turns out this special case is expressible _entirely as a composition of log-sums of exponentials_.

Initially, we didn't realize our modification was a special case of linear attention. In hindsight, we're a bit embarrassed that we didn't see it right away. Maybe our gray matter was temporarily stuck on subpar local optima? Please see shaochenze's comment [here](https://github.com/glassroom/heinsen_attention/issues/1).


*Q: "Can this be generalized to functions other than _exp()_ and _log()_?"*

A: Yes. If we define $\phi = \exp$, we have:

$$\overset{\text{modified}}{\text{Attention}}(Q, K, V) := \displaystyle \text{Softmax}\left( \phi^{-1} \left( \frac{\phi(Q) \phi(K)^T}{\phi(c)} \right) \right) V.$$

The question is whether there are other functions $\phi$ that are not $\exp$ (and do not exponentiate) which (a) are invertible, and (b) enable linearization of the Softmax function as a composition of (log-) sums. We suspect the answer is no. It might be possible to replace $\exp$ and $\log$ with two functions that are not each other's inverses and together enable linearization of the Softmax function as a composition of sums, but the result might not work as well or be... as elegant.


*Q: "How can I help?"*

A: Glad you asked! The most helpful thing anyone could do is write code that addresses the two [self-imposed limitations](#important-limitations) of our implementation with efficiency and numerical stability in PyTorch. Another thing that would be helpful is implementing our method in other software frameworks (e.g., JAX, TensorFlow) and languages (e.g., Julia, Mojo) that maybe could make it easier to address both limitations. Finally, our method has yet to be tested on a diverse set of tasks and benchmarks with larger models.


## Installation and Usage

```
pip install git+https://github.com/glassroom/heinsen_attention
```

Alternatively, you can download a single file to your project directory: [heinsen_attention.py](https://github.com/glassroom/heinsen_attention/blob/main/heinsen_attention/heinsen_attention.py).

The only dependency is a recent version of [PyTorch](https://pytorch.org/).


### Usage

Our implementation returns _the logarithm of Softmax attention_, which is a float tensor like `log_V`. In practice, we have found that computing `log_V` as a float tensor directly from token states and using the logarithm of our attention mechanism as input to subsequent model components works well!

```python
# Load PyTorch module:
from heinsen_attention import LogAttention

# Instantiate PyTorch module:
log_attn = LogAttention(is_causal=True)

# Compute log(Attention(...)):
log_Y = log_attn(Q, K, log_V)
```

To compute attention over additional tokens in the same sequence, pass `using_prev_context=True` to the module's forward pass:

```python
log_Y = log_attn(Q, K, log_V, using_prev_context=True)
```

For a concrete example of how we do this, see the residual layer of the generative language model we use in our experiments, defined in the file `generative_language_model.py`.


## Important Limitations

For simplicity and expediency, we limit our implementation in two significant ways:

1. We restrict the values $V$ to positive numbers to avoid dealing with complex floating-point numbers, which incur greater overhead and presently are more cumbersome to manipulate than real floating-point numbers. In practice, we have found this isn't an issue: We work with the logarithm of attention, which is in the same space as $\log V$. For a concrete example of how we do this, see the residual layer of the generative language model we use in our experiments, defined in the file `generative_language_model.py`.

2. When computing autoregressive attention in parallel over all tokens in a sequence, we first compute all latent states with two parallel scans (`logcumsumexp`'s), keeping all latent states simultaneously in memory as intermediate values, and then reduce them, which is memory-inefficient but easier to write than a memory-efficient implementation. In practice, this impacts the amount of memory required for training.

Neither limitation is intrinsic to our attention mechanism. Both can be addressed with code.


## Replicating Published Results

The generative language model we use in our experiments is defined in the file `generative_language_model.py`. The only additional requirement is [tqdm](https://tqdm.github.io/), for displaying a progress bar when generating tokens.

Build the model with:

```python
from generative_language_model import build_model
model = build_model()
```

To replicate our results, train the model on 300B tokens from The Pile ([Gao et al, 2020](https://arxiv.org/abs/2101.00027)) using a conventional setup: AdamW optimizer with weight decay 1e-1 and betas (0.90, 0.95), and one-cycle lr schedule with short warm-up, max lr 6e-4, min lr 6e-5 (e.g., you could use [this training script](https://github.com/karpathy/nanoGPT/blob/master/train.py) by Andrej Karpathy with minor modifications). For convenience, the model splits its parameters into groups with/without weight decay:

```python
param_groups = model.get_param_groups(self, weight_decay=1e-1)
optimizer = torch.optim.AdamW(param_groups)
```

For tokenization, we use [tiktoken](https://github.com/openai/tiktoken) with the 'gpt2' vocabulary.

For training hardware, we would recommend at least an 8XA100 40GB.


## Notes

We have tested the code in this repository only on Ubuntu Linux 22.04 with Python 3.10+.


## Citing

```
@misc{heinsen2024softmax,
      title={Softmax Attention with Constant Cost per Token}, 
      author={Franz A. Heinsen},
      year={2024},
      eprint={2404.05843},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
```


## How is this used at GlassRoom?

We conceived and implemented our attention mechanism for proprietary use. Most of the original work we do at GlassRoom tends to be tightly coupled to internal code, so we cannot share it with outsiders. In this case, however, we were able to isolate our code and release it as stand-alone open-source software without having to disclose any key intellectual property. We hope others find our work and our code useful.
