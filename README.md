# Flash-Muon: An Efficient Implementation of Muon Optimzer
This repo implements an efficient version of Muon optimizer through a customized CUDA kernel for matmuls in Newton-Schulz iterations.


## Installation

```sh
git clone --recurse-submodules https://github.com/nil0x9/flash-muon.git

pip install -e ./
```

## Usage

The usage is actually identical to the [original Muon implementation](https://github.com/KellerJordan/Muon/tree/master) as the python API was adapted from there.

```python
from flash_muon import Muon

# Find ≥2D parameters in the body of the network -- these should be optimized by Muon
muon_params = [p for p in model.body.parameters() if p.ndim == 2]
# Find everything else -- these should be optimized by AdamW
adamw_params = ([p for p in model.body.parameters() if p.ndim != 2]
              + [*model.head.parameters(), *model.embed.parameters()])
# Create the optimizer
optimizers = [Muon(muon_params, lr=0.02, momentum=0.95, rank=0, world_size=1),
              torch.optim.AdamW(adamw_params, lr=3e-4, betas=(0.90, 0.95), weight_decay=0.01)]
...

# in the training step
for opt in optimizers:
    opt.step()
```


## How it works

The core of Muon is  Newton-Schulz iterations, which involves the calcultion of a "matrix polynomial" of the (normalized) gradient matrix:
$$
X\leftarrow aX + b(XX^\top)X + c(XX^\top)(XX^\top)X
$$

In [Jordan's implementation](https://github.com/KellerJordan/Muon/tree/master) this iteration is implemented with the following lines of codes:

```python
for _ in range(steps):
    A = X @ X.T
    B = b * A + c * A @ A
    X = a * X + B @ X
```

Note that with $A=XX^\top$, which is symetrical，we have $AA = AA^\top$. Say if we can implement an algorithm that calculate `lambda x: matmul(x, x.T)` faster than the native `matmul` op, then we can naturally boost the Newton-Schulz iterations to make Muon faster!

The intuition is very simple - We only calculate lower triangular parts of the result, and then transpose and copy each result tiles to the corresponding upper triangular parts. This is illustrated as follows:

![design of matmul_transpose kernel](assets/matmul_transpose_kernel.png)

This design effectively saves almost half the computation in GEMM (depending on the size of the matrix).

In our test (4096 device) this simple mechanism saves about half the compute time on matrices of dimension 9182:

![design of matmul_transpose kernel](assets/benchmark.png)


## Acknowledgement

The idea of boosting Muon by customizing a CUDA kernel for `matmul(x, x.T)` was originally proposed by Laker Newhouse in [this writing](https://www.lakernewhouse.com/assets/writing/faster-symmul-with-thunderkittens.pdf). However, they mention that their kernel "has incorrect behavior in the transpose-and-store step". We take the idea further and implement our own version of the kernel.

The cuda code was adapted from GEMM codes provided by [DD-DuDa](https://github.com/DD-DuDa) and the Muon code was adapted from [Jordan's implementation](https://github.com/KellerJordan/Muon/tree/master).

## Limitations

Currently the implementation is not compatible with cases where `x.size(1) % 8 != 0`, i.e., the reduction dimension must be a multiple of 8. The kernel is currently undertuned so it shouldn't be faster than native implementation for matrices with dimension `< 4096`. 


## Citation
```
@misc{lin2025flash,
  author       = {Tianyang Lin},
  title        = {Flash-Muon: An Efficient Implementation of Muon Optimzer},
  year         = {2025},
  url          = {https://github.com/nil0x9/flash-muon}
}
```