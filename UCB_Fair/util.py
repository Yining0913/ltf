#!/usr/bin/env python3

# from
# ML Frameworks Interoperability Cheat Sheet
# https://bl.ocks.org/miguelusque/raw/f44a8e729896a96d0a3e4b07b5176af4

import jax.dlpack
from jax import numpy as jnp
import torch

def jax_to_torch(jax_tensor):
    return torch.utils.dlpack.from_dlpack(jax.dlpack.to_dlpack(jax_tensor))

def torch_to_jax(torch_tensor):
    return jnp.asarray(src)
