---
jupytext:
  cell_metadata_filter: -all
  formats: md:myst
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.11.5
kernelspec:
  display_name: Python 3
  language: python
  name: python3
---


# User Guide

By default, JAX does all floating computations in 32-bit.
We should configure it to use 64-bit at the beginning of any program.

```{code-cell}
# Configure JAX for 64-bit computing
from jax.config import config
config.update("jax_enable_x64", True)
```

NumPy features from JAX

```{code-cell}
import jax.numpy as jnp
```

We often need random keys generator from JAX.
It is convenient to generate a set of keys in advance in a script
and use them as needed.

```{code-cell}
from jax import random
# Some keys for generating random numbers
key = random.PRNGKey(0)
keys = random.split(key, 4)
```

## Import Conventions

`CR-Suite` contains a large collection of functions organized
the form of modules. Here we summarize basic conventions of
how to load individual modules. These conventions will
be followed in the coding examples in rest of the book.

### CR-Nimble


Top level functions:

```{code-cell}
import cr.nimble as crn
```


Digital Signal Processing functions:

```{code-cell}
import cr.nimble.dsp as crdsp
```


### CR-Wavelets

```{code-cell}
import cr.wavelets as crwt
```


### CR-Sparse

Top level functions:

```{code-cell}
import cr.sparse as crs
```


Linear operators:

```{code-cell}
import cr.sparse.lop as crlop
```
