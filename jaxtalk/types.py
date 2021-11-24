from typing import Any, Dict, List, Union

import jax.numpy as jnp
import numpy as np

DictNest = Dict[str, jnp.ndarray]
ListNest = List[jnp.ndarray]
ListDictNest = List[DictNest]
DictListNest = Dict[str, ListNest]
Nest = Union[DictNest, ListNest, ListDictNest, DictListNest]

JaxArray = jnp.ndarray
NumpyArray = np.ndarray
