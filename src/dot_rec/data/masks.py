"""Explicit missingness; zero-valued content is not a missingness signal."""
import numpy as np
import torch


def fixed_mask(natural, rate, seed, pattern="independent"):
    natural = np.asarray(natural)
    if natural.dtype != np.bool_ or natural.ndim != 2 or natural.shape[1] != 2:
        raise ValueError("natural must be boolean [I,2]")
    if type(rate) not in (int, float) or not np.isfinite(rate) or not 0 <= rate <= 1:
        raise ValueError("Missing rate must be in [0,1]")
    if type(seed) is not int or seed < 0:
        raise ValueError("seed must be a nonnegative integer, not a boolean")
    rng = np.random.default_rng(seed)
    if pattern == "independent":
        keep = rng.random(natural.shape) >= rate
    elif pattern == "coupled":
        keep = np.repeat(rng.random((len(natural), 1)) >= rate, 2, axis=1)
    elif pattern == "visual":
        keep = np.ones_like(natural)
        keep[:, 0] = rng.random(len(natural)) >= rate
    elif pattern == "text":
        keep = np.ones_like(natural)
        keep[:, 1] = rng.random(len(natural)) >= rate
    else:
        raise ValueError("Unknown missingness pattern")
    return natural & keep


def dropout_view(observed, rate, generator):
    if type(rate) not in (int, float) or not np.isfinite(rate) or not 0 <= rate <= 1:
        raise ValueError("Drop rate must be in [0,1]")
    # CPU RNG makes the view reproducible independently of model device.
    keep = torch.rand(observed.shape, generator=generator, device="cpu") >= rate
    return observed & keep.to(observed.device)
