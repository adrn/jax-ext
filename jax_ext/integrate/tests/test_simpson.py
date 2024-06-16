import sys

import jax
import jax.numpy as jnp
import pytest
from scipy.integrate import simpson as scipy_simpson

from ..simpson import ln_simpson, simpson

jax.config.update("jax_enable_x64", True)


@pytest.mark.parametrize("N", [2, 128, 129])
def test_simpson(N):
    x = jnp.linspace(0.0, 1.0, N)
    y = jnp.sin(x)

    result = simpson(y, x)
    expected = scipy_simpson(y, x)

    assert jnp.allclose(result, expected, atol=sys.float_info.epsilon)


@pytest.mark.parametrize("shape", [(32, 128), (25, 101)])
@pytest.mark.parametrize("axis", [0, 1])
def test_simpson_nd(shape, axis):
    x = jnp.linspace(0.0, 1.0, shape[0])
    x = jnp.repeat(x[:, None], shape[1], axis=1)
    y = jnp.sin(x)

    result = simpson(y, x, axis=axis)
    expected = scipy_simpson(y, x, axis=axis)

    assert jnp.allclose(result, expected, atol=sys.float_info.epsilon)


@pytest.mark.parametrize("N", [2, 128, 129])
def test_ln_simpson(N):
    x = jnp.linspace(0.0, 1.0, N)
    y = jnp.sin(x) ** 2 + 0.341  # random positive function

    result = ln_simpson(jnp.log(y), x)
    expected = jnp.log(scipy_simpson(y, x))

    assert jnp.allclose(result, expected, atol=sys.float_info.epsilon)


@pytest.mark.parametrize("shape", [(32, 128), (25, 101)])
@pytest.mark.parametrize("axis", [0, 1])
def test_ln_simpson_nd(shape, axis):
    x = jnp.linspace(0.0, 1.0, shape[0])
    x = jnp.repeat(x[:, None], shape[1], axis=1)
    y = jnp.sin(x) ** 2 + 0.341  # random positive function

    result = ln_simpson(jnp.log(y), x, axis=axis)
    expected = jnp.log(scipy_simpson(y, x, axis=axis))

    assert jnp.allclose(result, expected, atol=sys.float_info.epsilon)
