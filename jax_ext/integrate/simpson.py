from functools import partial

import jax
import jax.numpy as jnp
import jax.typing as JT
from jax.scipy.special import logsumexp

__all__ = ["simpson", "ln_simpson"]

@partial(jax.jit, static_argnums=1)
def _basic_simpson(y: JT.ArrayLike, stop: int, x: JT.ArrayLike):
    """
    Note: Interface comes from scipy.integrate.simpson implementation
    """
    step = 2

    slice0 = slice(0, stop, step)
    slice1 = slice(1, stop + 1, step)
    slice2 = slice(2, stop + 2, step)

    # Account for possibly different spacings.
    #    Simpson's rule changes a bit.
    h = jnp.diff(x)
    h0 = h[slice0]
    h1 = h[slice1]
    hsum = h0 + h1
    hprod = h0 * h1
    h0divh1 = jnp.where(h1 != 0, h0 / h1, 0.0)
    tmp = (
        hsum
        / 6.0
        * (
            y[slice0] * (2.0 - jnp.where(h0divh1 != 0, 1 / h0divh1, 0.0))
            + y[slice1] * (hsum * jnp.where(hprod != 0, hsum / hprod, 0.0))
            + y[slice2] * (2.0 - h0divh1)
        )
    )
    result = jnp.sum(tmp)

    return result


@jax.jit
def simpson(y: JT.ArrayLike, x: JT.ArrayLike) -> float:
    """
    Integrate y(x) using values `y` evaluated at the locations `x`

    If there are an even number of samples, N, then there are an odd number of intervals
    (N-1), but Simpson's rule requires an even number of intervals. In this case, the
    implementation uses Simpson's rule for the first N-2 intervals with the addition of
    a 3-point parabolic segment for the last interval using equations outlined by
    Cartwright. See the docstring for `scipy.integrate.simpson` for more information.

    Note: `x` values must be increasing and `x` and `ln_y` must have the same length.

    Parameters
    ----------
    y : array_like
        Array of values to be integrated.
    x : array_like
        The points at which `y` is evaluated.

    Returns
    -------
    float
        The estimated integral computed with the composite Simpson's rule.
    """
    y = jnp.array(y)
    x = jnp.array(x)
    N = len(y)

    if N % 2 == 0:  # Even number of points, odd intervals
        result = 0.0

        if N == 2:
            # need at least 3 points in integration axis to form parabolic segment
            last_dx = x[-1] - x[-2]
            result += 0.5 * last_dx * (y[-1] + y[-2])

        else:
            # use Simpson's rule on first intervals
            result = _basic_simpson(y, N - 3, x)

            # grab the last two spacings from the appropriate axis
            hm2 = slice(-2, -1, 1)
            hm1 = slice(-1, None, 1)

            diffs = jnp.diff(x)
            h0 = jnp.squeeze(diffs[hm2])
            h1 = jnp.squeeze(diffs[hm1])

            # This is the correction for the last interval according to
            # Cartwright.
            # However, I used the equations given at
            # https://en.wikipedia.org/wiki/Simpson%27s_rule#Composite_Simpson's_rule_for_irregularly_spaced_data
            # A footnote on Wikipedia says:
            # Cartwright 2017, Equation 8. The equation in Cartwright is
            # calculating the first interval whereas the equations in the
            # Wikipedia article are adjusting for the last integral. If the
            # proper algebraic substitutions are made, the equation results in
            # the values shown.
            num = 2 * h1**2 + 3 * h0 * h1
            den = 6 * (h1 + h0)
            alpha = jnp.where(den != 0, num / den, 0.0)

            num = h1**2 + 3.0 * h0 * h1
            den = 6 * h0
            beta = jnp.where(den != 0, num / den, 0.0)

            num = 1 * h1**3
            den = 6 * h0 * (h0 + h1)
            eta = jnp.where(den != 0, num / den, 0.0)

            result += alpha * y[-1] + beta * y[-2] - eta * y[-3]

    else:
        result = _basic_simpson(y, N - 2, x)

    return result


@partial(jax.jit, static_argnums=1)
def _basic_ln_simpson(ln_y, stop, x):
    """
    Note: Interface comes from scipy.integrate.simpson implementation
    """
    step = 2

    slice0 = slice(0, stop, step)
    slice1 = slice(1, stop + 1, step)
    slice2 = slice(2, stop + 2, step)

    # Account for possibly different spacings.
    #    Simpson's rule changes a bit.
    h = jnp.diff(x)
    h0 = h[slice0]
    h1 = h[slice1]
    hsum = h0 + h1
    hprod = h0 * h1
    h0divh1 = jnp.where(h1 != 0, h0 / h1, 0.0)

    term = logsumexp(
        jnp.array([ln_y[slice0], ln_y[slice1], ln_y[slice2]]),
        b=jnp.array(
            [
                (2.0 - jnp.where(h0divh1 != 0, 1 / h0divh1, 0.0)),
                (hsum * jnp.where(hprod != 0, hsum / hprod, 0.0)),
                (2.0 - h0divh1),
            ]
        ),
        axis=0,
    )
    tmp = jnp.log(hsum / 6.0) + term
    result = logsumexp(tmp)

    return result


@jax.jit
def ln_simpson(ln_y, x) -> float:
    """
    Integrate y(x) using log values of the function `ln_y` evaluated at the locations
    `x`, and return the log of the integral

    If there are an even number of samples, N, then there are an odd number of intervals
    (N-1), but Simpson's rule requires an even number of intervals. In this case, the
    implementation uses Simpson's rule for the first N-2 intervals with the addition of
    a 3-point parabolic segment for the last interval using equations outlined by
    Cartwright. See the docstring for `scipy.integrate.simpson` for more information.

    Note: `x` values must be increasing and `x` and `ln_y` must have the same length.

    Parameters
    ----------
    ln_y : array_like
        Array of log function values to be integrated.
    x : array_like
        The points at which `y` is evaluated.

    Returns
    -------
    float
        The estimated log integral computed with the composite Simpson's rule.
    """
    ln_y = jnp.array(ln_y)
    x = jnp.array(x)
    N = len(ln_y)

    if N % 2 == 0:  # Even number of points, odd intervals

        if N == 2:
            # need at least 3 points in integration axis to form parabolic segment
            last_dx = x[-1] - x[-2]
            result = jnp.log(0.5 * last_dx) + jnp.logaddexp(ln_y[-1], ln_y[-2])

        else:
            # use Simpson's rule on first intervals
            result = _basic_ln_simpson(ln_y, N - 3, x)

            # grab the last two spacings from the appropriate axis
            hm2 = slice(-2, -1, 1)
            hm1 = slice(-1, None, 1)

            diffs = jnp.diff(x)
            h0 = jnp.squeeze(diffs[hm2])
            h1 = jnp.squeeze(diffs[hm1])

            # This is the correction for the last interval according to
            # Cartwright.
            # However, I used the equations given at
            # https://en.wikipedia.org/wiki/Simpson%27s_rule#Composite_Simpson's_rule_for_irregularly_spaced_data
            # A footnote on Wikipedia says:
            # Cartwright 2017, Equation 8. The equation in Cartwright is
            # calculating the first interval whereas the equations in the
            # Wikipedia article are adjusting for the last integral. If the
            # proper algebraic substitutions are made, the equation results in
            # the values shown.
            num = 2 * h1**2 + 3 * h0 * h1
            den = 6 * (h1 + h0)
            alpha = jnp.where(den != 0, num / den, 0.0)

            num = h1**2 + 3.0 * h0 * h1
            den = 6 * h0
            beta = jnp.where(den != 0, num / den, 0.0)

            num = 1 * h1**3
            den = 6 * h0 * (h0 + h1)
            eta = jnp.where(den != 0, num / den, 0.0)

            term = logsumexp(ln_y[-3:], b=jnp.array([-eta, beta, alpha]), axis=0)
            result = jnp.logaddexp(result, term)

    else:
        result = _basic_ln_simpson(ln_y, N - 2, x)

    return result
