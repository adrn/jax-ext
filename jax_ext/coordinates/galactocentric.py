from functools import partial

import jax
import jax.numpy as jnp

__all__ = ["GalactocentricCoordinateTransform"]


class GalactocentricCoordinateTransform:
    def __init__(self, galcen_frame):
        import astropy.units as u
        from astropy.coordinates.builtin_frames.galactocentric import get_matrix_vectors

        self.gc_M, gc_offset = get_matrix_vectors(galcen_frame)
        self.gc_dx = gc_offset.xyz.to_value(u.kpc)

    @partial(jax.jit, static_argnums=(0,))
    def icrs_to_galcen_cart(self, ra, dec, d):
        x_icrs = jnp.stack(
            (
                d * jnp.cos(ra) * jnp.cos(dec),
                d * jnp.sin(ra) * jnp.cos(dec),
                d * jnp.sin(dec),
            )
        )
        return self.gc_M @ x_icrs + self.gc_dx

    icrs_to_galcen_cart_vmap = jax.vmap(
        icrs_to_galcen_cart, in_axes=(None, 0, 0, 0), out_axes=1
    )

    @partial(jax.jit, static_argnums=(0,))
    def galcen_to_icrs_sph(self, x_gc):
        x_icrs = self.gc_M.T @ (x_gc - self.gc_dx)
        d = jnp.linalg.norm(x_icrs)
        ra = jnp.arctan2(x_icrs[1], x_icrs[0])
        dec = jnp.arctan2(x_icrs[2], jnp.sqrt(x_icrs[0] ** 2 + x_icrs[1] ** 2))
        return jnp.array([ra, dec, d])

    galcen_to_icrs_sph_vmap = jax.vmap(
        galcen_to_icrs_sph, in_axes=(None, 1), out_axes=1
    )
