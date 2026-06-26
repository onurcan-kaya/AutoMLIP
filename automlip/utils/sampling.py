"""Scalar-or-range sampling.

A spec is either a single value or a (min, max) pair. A single value is fixed.
A (min, max) pair is drawn uniformly on each call. One convention covers every
sampled knob: density, composition, cell size, temperature, step counts and so
on. This module imports nothing from the rest of the package.
"""

from typing import Dict, List


def _is_range(spec) -> bool:
	return (
		isinstance(spec, (tuple, list))
		and not isinstance(spec, str)
		and len(spec) == 2
	)


def sample_scalar(spec, rng, as_int: bool = False):
	"""Return a value from a scalar-or-range spec.

	spec is a number (returned as is) or a (min, max) pair (drawn uniformly).
	With as_int the draw is an inclusive integer.
	"""
	if spec is None:
		return None
	if _is_range(spec):
		lo, hi = spec
		if lo > hi:
			raise ValueError("range minimum %r is greater than maximum %r"
							 % (lo, hi))
		if as_int:
			return int(rng.integers(int(lo), int(hi) + 1))
		return float(rng.uniform(lo, hi))
	return int(spec) if as_int else float(spec)


def sample_composition(composition: Dict, elements: List[str], rng) -> Dict:
	"""Draw a fraction per element from scalar-or-range specs, renormalise to 1.

	composition maps element to a scalar-or-range. Elements absent from the
	dict get 0. If every fraction is 0 the elements are split evenly.
	"""
	frac = {}
	for el in elements:
		val = sample_scalar(composition.get(el, 0.0), rng)
		frac[el] = val if val is not None else 0.0
	total = sum(frac.values())
	if total <= 0:
		return {el: 1.0 / len(elements) for el in elements}
	return {el: frac[el] / total for el in elements}
