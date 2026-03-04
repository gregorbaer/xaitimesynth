"""Tests for generator functions in xaitimesynth.generators.

These are sanity checks ensuring generators meet basic contracts.
Detailed behavior is tested through builder integration tests.
"""

import numpy as np
import pytest

from xaitimesynth.generators import (
    generate_component,
    generate_constant,
    generate_ecg_like,
    generate_gaussian,
    generate_gaussian_pulse,
    generate_manual,
    generate_peak,
    generate_pseudo_periodic,
    generate_random_walk,
    generate_red_noise,
    generate_seasonal,
    generate_trend,
    generate_trough,
    generate_uniform,
)


@pytest.fixture
def rng():
    """Provide a seeded random state for reproducible tests."""
    return np.random.RandomState(42)


# All generators except manual (requires special args) and ecg_like (needs large n)
STANDARD_GENERATORS = [
    generate_constant,
    generate_random_walk,
    generate_gaussian,
    generate_uniform,
    generate_seasonal,
    generate_trend,
    generate_peak,
    generate_trough,
    generate_red_noise,
    generate_gaussian_pulse,
    generate_pseudo_periodic,
]

STOCHASTIC_GENERATORS = [
    generate_random_walk,
    generate_gaussian,
    generate_uniform,
    generate_red_noise,
    generate_pseudo_periodic,
]


# =============================================================================
# Parametrized sanity checks
# =============================================================================


@pytest.mark.parametrize("generator_func", STANDARD_GENERATORS)
def test_generator_returns_correct_length(generator_func, rng):
    """Generators return ndarray with correct length."""
    result = generator_func(n_timesteps=50, rng=rng)
    assert isinstance(result, np.ndarray)
    assert len(result) == 50


@pytest.mark.parametrize("generator_func", STANDARD_GENERATORS)
def test_generator_length_overrides_n_timesteps(generator_func, rng):
    """Length parameter overrides n_timesteps."""
    result = generator_func(n_timesteps=100, length=25, rng=rng)
    assert len(result) == 25


@pytest.mark.parametrize("generator_func", STANDARD_GENERATORS)
def test_generator_accepts_extra_kwargs(generator_func, rng):
    """Generators accept **kwargs without error."""
    result = generator_func(n_timesteps=50, rng=rng, unknown_param=42)
    assert len(result) == 50


@pytest.mark.parametrize("generator_func", STOCHASTIC_GENERATORS)
def test_stochastic_generator_reproducible(generator_func):
    """Stochastic generators produce same output with same seed."""
    rng1 = np.random.RandomState(123)
    rng2 = np.random.RandomState(123)

    result1 = generator_func(n_timesteps=50, rng=rng1)
    result2 = generator_func(n_timesteps=50, rng=rng2)

    np.testing.assert_array_equal(result1, result2)


# =============================================================================
# Special cases
# =============================================================================


def test_ecg_like_works_with_large_n(rng):
    """ECG-like generator works with sufficient n_timesteps."""
    # ecg_like has a bug with small n_timesteps, use larger value
    result = generate_ecg_like(n_timesteps=500, rng=rng)
    assert isinstance(result, np.ndarray)
    assert len(result) == 500


def test_manual_with_values():
    """Manual generator returns provided values."""
    values = np.array([1.0, 2.0, 3.0])
    result = generate_manual(n_timesteps=3, values=values)
    np.testing.assert_array_equal(result, values)


def test_manual_with_generator_function(rng):
    """Manual generator calls custom function."""

    def custom_gen(n_timesteps, rng, length, **kwargs):
        return np.ones(length) * kwargs.get("scale", 1.0)

    result = generate_manual(n_timesteps=10, generator=custom_gen, rng=rng, scale=5.0)
    np.testing.assert_array_equal(result, np.full(10, 5.0))


def test_manual_raises_without_values_or_generator():
    """Manual generator raises error if neither values nor generator provided."""
    with pytest.raises(ValueError, match="Either 'values' or 'generator'"):
        generate_manual(n_timesteps=10)


# =============================================================================
# seasonal phase parameter
# =============================================================================


def test_seasonal_phase_shifts_output(rng):
    """Phase parameter shifts seasonal output; cosine (phase=π/2) differs from sine (phase=0)."""
    import math

    sine = generate_seasonal(
        n_timesteps=50, period=20, amplitude=1.0, phase=0.0, rng=rng
    )
    cosine = generate_seasonal(
        n_timesteps=50, period=20, amplitude=1.0, phase=math.pi / 2, rng=rng
    )
    assert not np.allclose(sine, cosine)
    np.testing.assert_allclose(np.max(np.abs(sine)), np.max(np.abs(cosine)), rtol=0.05)


# =============================================================================
# Dispatcher
# =============================================================================


def test_generate_component_dispatches_correctly(rng):
    """generate_component calls correct generator."""
    result = generate_component("constant", n_timesteps=10, rng=rng, value=7.0)
    np.testing.assert_array_equal(result, np.full(10, 7.0))


def test_generate_component_unknown_type_raises(rng):
    """generate_component raises ValueError for unknown types."""
    with pytest.raises(ValueError, match="Unknown component type.*Available types"):
        generate_component("nonexistent", n_timesteps=10, rng=rng)
