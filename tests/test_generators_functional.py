import numpy as np
import pytest

from xaitimesynth import (
    TimeSeriesBuilder,
    constant,
    ecg_like,
    gaussian,
    gaussian_pulse,
    manual,
    peak,
    pseudo_periodic,
    random_walk,
    red_noise,
    seasonal,
    trend,
    trough,
    uniform,
)

ALL_COMPONENT_FUNCS = [
    constant,
    seasonal,
    trend,
    manual,
    random_walk,
    gaussian,
    uniform,
    red_noise,
    peak,
    trough,
    gaussian_pulse,
    ecg_like,
    pseudo_periodic,
]


def _ones_generator(n_timesteps, rng, length, **kwargs):
    return np.ones(length)


@pytest.mark.parametrize("component_func", ALL_COMPONENT_FUNCS)
def test_all_components_build_as_feature(component_func):
    """Smoke test: every component can be used as a feature without raising."""
    n_timesteps = 500 if component_func == ecg_like else 100
    feature = (
        manual(generator=_ones_generator)
        if component_func == manual
        else component_func()
    )

    dataset = (
        TimeSeriesBuilder(
            n_timesteps=n_timesteps,
            n_samples=20,
            random_state=42,
            normalization="zscore",
        )
        .for_class(0)
        .add_signal(gaussian(sigma=0.1))
        .for_class(1)
        .add_signal(gaussian(sigma=0.1))
        .add_feature(feature, start_pct=0.3, end_pct=0.7)
        .build()
    )
    assert dataset is not None
