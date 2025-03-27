from typing import Callable, List, Optional

import numpy as np


def generate_constant(
    n_timesteps: int, rng: np.random.RandomState, value: float = 0.0, **kwargs
) -> np.ndarray:
    """Generate a constant signal.

    Args:
        n_timesteps: Length of time series.
        rng: Random number generator.
        value: Constant value.
        **kwargs: Additional parameters.

    Returns:
        Constant signal vector.
    """
    return np.full(n_timesteps, value)


def generate_random_walk(
    n_timesteps: int, rng: np.random.RandomState, step_size: float = 0.1, **kwargs
) -> np.ndarray:
    """Generate a random walk signal.

    Args:
        n_timesteps: Length of time series.
        rng: Random number generator.
        step_size: Standard deviation of random steps.
        **kwargs: Additional parameters.

    Returns:
        Random walk signal vector.
    """
    steps = rng.normal(0, step_size, n_timesteps)
    return np.cumsum(steps)


def generate_autoregressive(
    n_timesteps: int,
    rng: np.random.RandomState,
    coefficients: List[float] = [0.8],
    sigma: float = 1.0,
    **kwargs,
) -> np.ndarray:
    """Generate an autoregressive signal.

    Args:
        n_timesteps: Length of time series.
        rng: Random number generator.
        coefficients: AR coefficients.
        sigma: Noise standard deviation.
        **kwargs: Additional parameters.

    Returns:
        Autoregressive signal vector.
    """
    result = np.zeros(n_timesteps)
    p = len(coefficients)

    # Initialize with random values
    result[:p] = rng.normal(0, sigma, p)

    # Generate the autoregressive process
    for t in range(p, n_timesteps):
        result[t] = np.sum(
            [coefficients[i] * result[t - i - 1] for i in range(p)]
        ) + rng.normal(0, sigma)

    return result


def generate_gaussian(
    n_timesteps: int,
    rng: np.random.RandomState,
    mu: float = 0.0,
    sigma: float = 0.1,
    **kwargs,
) -> np.ndarray:
    """Generate a Gaussian noise signal.

    Args:
        n_timesteps: Length of time series.
        rng: Random number generator.
        mu: Mean of the Gaussian distribution.
        sigma: Standard deviation of the Gaussian distribution.
        **kwargs: Additional parameters.

    Returns:
        Gaussian noise vector.
    """
    return rng.normal(mu, sigma, n_timesteps)


def generate_uniform(
    n_timesteps: int,
    rng: np.random.RandomState,
    low: float = -0.1,
    high: float = 0.1,
    **kwargs,
) -> np.ndarray:
    """Generate a uniform noise signal.

    Args:
        n_timesteps: Length of time series.
        rng: Random number generator.
        low: Lower bound of the uniform distribution.
        high: Upper bound of the uniform distribution.
        **kwargs: Additional parameters.

    Returns:
        Uniform noise vector.
    """
    return rng.uniform(low, high, n_timesteps)


def generate_seasonal(
    n_timesteps: int,
    rng: np.random.RandomState,
    period: int = 10,
    amplitude: float = 1.0,
    **kwargs,
) -> np.ndarray:
    """Generate a seasonal signal.

    Args:
        n_timesteps: Length of time series.
        rng: Random number generator.
        period: Length of seasonal period.
        amplitude: Amplitude of seasonal pattern.
        **kwargs: Additional parameters.

    Returns:
        Seasonal signal vector.
    """
    t = np.arange(n_timesteps)
    return amplitude * np.sin(2 * np.pi * t / period)


def generate_shapelet(
    n_timesteps: int,
    rng: np.random.RandomState,
    length: int,
    scale: float = 1.0,
    pattern: Optional[np.ndarray] = None,
    **kwargs,
) -> np.ndarray:
    """Generate a shapelet feature.

    Args:
        n_timesteps: Length of time series.
        rng: Random number generator.
        length: Length of feature in timesteps.
        scale: Scale of the shapelet pattern.
        pattern: Custom pattern values. If None, generate a Gaussian bump.
        **kwargs: Additional parameters.

    Returns:
        Shapelet feature vector.
    """
    if pattern is not None:
        # Ensure pattern length matches feature length
        if len(pattern) != length:
            pattern = np.interp(
                np.linspace(0, 1, length), np.linspace(0, 1, len(pattern)), pattern
            )
    else:
        # Default shapelet is a bump
        t = np.linspace(-1, 1, length)
        pattern = np.exp(-5 * t**2)

    return scale * pattern


def generate_level_change(
    n_timesteps: int,
    rng: np.random.RandomState,
    length: int,
    amplitude: float = 1.0,
    **kwargs,
) -> np.ndarray:
    """Generate a level change feature.

    Args:
        n_timesteps: Length of time series.
        rng: Random number generator.
        length: Length of feature in timesteps.
        amplitude: Amplitude of the level change.
        **kwargs: Additional parameters.

    Returns:
        Level change feature vector.
    """
    return np.full(length, amplitude)


def generate_trend(
    n_timesteps: int,
    rng: np.random.RandomState,
    length: int,
    slope: float = 0.1,
    **kwargs,
) -> np.ndarray:
    """Generate a trend feature.

    Args:
        n_timesteps: Length of time series.
        rng: Random number generator.
        length: Length of feature in timesteps.
        slope: Slope of the trend.
        **kwargs: Additional parameters.

    Returns:
        Trend feature vector.
    """
    t = np.arange(length)
    return slope * t


def generate_peak(
    n_timesteps: int,
    rng: np.random.RandomState,
    length: int,
    amplitude: float = 1.0,
    width: int = 3,
    **kwargs,
) -> np.ndarray:
    """Generate a peak feature.

    Args:
        n_timesteps: Length of time series.
        rng: Random number generator.
        length: Length of feature in timesteps.
        amplitude: Amplitude of the peak.
        width: Width of the peak in timesteps.
        **kwargs: Additional parameters.

    Returns:
        Peak feature vector.
    """
    result = np.zeros(length)

    # Generate a peak centered in the feature region
    center_idx = length // 2
    half_width = min(width // 2, center_idx)

    peak_start = center_idx - half_width
    peak_end = center_idx + half_width + 1

    result[peak_start:peak_end] = amplitude

    return result


def generate_trough(
    n_timesteps: int,
    rng: np.random.RandomState,
    length: int,
    amplitude: float = 1.0,
    width: int = 3,
    **kwargs,
) -> np.ndarray:
    """Generate a trough feature.

    Args:
        n_timesteps: Length of time series.
        rng: Random number generator.
        length: Length of feature in timesteps.
        amplitude: Amplitude of the trough.
        width: Width of the trough in timesteps.
        **kwargs: Additional parameters.

    Returns:
        Trough feature vector.
    """
    return generate_peak(n_timesteps, rng, length, -amplitude, width, **kwargs)


def generate_time_frequency(
    n_timesteps: int,
    rng: np.random.RandomState,
    length: int,
    frequency: float = 0.1,
    amplitude: float = 1.0,
    **kwargs,
) -> np.ndarray:
    """Generate a time frequency feature.

    Args:
        n_timesteps: Length of time series.
        rng: Random number generator.
        length: Length of feature in timesteps.
        frequency: Frequency of the pattern.
        amplitude: Amplitude of the pattern.
        **kwargs: Additional parameters.

    Returns:
        Time frequency feature vector.
    """
    t = np.arange(length)
    return amplitude * np.sin(2 * np.pi * frequency * t / n_timesteps)


def generate_manual(
    n_timesteps: int,
    rng: np.random.RandomState,
    length: Optional[int] = None,
    values: Optional[np.ndarray] = None,
    generator: Optional[Callable] = None,
    **kwargs,
) -> np.ndarray:
    """Generate a manual component from values or a generator function.

    Args:
        n_timesteps: Length of time series.
        rng: Random number generator.
        length: Length of component in timesteps. If None, uses n_timesteps.
        values: Array of values for the component.
        generator: Function that generates the component.
            Should accept length, rng, and **kwargs.
        **kwargs: Additional parameters.

    Returns:
        Manual component vector.
    """
    # If length is not provided, use the entire time series length
    if length is None:
        length = n_timesteps

    if values is not None:
        # Ensure values length matches the required length
        if len(values) != length:
            return np.interp(
                np.linspace(0, 1, length), np.linspace(0, 1, len(values)), values
            )
        return values

    if generator is not None:
        return generator(length, rng, **kwargs)

    raise ValueError("Either values or generator must be provided")


# Dictionary mapping component types to generator functions
GENERATOR_FUNCS = {
    "constant": generate_constant,
    "random_walk": generate_random_walk,
    "autoregressive": generate_autoregressive,
    "gaussian": generate_gaussian,
    "uniform": generate_uniform,
    "seasonal": generate_seasonal,
    "shapelet": generate_shapelet,
    "level_change": generate_level_change,
    "trend": generate_trend,
    "peak": generate_peak,
    "trough": generate_trough,
    "time_frequency": generate_time_frequency,
    "manual": generate_manual,
}


def generate_component(
    component_type: str, n_timesteps: int, rng: np.random.RandomState, **kwargs
) -> np.ndarray:
    """Generate a component vector.

    Args:
        component_type: Type of component.
        n_timesteps: Length of time series.
        rng: Random number generator.
        **kwargs: Component parameters.

    Returns:
        Component vector.
    """
    if component_type not in GENERATOR_FUNCS:
        raise ValueError(f"Unknown component type: {component_type}")

    return GENERATOR_FUNCS[component_type](n_timesteps, rng, **kwargs)
