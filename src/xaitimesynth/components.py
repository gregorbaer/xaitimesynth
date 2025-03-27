from typing import Any, Callable, Dict, List, Optional

import numpy as np

# Signal Components


def constant(value: float = 0.0, **kwargs) -> Dict[str, Any]:
    """Create a constant signal component.

    Args:
        value: The constant value.
        **kwargs: Additional parameters.

    Returns:
        Component definition dictionary.
    """
    return {"type": "constant", "value": value, **kwargs}


def random_walk(step_size: float = 0.1, **kwargs) -> Dict[str, Any]:
    """Create a random walk signal component.

    Args:
        step_size: Standard deviation of random steps.
        **kwargs: Additional parameters.

    Returns:
        Component definition dictionary.
    """
    return {"type": "random_walk", "step_size": step_size, **kwargs}


def autoregressive(
    coefficients: List[float] = [0.8], sigma: float = 1.0, **kwargs
) -> Dict[str, Any]:
    """Create an autoregressive signal component.

    Args:
        coefficients: AR coefficients.
        sigma: Noise standard deviation.
        **kwargs: Additional parameters.

    Returns:
        Component definition dictionary.
    """
    return {
        "type": "autoregressive",
        "coefficients": coefficients,
        "sigma": sigma,
        **kwargs,
    }


def gaussian(mu: float = 0.0, sigma: float = 0.1, **kwargs) -> Dict[str, Any]:
    """Create a Gaussian noise component.

    Args:
        mu: Mean of the Gaussian distribution.
        sigma: Standard deviation of the Gaussian distribution.
        **kwargs: Additional parameters.

    Returns:
        Component definition dictionary.
    """
    return {"type": "gaussian", "mu": mu, "sigma": sigma, **kwargs}


def uniform(low: float = -0.1, high: float = 0.1, **kwargs) -> Dict[str, Any]:
    """Create a uniform noise component.

    Args:
        low: Lower bound of the uniform distribution.
        high: Upper bound of the uniform distribution.
        **kwargs: Additional parameters.

    Returns:
        Component definition dictionary.
    """
    return {"type": "uniform", "low": low, "high": high, **kwargs}


def seasonal(period: int = 10, amplitude: float = 1.0, **kwargs) -> Dict[str, Any]:
    """Create a seasonal signal component.

    Args:
        period: Length of seasonal period.
        amplitude: Amplitude of seasonal pattern.
        **kwargs: Additional parameters.

    Returns:
        Component definition dictionary.
    """
    return {"type": "seasonal", "period": period, "amplitude": amplitude, **kwargs}


# Feature Components


def shapelet(scale: float = 1.0, **kwargs) -> Dict[str, Any]:
    """Create a shapelet feature component.

    Args:
        scale: Scale of the shapelet pattern.
        **kwargs: Additional parameters. Can include 'pattern' for custom shapelet.

    Returns:
        Component definition dictionary.
    """
    return {"type": "shapelet", "scale": scale, **kwargs}


def level_change(amplitude: float = 1.0, **kwargs) -> Dict[str, Any]:
    """Create a level change feature component.

    Args:
        amplitude: Amplitude of the level change.
        **kwargs: Additional parameters.

    Returns:
        Component definition dictionary.
    """
    return {"type": "level_change", "amplitude": amplitude, **kwargs}


def trend(slope: float = 0.1, **kwargs) -> Dict[str, Any]:
    """Create a trend feature component.

    Args:
        slope: Slope of the trend.
        **kwargs: Additional parameters.

    Returns:
        Component definition dictionary.
    """
    return {"type": "trend", "slope": slope, **kwargs}


def peak(amplitude: float = 1.0, width: int = 3, **kwargs) -> Dict[str, Any]:
    """Create a peak feature component.

    Args:
        amplitude: Amplitude of the peak.
        width: Width of the peak in timesteps.
        **kwargs: Additional parameters.

    Returns:
        Component definition dictionary.
    """
    return {"type": "peak", "amplitude": amplitude, "width": width, **kwargs}


def trough(amplitude: float = 1.0, width: int = 3, **kwargs) -> Dict[str, Any]:
    """Create a trough feature component.

    Args:
        amplitude: Amplitude of the trough (will be negated).
        width: Width of the trough in timesteps.
        **kwargs: Additional parameters.

    Returns:
        Component definition dictionary.
    """
    return {"type": "trough", "amplitude": amplitude, "width": width, **kwargs}


def time_frequency(
    frequency: float = 0.1, amplitude: float = 1.0, **kwargs
) -> Dict[str, Any]:
    """Create a time frequency feature component.

    Args:
        frequency: Frequency of the pattern.
        amplitude: Amplitude of the pattern.
        **kwargs: Additional parameters.

    Returns:
        Component definition dictionary.
    """
    return {
        "type": "time_frequency",
        "frequency": frequency,
        "amplitude": amplitude,
        **kwargs,
    }


def manual(
    values: Optional[np.ndarray] = None, generator: Optional[Callable] = None, **kwargs
) -> Dict[str, Any]:
    """Create a manual component from values or a generator function.

    Args:
        values: Array of values for the component.
        generator: Function that generates the component.
            Should accept n_timesteps, rng, and **kwargs.
        **kwargs: Additional parameters for the generator.

    Returns:
        Component definition dictionary.
    """
    component = {"type": "manual", **kwargs}

    if values is not None:
        component["values"] = values
    elif generator is not None:
        component["generator"] = generator
    else:
        raise ValueError("Either 'values' or 'generator' must be provided")

    return component
