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


def ecg_like(
    heart_rate: float = 70.0,
    p_amplitude: float = 0.15,
    qrs_amplitude: float = 1.0,
    t_amplitude: float = 0.3,
    p_width: float = 0.09,
    qrs_width: float = 0.08,
    t_width: float = 0.16,
    pr_interval: float = 0.16,
    st_segment: float = 0.1,
    noise_level: float = 0.03,
    sampling_rate: float = 250.0,
    hr_variability: float = 0.05,
    baseline_wander: float = 0.02,
    **kwargs,
) -> Dict[str, Any]:
    """Create an ECG-like signal component.

    Args:
        heart_rate: Heart rate in beats per minute (BPM). Normal resting is 60-100 BPM.
            Default is 70.0 (typical resting adult).
        p_amplitude: Amplitude of P wave in millivolts. Represents atrial depolarization.
            Default is 0.15 (typical range 0.1-0.2 mV).
        qrs_amplitude: Amplitude of the QRS complex in millivolts. Represents ventricular depolarization.
            Default is 1.0 (typical range 0.8-1.2 mV).
        t_amplitude: Amplitude of T wave in millivolts. Represents ventricular repolarization.
            Default is 0.3 (typical range 0.3-0.4 mV).
        p_width: Width/duration of P wave in seconds.
            Default is 0.09 (typical range 0.08-0.1 s).
        qrs_width: Width/duration of QRS complex in seconds.
            Default is 0.08 (typical range 0.06-0.1 s).
        t_width: Width/duration of T wave in seconds.
            Default is 0.16 (typical range 0.16-0.2 s).
        pr_interval: Interval between start of P wave and start of QRS complex in seconds.
            Default is 0.16 (typical range 0.12-0.2 s).
        st_segment: Duration of the ST segment between S wave and T wave in seconds.
            Default is 0.1 (typical range 0.08-0.12 s).
        noise_level: Amplitude of random noise in millivolts, simulating measurement noise.
            Default is 0.03.
        sampling_rate: Sampling rate in Hz (samples per second).
            Default is 250.0 (clinical standard).
        hr_variability: Heart rate variability factor (0-1). Higher values mean more variable beat intervals.
            Default is 0.05.
        baseline_wander: Magnitude of low-frequency baseline wandering in millivolts.
            Default is 0.02.
        **kwargs: Additional parameters.

    Returns:
        Component definition dictionary.
    """
    return {
        "type": "ecg_like",
        "heart_rate": heart_rate,
        "p_amplitude": p_amplitude,
        "qrs_amplitude": qrs_amplitude,
        "t_amplitude": t_amplitude,
        "p_width": p_width,
        "qrs_width": qrs_width,
        "t_width": t_width,
        "pr_interval": pr_interval,
        "st_segment": st_segment,
        "noise_level": noise_level,
        "sampling_rate": sampling_rate,
        "hr_variability": hr_variability,
        "baseline_wander": baseline_wander,
        **kwargs,
    }


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
