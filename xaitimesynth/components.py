from typing import Any, Callable, Dict, List, Optional

import numpy as np


## Signal Components
def constant(value: float = 0.0, **kwargs) -> Dict[str, Any]:
    """Create a constant signal component.

    Args:
        value (float): The constant value.
        **kwargs: Additional parameters.

    Returns:
        Dict[str, Any]: Component definition dictionary.
    """
    return {"type": "constant", "value": value, **kwargs}


def random_walk(step_size: float = 0.1, **kwargs) -> Dict[str, Any]:
    """Create a random walk signal component.

    Args:
        step_size (float): Standard deviation of random steps.
        **kwargs: Additional parameters.

    Returns:
        Dict[str, Any]: Component definition dictionary.
    """
    return {"type": "random_walk", "step_size": step_size, **kwargs}


def autoregressive(
    coefficients: List[float] = [0.8], sigma: float = 1.0, **kwargs
) -> Dict[str, Any]:
    """Create an autoregressive signal component.

    Args:
        coefficients (List[float]): AR coefficients.
        sigma (float): Noise standard deviation.
        **kwargs: Additional parameters.

    Returns:
        Dict[str, Any]: Component definition dictionary.
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
        mu (float): Mean of the Gaussian distribution.
        sigma (float): Standard deviation of the Gaussian distribution.
        **kwargs: Additional parameters.

    Returns:
        Dict[str, Any]: Component definition dictionary.
    """
    return {"type": "gaussian", "mu": mu, "sigma": sigma, **kwargs}


def uniform(low: float = -0.1, high: float = 0.1, **kwargs) -> Dict[str, Any]:
    """Create a uniform noise component.

    Args:
        low (float): Lower bound of the uniform distribution.
        high (float): Upper bound of the uniform distribution.
        **kwargs: Additional parameters.

    Returns:
        Dict[str, Any]: Component definition dictionary.
    """
    return {"type": "uniform", "low": low, "high": high, **kwargs}


def seasonal(period: int = 10, amplitude: float = 1.0, **kwargs) -> Dict[str, Any]:
    """Create a seasonal signal component.

    Args:
        period (int): Length of seasonal period.
        amplitude (float): Amplitude of seasonal pattern.
        **kwargs: Additional parameters.

    Returns:
        Dict[str, Any]: Component definition dictionary.
    """
    return {"type": "seasonal", "period": period, "amplitude": amplitude, **kwargs}


def red_noise(
    mean: float = 0.0, std: float = 1.0, phi: float = 0.9, **kwargs
) -> Dict[str, Any]:
    """Create a red noise signal component using an AR(1) process.

    Generates noise where successive values are correlated. The strength and nature
    of this correlation are controlled by the `phi` parameter.

    Args:
        mean (float): The mean value around which the noise oscillates. Defaults to 0.0.
        std (float): The overall standard deviation (amplitude) of the noise. Defaults to 1.0.
        phi (float): The autocorrelation coefficient. Must be strictly between -1 and 1
            (-1 < phi < 1). Controls the "color" of the noise:
            - Positive phi (0 < phi < 1): Creates smoother, low-frequency dominant "red noise".
              Closer to 1 means stronger correlation and slower changes.
            - Negative phi (-1 < phi < 0): Creates rapidly oscillating, high-frequency dominant "blue noise".
              Closer to -1 means stronger anti-correlation.
            - phi = 0: Results in uncorrelated white noise.
            Defaults to 0.9 (strong red noise).
        **kwargs: Additional parameters.

    Returns:
        Dict[str, Any]: Component definition dictionary.
    """
    return {"type": "red_noise", "mean": mean, "std": std, "phi": phi, **kwargs}


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
        heart_rate (float): Heart rate in beats per minute (BPM). Normal resting is 60-100 BPM.
            Default is 70.0 (typical resting adult).
        p_amplitude (float): Amplitude of P wave in millivolts. Represents atrial depolarization.
            Default is 0.15 (typical range 0.1-0.2 mV).
        qrs_amplitude (float): Amplitude of the QRS complex in millivolts. Represents ventricular depolarization.
            Default is 1.0 (typical range 0.8-1.2 mV).
        t_amplitude (float): Amplitude of T wave in millivolts. Represents ventricular repolarization.
            Default is 0.3 (typical range 0.3-0.4 mV).
        p_width (float): Width/duration of P wave in seconds.
            Default is 0.09 (typical range 0.08-0.1 s).
        qrs_width (float): Width/duration of QRS complex in seconds.
            Default is 0.08 (typical range 0.06-0.1 s).
        t_width (float): Width/duration of T wave in seconds.
            Default is 0.16 (typical range 0.16-0.2 s).
        pr_interval (float): Interval between start of P wave and start of QRS complex in seconds.
            Default is 0.16 (typical range 0.12-0.2 s).
        st_segment (float): Duration of the ST segment between S wave and T wave in seconds.
            Default is 0.1 (typical range 0.08-0.12 s).
        noise_level (float): Amplitude of random noise in millivolts, simulating measurement noise.
            Default is 0.03.
        sampling_rate (float): Sampling rate in Hz (samples per second).
            Default is 250.0 (clinical standard).
        hr_variability (float): Heart rate variability factor (0-1). Higher values mean more variable beat intervals.
            Default is 0.05.
        baseline_wander (float): Magnitude of low-frequency baseline wandering in millivolts.
            Default is 0.02.
        **kwargs: Additional parameters.

    Returns:
        Dict[str, Any]: Component definition dictionary.
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


## Feature Components
def trend(
    slope: float = 0.1, endpoints: Optional[List[float]] = None, **kwargs
) -> Dict[str, Any]:
    """Create a trend feature component.

    Args:
        slope (float): Slope of the trend.
        endpoints (Optional[List[float]]): List containing [start_value, end_value].
            If provided, overrides the slope parameter.
        **kwargs: Additional parameters.

    Returns:
        Dict[str, Any]: Component definition dictionary.
    """
    return {"type": "trend", "slope": slope, "endpoints": endpoints, **kwargs}


# TODO: combine peak and trough into one?
# also, maybe use "value" instead of "amplitude" for consistency?
# also, maybe we can use "constant" signal function for peak/trough, to avoid code duplication?
def peak(amplitude: float = 1.0, width: int = 3, **kwargs) -> Dict[str, Any]:
    """Create a peak feature component.

    Args:
        amplitude (float): Amplitude of the peak.
        width (int): Width of the peak in timesteps.
        **kwargs: Additional parameters.

    Returns:
        Dict[str, Any]: Component definition dictionary.
    """
    return {"type": "peak", "amplitude": amplitude, "width": width, **kwargs}


def trough(amplitude: float = 1.0, width: int = 3, **kwargs) -> Dict[str, Any]:
    """Create a trough feature component.

    Args:
        amplitude (float): Amplitude of the trough (will be negated).
        width (int): Width of the trough in timesteps.
        **kwargs: Additional parameters.

    Returns:
        Dict[str, Any]: Component definition dictionary.
    """
    return {"type": "trough", "amplitude": amplitude, "width": width, **kwargs}


def manual(
    values: Optional[np.ndarray] = None, generator: Optional[Callable] = None, **kwargs
) -> Dict[str, Any]:
    """Create a manual component from values or a generator function.

    Args:
        values (Optional[np.ndarray]): Array of values for the component.
        generator (Optional[Callable]): Function that generates the component.
            Should accept n_timesteps, rng, and **kwargs.
        **kwargs: Additional parameters for the generator.

    Returns:
        Dict[str, Any]: Component definition dictionary.
    """
    component = {"type": "manual", **kwargs}

    if values is not None:
        component["values"] = values
    elif generator is not None:
        component["generator"] = generator
    else:
        raise ValueError("Either 'values' or 'generator' must be provided")

    return component
