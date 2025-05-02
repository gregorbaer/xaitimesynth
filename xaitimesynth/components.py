from typing import Any, Callable, Dict, List, Optional

import numpy as np


## Signal Components
def constant(value: float = 0.0, **kwargs) -> Dict[str, Any]:
    """Create a definition for a constant signal component.

    This component represents a time series with a constant value.

    Args:
        value (float): The constant value to fill the series with. Defaults to 0.0.
        **kwargs: Additional parameters passed to the generator during build time.

    Returns:
        Dict[str, Any]: A dictionary defining the 'constant' component with its parameters.
    """
    return {"type": "constant", "value": value, **kwargs}


def random_walk(step_size: float = 0.1, **kwargs) -> Dict[str, Any]:
    """Create a definition for a random walk signal component.

    This component represents a random walk where each step is drawn from a normal distribution.

    Args:
        step_size (float): Standard deviation of the random steps taken at each timestep.
            Defaults to 0.1.
        **kwargs: Additional parameters passed to the generator during build time.

    Returns:
        Dict[str, Any]: A dictionary defining the 'random_walk' component with its parameters.
    """
    return {"type": "random_walk", "step_size": step_size, **kwargs}


def gaussian(mu: float = 0.0, sigma: float = 1, **kwargs) -> Dict[str, Any]:
    """Create a definition for a Gaussian noise component.

    This component represents a time series with values drawn from a Gaussian (normal) distribution.

    Args:
        mu (float): Mean of the Gaussian distribution. Defaults to 0.0.
        sigma (float): Standard deviation of the Gaussian distribution. Defaults to 1.
        **kwargs: Additional parameters passed to the generator during build time.

    Returns:
        Dict[str, Any]: A dictionary defining the 'gaussian' component with its parameters.
    """
    return {"type": "gaussian", "mu": mu, "sigma": sigma, **kwargs}


def uniform(low: float = 0, high: float = 1, **kwargs) -> Dict[str, Any]:
    """Create a definition for a uniform noise component.

    This component represents a time series with values drawn from a uniform distribution.

    Args:
        low (float): Lower bound of the uniform distribution (inclusive). Defaults to 0.
        high (float): Upper bound of the uniform distribution (exclusive). Defaults to 1.
        **kwargs: Additional parameters passed to the generator during build time.

    Returns:
        Dict[str, Any]: A dictionary defining the 'uniform' component with its parameters.
    """
    return {"type": "uniform", "low": low, "high": high, **kwargs}


def seasonal(period: int = 10, amplitude: float = 1.0, **kwargs) -> Dict[str, Any]:
    """Create a definition for a seasonal (sine wave) signal component.

    This component represents a periodic pattern based on a sine wave.

    Args:
        period (int): The number of timesteps in one full cycle of the sine wave.
            Defaults to 10.
        amplitude (float): The peak amplitude of the sine wave. Defaults to 1.0.
        **kwargs: Additional parameters passed to the generator during build time.

    Returns:
        Dict[str, Any]: A dictionary defining the 'seasonal' component with its parameters.
    """
    return {"type": "seasonal", "period": period, "amplitude": amplitude, **kwargs}


def red_noise(
    mean: float = 0.0, std: float = 1.0, phi: float = 0.9, **kwargs
) -> Dict[str, Any]:
    """Create a definition for a red noise signal component using an AR(1) process.

    This component represents red noise, which exhibits positive autocorrelation,
    meaning successive values are likely to be close to each other, resulting in
    smoother, slower fluctuations compared to white noise.

    Args:
        mean (float): The mean value around which the noise oscillates. Defaults to 0.0.
        std (float): The overall standard deviation (amplitude) of the noise process.
            Defaults to 1.0.
        phi (float): The autocorrelation coefficient (-1 < phi < 1). Controls the
            "memory" or smoothness. Values closer to 1 result in stronger positive
            autocorrelation (smoother noise). Defaults to 0.9.
        **kwargs: Additional parameters passed to the generator during build time.

    Returns:
        Dict[str, Any]: A dictionary defining the 'red_noise' component with its parameters.
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
    """Create a definition for a synthetic ECG-like signal component.

    This component simulates an electrocardiogram (ECG) signal mimicking the characteristic
    P-QRS-T wave pattern, allowing customization of various ECG parameters.

    Note:
        This simulation is for illustrative purposes and not medical diagnosis.

    Args:
        heart_rate (float): Average heart rate in beats per minute (BPM). Defaults to 70.0.
        p_amplitude (float): Amplitude of the P wave (mV). Defaults to 0.15.
        qrs_amplitude (float): Amplitude of the QRS complex (mV). Defaults to 1.0.
        t_amplitude (float): Amplitude of the T wave (mV). Defaults to 0.3.
        p_width (float): Duration of the P wave (seconds). Defaults to 0.09.
        qrs_width (float): Duration of the QRS complex (seconds). Defaults to 0.08.
        t_width (float): Duration of the T wave (seconds). Defaults to 0.16.
        pr_interval (float): Time from P wave start to QRS start (seconds). Defaults to 0.16.
        st_segment (float): Duration of the isoelectric ST segment (seconds). Defaults to 0.1.
        noise_level (float): Standard deviation of additive Gaussian noise (mV). Defaults to 0.03.
        sampling_rate (float): Sampling frequency in Hz (samples per second). Defaults to 250.0.
        hr_variability (float): Factor controlling beat-to-beat interval variation (0-1).
            Defaults to 0.05.
        baseline_wander (float): Amplitude of low-frequency baseline drift (mV). Defaults to 0.02.
        **kwargs: Additional parameters passed to the generator during build time.

    Returns:
        Dict[str, Any]: A dictionary defining the 'ecg_like' component with its parameters.
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
    """Create a definition for a linear trend feature component.

    This component represents a feature that increases or decreases linearly over time.
    The trend can be defined either by a slope (starting from 0) or by
    specifying the start and end values.

    Args:
        slope (float): The slope of the trend (change per timestep). Used if `endpoints`
            is None. Assumes trend starts at 0. Defaults to 0.1.
        endpoints (Optional[List[float]]): A list or tuple `[start_value, end_value]`.
            If provided, the trend is generated between these values, and `slope` is ignored.
            Defaults to None.
        **kwargs: Additional parameters passed to the generator during build time.

    Returns:
        Dict[str, Any]: A dictionary defining the 'trend' component with its parameters.
    """
    return {"type": "trend", "slope": slope, "endpoints": endpoints, **kwargs}


# TODO: combine peak and trough into one?
# also, maybe use "value" instead of "amplitude" for consistency?
# also, maybe we can use "constant" signal function for peak/trough, to avoid code duplication?
def peak(amplitude: float = 1.0, width: int = 3, **kwargs) -> Dict[str, Any]:
    """Create a definition for a peak feature component.

    This component represents a single triangular peak centered within the feature's length.

    Args:
        amplitude (float): The height of the peak relative to the baseline (0). Defaults to 1.0.
        width (int): The width of the peak's base in timesteps. Should ideally be odd
            for a single central maximum point. Defaults to 3.
        **kwargs: Additional parameters passed to the generator during build time.

    Returns:
        Dict[str, Any]: A dictionary defining the 'peak' component with its parameters.
    """
    return {"type": "peak", "amplitude": amplitude, "width": width, **kwargs}


def trough(amplitude: float = 1.0, width: int = 3, **kwargs) -> Dict[str, Any]:
    """Create a definition for a trough feature component.

    This component represents a single triangular trough centered within the feature's length.
    It is generated by creating a peak with the given amplitude and negating it.

    Args:
        amplitude (float): The depth of the trough relative to the baseline (0). The generated
            peak will have this amplitude, and the result will be negated. Defaults to 1.0.
        width (int): The width of the trough's base in timesteps. Should ideally be odd
            for a single central minimum point. Defaults to 3.
        **kwargs: Additional parameters passed to the generator during build time.

    Returns:
        Dict[str, Any]: A dictionary defining the 'trough' component with its parameters.
    """
    return {"type": "trough", "amplitude": amplitude, "width": width, **kwargs}


def manual(
    values: Optional[np.ndarray] = None, generator: Optional[Callable] = None, **kwargs
) -> Dict[str, Any]:
    """Create a definition for a manual component from provided values or a custom generator function.

    Allows direct specification of the component's values or using a custom function
    for generation, providing flexibility beyond the standard components.

    Args:
        values (Optional[np.ndarray]): A numpy array of values to use directly for the
            component. If provided, `generator` is ignored. The length must match
            the required output length during generation. Defaults to None.
        generator (Optional[Callable]): A function to generate the values. Ignored if
            `values` is provided. The function should accept `n_timesteps`, `rng`,
            `length`, and `**kwargs` as arguments and return a numpy array of the
            specified `length`. Defaults to None.
        **kwargs: Additional keyword arguments passed directly to the `generator` function
            during build time, or stored if `values` are provided.

    Returns:
        Dict[str, Any]: A dictionary defining the 'manual' component with its parameters.

    Raises:
        ValueError: If neither `values` nor `generator` is provided.
    """
    component = {"type": "manual", **kwargs}

    if values is not None:
        component["values"] = values
    elif generator is not None:
        component["generator"] = generator
    else:
        raise ValueError("Either 'values' or 'generator' must be provided")

    return component
