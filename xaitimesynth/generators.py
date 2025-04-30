from typing import Callable, List, Optional

import numpy as np


def generate_constant(
    n_timesteps: int,
    rng: np.random.RandomState,
    length: Optional[int] = None,
    value: float = 0.0,
    **kwargs,
) -> np.ndarray:
    """Generate a constant signal.

    Args:
        n_timesteps (int): Length of time series.
        rng (np.random.RandomState): Random number generator.
        length (Optional[int]): Length of feature in timesteps. If None, uses n_timesteps.
        value (float): Constant value.
        **kwargs: Additional parameters.

    Returns:
        np.ndarray: Constant signal vector.
    """
    # If length is not provided, use the entire time series length
    if length is None:
        length = n_timesteps

    return np.full(length, value)


def generate_random_walk(
    n_timesteps: int,
    rng: np.random.RandomState,
    length: Optional[int] = None,
    step_size: float = 0.1,
    **kwargs,
) -> np.ndarray:
    """Generate a random walk signal.

    Args:
        n_timesteps (int): Length of time series.
        rng (np.random.RandomState): Random number generator.
        length (Optional[int]): Length of feature in timesteps. If None, uses n_timesteps.
        step_size (float): Standard deviation of random steps.
        **kwargs: Additional parameters.

    Returns:
        np.ndarray: Random walk signal vector.
    """
    # If length is not provided, use the entire time series length
    if length is None:
        length = n_timesteps

    steps = rng.normal(0, step_size, length)
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
        n_timesteps (int): Length of time series.
        rng (np.random.RandomState): Random number generator.
        coefficients (List[float]): AR coefficients.
        sigma (float): Noise standard deviation.
        **kwargs: Additional parameters.

    Returns:
        np.ndarray: Autoregressive signal vector.
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
    length: Optional[int] = None,
    mu: float = 0.0,
    sigma: float = 0.1,
    **kwargs,
) -> np.ndarray:
    """Generate a Gaussian noise signal.

    Args:
        n_timesteps (int): Length of time series.
        rng (np.random.RandomState): Random number generator.
        length (Optional[int]): Length of feature in timesteps. If None, uses n_timesteps.
        mu (float): Mean of the Gaussian distribution.
        sigma (float): Standard deviation of the Gaussian distribution.
        **kwargs: Additional parameters.

    Returns:
        np.ndarray: Gaussian noise vector.
    """
    # If length is not provided, use the entire time series length
    if length is None:
        length = n_timesteps

    return rng.normal(mu, sigma, length)


def generate_uniform(
    n_timesteps: int,
    rng: np.random.RandomState,
    length: Optional[int] = None,
    low: float = -0.1,
    high: float = 0.1,
    **kwargs,
) -> np.ndarray:
    """Generate a uniform noise signal.

    Args:
        n_timesteps (int): Length of time series.
        rng (np.random.RandomState): Random number generator.
        length (Optional[int]): Length of feature in timesteps. If None, uses n_timesteps.
        low (float): Lower bound of the uniform distribution.
        high (float): Upper bound of the uniform distribution.
        **kwargs: Additional parameters.

    Returns:
        np.ndarray: Uniform noise vector.
    """
    # If length is not provided, use the entire time series length
    if length is None:
        length = n_timesteps

    return rng.uniform(low, high, length)


def generate_seasonal(
    n_timesteps: int,
    rng: np.random.RandomState,
    length: Optional[int] = None,
    period: int = 10,
    amplitude: float = 1.0,
    **kwargs,
) -> np.ndarray:
    """Generate a seasonal signal.

    Args:
        n_timesteps (int): Length of time series.
        rng (np.random.RandomState): Random number generator.
        length (Optional[int]): Length of feature in timesteps. If None, uses n_timesteps.
        period (int): Length of seasonal period.
        amplitude (float): Amplitude of seasonal pattern.
        **kwargs: Additional parameters.

    Returns:
        np.ndarray: Seasonal signal vector.
    """
    # If length is not provided, use the entire time series length
    if length is None:
        length = n_timesteps

    t = np.arange(length)
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
        n_timesteps (int): Length of time series.
        rng (np.random.RandomState): Random number generator.
        length (int): Length of feature in timesteps.
        scale (float): Scale of the shapelet pattern.
        pattern (Optional[np.ndarray]): Custom pattern values. If None, generate a Gaussian bump.
        **kwargs: Additional parameters.

    Returns:
        np.ndarray: Shapelet feature vector.
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
        n_timesteps (int): Length of time series.
        rng (np.random.RandomState): Random number generator.
        length (int): Length of feature in timesteps.
        amplitude (float): Amplitude of the level change.
        **kwargs: Additional parameters.

    Returns:
        np.ndarray: Level change feature vector.
    """
    return np.full(length, amplitude)


def generate_trend(
    n_timesteps: int,
    rng: np.random.RandomState,
    length: int,
    slope: float = 0.1,
    endpoints: Optional[List[float]] = None,
    **kwargs,
) -> np.ndarray:
    """Generate a trend feature.

    Args:
        n_timesteps (int): Length of time series.
        rng (np.random.RandomState): Random number generator.
        length (int): Length of feature in timesteps.
        slope (float): Slope of the trend.
        endpoints (Optional[List[float]]): List containing [start_value, end_value].
            If provided, overrides the slope parameter.
        **kwargs: Additional parameters.

    Returns:
        np.ndarray: Trend feature vector.
    """
    t = np.arange(length)

    if endpoints is not None:
        if len(endpoints) != 2:
            raise ValueError(
                "endpoints must be a list of exactly two values: [start_value, end_value]"
            )
        start_value, end_value = endpoints

        if length <= 1:
            return np.full(length, start_value)

        # Create a linear trend from start_value to end_value
        return np.linspace(start_value, end_value, length)
    else:
        # If only slope is provided, create a linear trend starting from 0
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
        n_timesteps (int): Length of time series.
        rng (np.random.RandomState): Random number generator.
        length (int): Length of feature in timesteps.
        amplitude (float): Amplitude of the peak.
        width (int): Width of the peak in timesteps.
        **kwargs: Additional parameters.

    Returns:
        np.ndarray: Peak feature vector.
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
        n_timesteps (int): Length of time series.
        rng (np.random.RandomState): Random number generator.
        length (int): Length of feature in timesteps.
        amplitude (float): Amplitude of the trough.
        width (int): Width of the trough in timesteps.
        **kwargs: Additional parameters.

    Returns:
        np.ndarray: Trough feature vector.
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
        n_timesteps (int): Length of time series.
        rng (np.random.RandomState): Random number generator.
        length (int): Length of feature in timesteps.
        frequency (float): Frequency of the pattern.
        amplitude (float): Amplitude of the pattern.
        **kwargs: Additional parameters.

    Returns:
        np.ndarray: Time frequency feature vector.
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
        n_timesteps (int): Length of time series.
        rng (np.random.RandomState): Random number generator.
        length (Optional[int]): Length of component in timesteps. If None, uses n_timesteps.
        values (Optional[np.ndarray]): Array of values for the component.
        generator (Optional[Callable]): Function that generates the component.
            Should accept length, rng, and **kwargs.
        **kwargs: Additional parameters.

    Returns:
        np.ndarray: Manual component vector.
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


def generate_ecg_like(
    n_timesteps: int,
    rng: np.random.RandomState,
    length: Optional[int] = None,
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
) -> np.ndarray:
    """Generate a synthetic ECG-like signal with physiologically plausible parameters.

    Creates a simulated electrocardiogram (ECG) signal that mimics the characteristic
    P-QRS-T wave pattern seen in normal cardiac electrical activity. This function allows
    customization of various ECG components to model different cardiac conditions.

    Note:
        This simulation is for illustrative and testing purposes only and is not intended
        to replicate actual medical data. It should not be used for clinical decision making
        or medical research.

    Args:
        n_timesteps (int): Total length of the time series in samples.
        rng (np.random.RandomState): Random number generator instance for reproducibility.
        length (Optional[int]): Length of feature in timesteps. If None, uses n_timesteps.
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
        np.ndarray: Synthetic ECG signal of specified length.

    Example:
        >>> rng = np.random.RandomState(42)
        >>> # Normal adult ECG at rest
        >>> ecg = generate_ecg_like(1000, rng, heart_rate=65)
        >>> # Tachycardia (fast heart rate)
        >>> tachy_ecg = generate_ecg_like(1000, rng, heart_rate=120)
    """
    # If length is not provided, use the entire time series length
    if length is None:
        length = n_timesteps

    # Calculate period in samples
    period_samples = int(60.0 / heart_rate * sampling_rate)

    # Number of heartbeats to generate
    n_beats = int(np.ceil(length / period_samples))

    # Create time array for a single beat
    beat_time = np.arange(period_samples) / sampling_rate

    # Create a single heartbeat
    beat = np.zeros(period_samples)

    # Convert interval specifications to time positions
    p_center = pr_interval - (p_width / 2)  # P wave center
    qrs_center = pr_interval + (qrs_width / 2)  # QRS complex center
    t_center = qrs_center + qrs_width / 2 + st_segment + t_width / 2  # T wave center

    # P wave (atrial depolarization)
    p_sigma = p_width / 3.0  # 3 sigma spans approximately the width
    beat += p_amplitude * np.exp(-((beat_time - p_center) ** 2) / (2 * p_sigma**2))

    # QRS complex (ventricular depolarization)
    q_center = qrs_center - qrs_width / 3
    r_center = qrs_center
    s_center = qrs_center + qrs_width / 3
    qrs_sigma = qrs_width / 6.0

    # Q wave (small negative deflection)
    beat -= (
        qrs_amplitude
        * 0.25
        * np.exp(-((beat_time - q_center) ** 2) / (2 * (qrs_sigma * 0.8) ** 2))
    )

    # R wave (large positive deflection)
    beat += qrs_amplitude * np.exp(-((beat_time - r_center) ** 2) / (2 * qrs_sigma**2))

    # S wave (negative deflection after R)
    beat -= (
        qrs_amplitude
        * 0.35
        * np.exp(-((beat_time - s_center) ** 2) / (2 * (qrs_sigma * 0.8) ** 2))
    )

    # T wave (ventricular repolarization)
    t_sigma = t_width / 3.0
    beat += t_amplitude * np.exp(-((beat_time - t_center) ** 2) / (2 * t_sigma**2))

    # Generate full signal by repeating the beat
    full_signal = np.tile(beat, n_beats)[:length]

    # Add heart rate variability
    if n_beats > 1 and hr_variability > 0:
        # Slightly modify periods between consecutive beats
        modified_signal = np.zeros(length)
        current_pos = 0

        for i in range(n_beats):
            # Add variability to period length
            if i < n_beats - 1:
                period_var = period_samples * (
                    1 + rng.uniform(-hr_variability, hr_variability)
                )
                period_var = int(period_var)
            else:
                period_var = length - current_pos

            # Ensure we don't exceed the signal length
            period_var = min(period_var, length - current_pos)

            # Extract and stretch/compress the beat
            if period_var > 0:
                beat_stretched = np.interp(
                    np.linspace(0, 1, period_var),
                    np.linspace(0, 1, period_samples),
                    beat,
                )
                modified_signal[current_pos : current_pos + period_var] = beat_stretched
                current_pos += period_var

            # Break if we've filled the signal
            if current_pos >= length:
                break

        full_signal = modified_signal

    # Add baseline wander (low frequency drift)
    if baseline_wander > 0:
        # Create a slow-moving sine wave for baseline drift
        t = np.arange(length) / sampling_rate
        wander_freq1 = 0.05  # ~respiratory frequency (0.05 Hz)
        wander_freq2 = 0.01  # very slow drift
        baseline = baseline_wander * np.sin(
            2 * np.pi * wander_freq1 * t
        ) + baseline_wander * 0.5 * np.sin(
            2 * np.pi * wander_freq2 * t + rng.uniform(0, 2 * np.pi)
        )
        full_signal += baseline

    # Add measurement noise
    if noise_level > 0:
        # High-frequency measurement noise
        noise = rng.normal(0, noise_level, size=length)

        # Add occasional small artifacts
        if rng.uniform() < 0.3:  # 30% chance of artifact
            artifact_pos = rng.randint(0, length - 10)
            artifact_len = rng.randint(5, 15)
            artifact_amp = noise_level * rng.uniform(2, 4)
            noise[artifact_pos : artifact_pos + artifact_len] += (
                artifact_amp * rng.normal(0, 1, artifact_len)
            )

        full_signal += noise

    return full_signal


def generate_red_noise(
    n_timesteps: int,
    rng: np.random.RandomState,
    length: Optional[int] = None,
    mean: float = 0.0,
    std: float = 1.0,
    phi: float = 0.9,
    **kwargs,
) -> np.ndarray:
    """Generate a red noise signal using an AR(1) process.

    Uses an Autoregressive model of order 1 (AR(1)):
    X_t = mean + phi * (X_{t-1} - mean) + epsilon_t
    where epsilon_t is white noise N(0, sigma_epsilon^2)
    and sigma_epsilon = std * sqrt(1 - phi^2) to ensure the stationary
    variance of X_t is std^2.

    Args:
        n_timesteps (int): Length of the full time series.
        rng (np.random.RandomState): Random number generator.
        length (Optional[int]): Length of the component. If None, uses n_timesteps.
        mean (float): Mean of the noise process. Defaults to 0.0.
        std (float): Standard deviation of the noise process. Defaults to 1.0.
        phi (float): Autocorrelation coefficient (-1 < phi < 1).
            Controls the noise "color". Positive phi -> red noise (smoother),
            negative phi -> blue noise (oscillating), phi=0 -> white noise.
            Defaults to 0.9 (strong red noise).
        **kwargs: Additional parameters (ignored).

    Returns:
        np.ndarray: Red noise signal vector.

    Raises:
        ValueError: If phi is not strictly between -1 and 1.
    """
    if not -1 < phi < 1:
        raise ValueError(
            "phi (autocorrelation coefficient) must be strictly between -1 and 1"
        )

    effective_length = length if length is not None else n_timesteps

    # Calculate std dev for the white noise component
    variance_epsilon = max(0, std**2 * (1 - phi**2))
    std_epsilon = np.sqrt(variance_epsilon)

    # Generate white noise
    epsilon = rng.normal(loc=0.0, scale=std_epsilon, size=effective_length)

    # Initialize output array
    red_noise = np.zeros(effective_length)

    # Set the first value using the stationary distribution
    red_noise[0] = mean + rng.normal(loc=0.0, scale=std)

    # Generate the AR(1) process iteratively
    for t in range(1, effective_length):
        red_noise[t] = mean + phi * (red_noise[t - 1] - mean) + epsilon[t]

    return red_noise


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
    "ecg_like": generate_ecg_like,
    "red_noise": generate_red_noise,
}


def generate_component(
    component_type: str, n_timesteps: int, rng: np.random.RandomState, **kwargs
) -> np.ndarray:
    """Generate a component vector.

    Args:
        component_type (str): Type of component.
        n_timesteps (int): Length of time series.
        rng (np.random.RandomState): Random number generator.
        **kwargs: Component parameters.

    Returns:
        np.ndarray: Component vector.
    """
    if component_type not in GENERATOR_FUNCS:
        raise ValueError(f"Unknown component type: {component_type}")

    return GENERATOR_FUNCS[component_type](n_timesteps, rng, **kwargs)
