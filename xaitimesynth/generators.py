from typing import Callable, List, Optional

import numpy as np


def generate_constant(
    n_timesteps: int,
    value: float = 0.0,
    rng: Optional[np.random.RandomState] = None,
    length: Optional[int] = None,
    **kwargs,
) -> np.ndarray:
    """Generate a time series with a constant value.

    Args:
        n_timesteps (int): The nominal length of the time series context. The actual output
            length is determined by the `length` parameter if provided, otherwise `n_timesteps`.
        value (float): The constant value to fill the series with. Defaults to 0.0.
        rng (Optional[np.random.RandomState]): Random number generator instance. Included for
            API consistency with other generators but unused here. Defaults to None.
        length (Optional[int]): The exact desired length of the output time series array.
            If provided, this overrides `n_timesteps`. If None, `n_timesteps` is used.
            Typically provided by the TimeSeriesBuilder. Defaults to None.
        **kwargs: Catches unused parameters passed by TimeSeriesBuilder for compatibility.

    Returns:
        np.ndarray: A 1D numpy array of the specified length filled with the constant value.

    Example:
        >>> generate_constant(n_timesteps=10, value=5.0)
        array([5., 5., 5., 5., 5., 5., 5., 5., 5., 5.])
        >>> generate_constant(n_timesteps=10, value=5.0, length=3)
        array([5., 5., 5.])
    """
    output_length = length if length is not None else n_timesteps
    return np.full(output_length, value)


def generate_random_walk(
    n_timesteps: int,
    step_size: float = 0.1,
    rng: Optional[np.random.RandomState] = None,
    length: Optional[int] = None,
    **kwargs,
) -> np.ndarray:
    """Generate a random walk time series.

    A random walk where each step is drawn from a normal distribution.

    Args:
        n_timesteps (int): The nominal length of the time series context. The actual output
            length is determined by the `length` parameter if provided, otherwise `n_timesteps`.
        step_size (float): Standard deviation of the random steps taken at each timestep.
            Defaults to 0.1.
        rng (Optional[np.random.RandomState]): Random number generator instance. If None,
            a default `np.random.RandomState()` is created. Typically provided by the
            TimeSeriesBuilder for reproducibility. Defaults to None.
        length (Optional[int]): The exact desired length of the output time series array.
            If provided, this overrides `n_timesteps`. If None, `n_timesteps` is used.
            Typically provided by the TimeSeriesBuilder. Defaults to None.
        **kwargs: Catches unused parameters passed by TimeSeriesBuilder for compatibility.

    Returns:
        np.ndarray: A 1D numpy array of the specified length representing a random walk.

    Example:
        >>> rng = np.random.RandomState(0)
        >>> generate_random_walk(n_timesteps=5, step_size=0.5, rng=rng)
        array([0.88202616, 1.0821086 , 1.3734961 , 2.49384918, 2.98797618])
    """
    if rng is None:
        rng = np.random.RandomState()
    output_length = length if length is not None else n_timesteps
    steps = rng.normal(0, step_size, output_length)
    return np.cumsum(steps)


def generate_gaussian(
    n_timesteps: int,
    mu: float = 0.0,
    sigma: float = 0.1,
    rng: Optional[np.random.RandomState] = None,
    length: Optional[int] = None,
    **kwargs,
) -> np.ndarray:
    """Generate a time series with Gaussian (normal) noise.

    Args:
        n_timesteps (int): The nominal length of the time series context. The actual output
            length is determined by the `length` parameter if provided, otherwise `n_timesteps`.
        mu (float): Mean of the Gaussian distribution. Defaults to 0.0.
        sigma (float): Standard deviation of the Gaussian distribution. Defaults to 0.1.
        rng (Optional[np.random.RandomState]): Random number generator instance. If None,
            a default `np.random.RandomState()` is created. Typically provided by the
            TimeSeriesBuilder for reproducibility. Defaults to None.
        length (Optional[int]): The exact desired length of the output time series array.
            If provided, this overrides `n_timesteps`. If None, `n_timesteps` is used.
            Typically provided by the TimeSeriesBuilder. Defaults to None.
        **kwargs: Catches unused parameters passed by TimeSeriesBuilder for compatibility.

    Returns:
        np.ndarray: A 1D numpy array of the specified length with values drawn from N(mu, sigma^2).

    Example:
        >>> rng = np.random.RandomState(1)
        >>> generate_gaussian(n_timesteps=5, mu=0, sigma=1, rng=rng)
        array([ 1.62434536, -0.61175641, -0.52817175, -1.07296862,  0.86540763])
    """
    if rng is None:
        rng = np.random.RandomState()
    output_length = length if length is not None else n_timesteps
    return rng.normal(mu, sigma, output_length)


def generate_uniform(
    n_timesteps: int,
    low: float = -0.1,
    high: float = 0.1,
    rng: Optional[np.random.RandomState] = None,
    length: Optional[int] = None,
    **kwargs,
) -> np.ndarray:
    """Generate a time series with uniform noise.

    Args:
        n_timesteps (int): The nominal length of the time series context. The actual output
            length is determined by the `length` parameter if provided, otherwise `n_timesteps`.
        low (float): Lower bound of the uniform distribution (inclusive). Defaults to -0.1.
        high (float): Upper bound of the uniform distribution (exclusive). Defaults to 0.1.
        rng (Optional[np.random.RandomState]): Random number generator instance. If None,
            a default `np.random.RandomState()` is created. Typically provided by the
            TimeSeriesBuilder for reproducibility. Defaults to None.
        length (Optional[int]): The exact desired length of the output time series array.
            If provided, this overrides `n_timesteps`. If None, `n_timesteps` is used.
            Typically provided by the TimeSeriesBuilder. Defaults to None.
        **kwargs: Catches unused parameters passed by TimeSeriesBuilder for compatibility.

    Returns:
        np.ndarray: A 1D numpy array of the specified length with values drawn from U(low, high).

    Example:
        >>> rng = np.random.RandomState(2)
        >>> generate_uniform(n_timesteps=5, low=0, high=1, rng=rng)
        array([0.43758721, 0.891773  , 0.96366276, 0.38344152, 0.79172504])
    """
    if rng is None:
        rng = np.random.RandomState()
    output_length = length if length is not None else n_timesteps
    return rng.uniform(low, high, output_length)


def generate_red_noise(
    n_timesteps: int,
    mean: float = 0.0,
    std: float = 1.0,
    phi: float = 0.9,
    rng: Optional[np.random.RandomState] = None,
    length: Optional[int] = None,
    **kwargs,
) -> np.ndarray:
    """Generate a red noise time series using an AR(1) process.

    Red noise exhibits positive autocorrelation, meaning successive values are
    likely to be close to each other, resulting in smoother, slower fluctuations
    compared to white noise.

    Args:
        n_timesteps (int): The nominal length of the time series context. The actual output
            length is determined by the `length` parameter if provided, otherwise `n_timesteps`.
        mean (float): The mean value around which the noise oscillates. Defaults to 0.0.
        std (float): The overall standard deviation (amplitude) of the noise process.
            Defaults to 1.0.
        phi (float): The autocorrelation coefficient (-1 < phi < 1). Controls the
            "memory" or smoothness. Values closer to 1 result in stronger positive
            autocorrelation (smoother noise). Defaults to 0.9.
        rng (Optional[np.random.RandomState]): Random number generator instance. If None,
            a default `np.random.RandomState()` is created. Typically provided by the
            TimeSeriesBuilder for reproducibility. Defaults to None.
        length (Optional[int]): The exact desired length of the output time series array.
            If provided, this overrides `n_timesteps`. If None, `n_timesteps` is used.
            Typically provided by the TimeSeriesBuilder. Defaults to None.
        **kwargs: Catches unused parameters passed by TimeSeriesBuilder for compatibility.

    Returns:
        np.ndarray: A 1D numpy array of the specified length representing red noise.

    Raises:
        ValueError: If phi is not strictly between -1 and 1.

    Example:
        >>> rng = np.random.RandomState(3)
        >>> generate_red_noise(n_timesteps=5, std=1, phi=0.8, rng=rng)
        array([-0.4891934 , -0.58890917, -0.4100916 , -0.08162811,  0.10116969])
    """
    if not -1 < phi < 1:
        raise ValueError("phi must be strictly between -1 and 1")
    if rng is None:
        rng = np.random.RandomState()

    output_length = length if length is not None else n_timesteps
    # Calculate the standard deviation of the white noise
    # Var(X_t) = Var(phi * X_{t-1} + eps_t) = phi^2 * Var(X_{t-1}) + Var(eps_t)
    # Assuming stationarity, Var(X_t) = Var(X_{t-1}) = std^2
    # std^2 = phi^2 * std^2 + Var(eps_t) => Var(eps_t) = std^2 * (1 - phi^2)
    epsilon_std = std * np.sqrt(1 - phi**2)
    epsilon = rng.normal(loc=0.0, scale=epsilon_std, size=output_length)

    # Initialize the series
    noise = np.zeros(output_length)
    # Start with the stationary mean adjusted by the first epsilon
    # Or start from a draw from the stationary distribution N(mean, std^2)
    noise[0] = rng.normal(loc=mean, scale=std)  # More robust start

    # Generate the AR(1) process
    for t in range(1, output_length):
        noise[t] = mean + phi * (noise[t - 1] - mean) + epsilon[t]

    return noise


def generate_seasonal(
    n_timesteps: int,
    period: int = 10,
    amplitude: float = 1.0,
    phase: float = 0.0,
    rng: Optional[np.random.RandomState] = None,
    length: Optional[int] = None,
    **kwargs,
) -> np.ndarray:
    """Generate a seasonal (sine wave) time series.

    The function generates a sine wave with the specified period, amplitude, and phase
    offset. To ensure the exact amplitude is achieved even for short periods where
    discrete sampling might not hit the peak values, the signal is automatically scaled
    when necessary.

    Args:
        n_timesteps (int): The nominal length of the time series context. The actual output
            length is determined by the `length` parameter if provided, otherwise `n_timesteps`.
        period (int): The number of timesteps in one full cycle of the sine wave.
            Defaults to 10.
        amplitude (float): The peak amplitude of the sine wave. The output is guaranteed
            to have exactly this maximum absolute value. Defaults to 1.0.
        phase (float): Phase offset in radians applied to the sine wave. Use ``math.pi / 2``
            to get a cosine wave. Defaults to 0.0.
        rng (Optional[np.random.RandomState]): Random number generator instance. Included for
            API consistency but unused in this deterministic generator. Defaults to None.
        length (Optional[int]): The exact desired length of the output time series array.
            If provided, this overrides `n_timesteps`. If None, `n_timesteps` is used.
            Typically provided by the TimeSeriesBuilder. Defaults to None.
        **kwargs: Catches unused parameters passed by TimeSeriesBuilder for compatibility.

    Returns:
        np.ndarray: A 1D numpy array of the specified length representing a seasonal pattern.
            The maximum absolute value is guaranteed to equal the specified amplitude.

    Note:
        For short periods relative to the signal length, discrete sampling of the continuous
        sine wave might not capture the exact peak values. This function automatically
        applies amplitude correction to ensure the specified amplitude is achieved.

    Example:
        >>> generate_seasonal(n_timesteps=5, period=4, amplitude=2)
        array([ 0.        ,  2.        ,  0.        , -2.        ,  0.        ])
        >>> # Short period example where scaling ensures exact amplitude
        >>> signal = generate_seasonal(n_timesteps=10, period=3, amplitude=1.0)
        >>> np.max(np.abs(signal))  # Will be exactly 1.0
        1.0
    """
    # rng is unused here
    output_length = length if length is not None else n_timesteps
    time = np.arange(output_length)

    # Generate the base sine wave with optional phase offset
    signal = amplitude * np.sin(2 * np.pi * time / period + phase)

    # For short periods, ensure we actually achieve the specified amplitude
    # by scaling if the actual maximum is significantly different
    actual_max = np.max(np.abs(signal))
    if actual_max > 0 and period <= output_length:
        # Only apply correction if we have a meaningful signal and period is reasonable
        # Use a small tolerance to avoid scaling due to floating point precision
        if abs(actual_max - amplitude) / amplitude > 1e-10:
            signal = signal * (amplitude / actual_max)

    return signal


def generate_trend(
    n_timesteps: int,
    slope: float = 0.1,
    endpoints: Optional[List[float]] = None,
    rng: Optional[np.random.RandomState] = None,
    length: Optional[int] = None,
    **kwargs,
) -> np.ndarray:
    """Generate a linear trend time series.

    Creates a series that increases or decreases linearly over time.
    The trend can be defined either by a slope (starting from 0) or by
    specifying the start and end values.

    Args:
        n_timesteps (int): The nominal length of the time series context. The actual output
            length is determined by the `length` parameter if provided, otherwise `n_timesteps`.
        slope (float): The slope of the trend (change per timestep). Used if `endpoints`
            is None. Assumes trend starts at 0. Defaults to 0.1.
        endpoints (Optional[List[float]]): A list or tuple `[start_value, end_value]`.
            If provided, the trend is generated between these values, and `slope` is ignored.
            Defaults to None.
        rng (Optional[np.random.RandomState]): Random number generator instance. Included for
            API consistency but unused in this deterministic generator. Defaults to None.
        length (Optional[int]): The exact desired length of the output time series array.
            If provided, this overrides `n_timesteps`. If None, `n_timesteps` is used.
            Typically provided by the TimeSeriesBuilder. Defaults to None.
        **kwargs: Catches unused parameters passed by TimeSeriesBuilder for compatibility.

    Returns:
        np.ndarray: A 1D numpy array of the specified length representing a linear trend.

    Example:
        >>> generate_trend(n_timesteps=5, slope=0.5)
        array([0. , 0.5, 1. , 1.5, 2. ])
        >>> generate_trend(n_timesteps=5, endpoints=[10, 12])
        array([10. , 10.5, 11. , 11.5, 12. ])
        >>> generate_trend(n_timesteps=5, endpoints=[10, 12], length=3)
        array([10., 11., 12.])
    """
    # rng is unused here
    output_length = length if length is not None else n_timesteps
    time = np.arange(output_length)

    if endpoints is not None:
        start_val, end_val = endpoints
        # Calculate slope based on endpoints and length
        # Avoid division by zero if length is 1
        if output_length > 1:
            slope = (end_val - start_val) / (output_length - 1)
        else:
            slope = 0  # Or handle as appropriate, maybe just return start_val?
            return np.array([start_val])
        intercept = start_val
    else:
        # Assume trend starts at 0 if only slope is given
        intercept = 0

    return intercept + slope * time


def generate_peak(
    n_timesteps: int,
    amplitude: float = 1.0,
    width: int = 3,
    rng: Optional[np.random.RandomState] = None,
    length: Optional[int] = None,
    **kwargs,
) -> np.ndarray:
    """Generate a time series with a single triangular peak centered within the length.

    Args:
        n_timesteps (int): The nominal length of the time series context. The actual output
            length is determined by the `length` parameter if provided, otherwise `n_timesteps`.
        amplitude (float): The height of the peak relative to the baseline (0). Defaults to 1.0.
        width (int): The width of the peak's base in timesteps. Should ideally be odd
            for a single central maximum point. Defaults to 3.
        rng (Optional[np.random.RandomState]): Random number generator instance. Included for
            API consistency but unused in this deterministic generator. Defaults to None.
        length (Optional[int]): The exact desired length of the output time series array.
            If provided, this overrides `n_timesteps`. If None, `n_timesteps` is used.
            Typically provided by the TimeSeriesBuilder. Defaults to None.
        **kwargs: Catches unused parameters passed by TimeSeriesBuilder for compatibility.

    Returns:
        np.ndarray: A 1D numpy array of the specified length containing a centered peak.

    Example:
        >>> generate_peak(n_timesteps=7, amplitude=1, width=5)
        array([0. , 0. , 0.5, 1. , 0.5, 0. , 0. ])
        >>> generate_peak(n_timesteps=7, amplitude=2, width=3, length=5)
        array([0. , 1. , 2. , 1. , 0. ])
    """
    # rng is unused here
    output_length = length if length is not None else n_timesteps
    peak_signal = np.zeros(output_length)

    if output_length == 0:
        return peak_signal

    center_index = output_length // 2
    half_width = width // 2

    start_index = max(0, center_index - half_width)
    # Ensure end_index doesn't exceed array bounds
    end_index = min(
        output_length, center_index + half_width + (width % 2)
    )  # +1 if width is odd

    # Create a simple triangular peak
    if width > 0:
        peak_values = np.linspace(
            0, amplitude, num=half_width + (width % 2), endpoint=True
        )
        if width % 2 == 1:  # Odd width, single max point
            # Ascending part
            peak_signal[start_index : center_index + 1] = peak_values
            # Descending part (reverse excluding the peak itself)
            if center_index + 1 < end_index:
                peak_signal[center_index + 1 : end_index] = peak_values[:-1][::-1]
        else:  # Even width, plateau
            # Ascending part: half_width - 1 values from 0 up to (not including) amplitude
            peak_signal[start_index : center_index - 1] = peak_values[:-1]
            # Plateau: two center points at amplitude
            peak_signal[center_index - 1 : center_index + 1] = amplitude
            # Descending part
            if center_index + 1 < end_index:
                peak_signal[center_index + 1 : end_index] = peak_values[:-1][::-1]

    # More sophisticated peak shapes could be implemented here (e.g., Gaussian)
    # Example: Gaussian peak
    # time = np.arange(output_length)
    # peak_signal = amplitude * np.exp(-((time - center_index)**2) / (2 * (width / 2.355)**2)) # FWHM approx

    return peak_signal


def generate_trough(
    n_timesteps: int,
    amplitude: float = 1.0,
    width: int = 3,
    rng: Optional[np.random.RandomState] = None,
    length: Optional[int] = None,
    **kwargs,
) -> np.ndarray:
    """Generate a time series with a single triangular trough centered within the length.

    This function simply generates a peak using `generate_peak` and negates the result.

    Args:
        n_timesteps (int): The nominal length of the time series context. The actual output
            length is determined by the `length` parameter if provided, otherwise `n_timesteps`.
        amplitude (float): The depth of the trough relative to the baseline (0). The generated
            peak will have this amplitude, and the result will be negated. Defaults to 1.0.
        width (int): The width of the trough's base in timesteps. Should ideally be odd
            for a single central minimum point. Defaults to 3.
        rng (Optional[np.random.RandomState]): Random number generator instance. Passed to
            `generate_peak` for API consistency. Defaults to None.
        length (Optional[int]): The exact desired length of the output time series array.
            If provided, this overrides `n_timesteps`. If None, `n_timesteps` is used.
            Typically provided by the TimeSeriesBuilder. Defaults to None.
        **kwargs: Catches unused parameters passed by TimeSeriesBuilder for compatibility.

    Returns:
        np.ndarray: A 1D numpy array of the specified length containing a centered trough.

    Example:
        >>> generate_trough(n_timesteps=7, amplitude=1, width=5)
        array([-0. , -0. , -0.5, -1. , -0.5, -0. , -0. ])
    """
    # Generate a peak and negate it. Pass rng along in case generate_peak changes.
    return -generate_peak(
        n_timesteps=n_timesteps,
        amplitude=amplitude,
        width=width,
        rng=rng,  # Pass rng along
        length=length,
        **kwargs,
    )


def generate_manual(
    n_timesteps: int,
    values: Optional[np.ndarray] = None,
    generator: Optional[Callable] = None,
    rng: Optional[np.random.RandomState] = None,
    length: Optional[int] = None,
    **kwargs,
) -> np.ndarray:
    """Generate a time series from provided values or a custom generator function.

    Allows direct specification of the time series values or using a custom function
    for generation, providing flexibility beyond the standard generators.

    Args:
        n_timesteps (int): The nominal length of the time series context. Used if `length`
            is None and `values` are not provided. Passed to `generator` if used.
        values (Optional[np.ndarray]): A numpy array of values to use directly for the
            time series. If provided, `generator` is ignored, and the length must match
            the required output length (`length` or `n_timesteps`). Defaults to None.
        generator (Optional[Callable]): A function to generate the values. Ignored if
            `values` is provided. The function should accept `n_timesteps`, `rng`,
            `length`, and `**kwargs` as arguments and return a numpy array of the
            specified `length`. Defaults to None.
        rng (Optional[np.random.RandomState]): Random number generator instance. Passed to
            `generator` if used. If None and `generator` is used, a default one is
            created. Defaults to None.
        length (Optional[int]): The exact desired length of the output time series array.
            If provided, this overrides `n_timesteps` and is passed to `generator`.
            If None, `n_timesteps` is used. Typically provided by the TimeSeriesBuilder.
            Defaults to None.
        **kwargs: Additional keyword arguments passed directly to the `generator` function.

    Returns:
        np.ndarray: A 1D numpy array of the specified length generated manually.

    Raises:
        ValueError: If neither `values` nor `generator` is provided.
        ValueError: If provided `values` do not match the required output length.

    Example:
        >>> manual_vals = np.array([1, 1, 0, 0, 1])
        >>> generate_manual(n_timesteps=5, values=manual_vals)
        array([1, 1, 0, 0, 1])
        >>> def custom_gen(n_timesteps, rng, length, **kwargs):
        ...     return np.linspace(0, kwargs.get('max_val', 1), length)
        >>> rng_ = np.random.RandomState(4)
        >>> generate_manual(n_timesteps=10, generator=custom_gen, rng=rng_, length=4, max_val=3)
        array([0., 1., 2., 3.])
    """
    output_length = length if length is not None else n_timesteps

    if values is not None:
        if len(values) != output_length:
            # Option 1: Raise error (strictest)
            raise ValueError(
                f"Provided 'values' length ({len(values)}) does not match required output length ({output_length})"
            )
            # Option 2: Truncate or pad (more flexible, but potentially hides issues)
            # result = np.zeros(output_length)
            # copy_len = min(len(values), output_length)
            # result[:copy_len] = values[:copy_len]
            # return result
        return np.array(values)  # Ensure it's a numpy array
    elif generator is not None:
        # Pass necessary arguments to the custom generator
        # Ensure rng is provided if generator needs it
        if rng is None:
            rng = np.random.RandomState()  # Create default if needed for generator
        return generator(
            n_timesteps=n_timesteps, rng=rng, length=output_length, **kwargs
        )
    else:
        raise ValueError(
            "Either 'values' or 'generator' must be provided for manual component"
        )


def generate_ecg_like(
    n_timesteps: int,
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
    rng: Optional[np.random.RandomState] = None,
    length: Optional[int] = None,
    **kwargs,
) -> np.ndarray:
    """Generate a synthetic ECG-like signal with physiologically plausible parameters.

    Creates a simulated electrocardiogram (ECG) signal mimicking the characteristic
    P-QRS-T wave pattern. Allows customization of various ECG components.

    Note:
        This simulation is for illustrative purposes and not medical diagnosis.

    Args:
        n_timesteps (int): The nominal length of the time series context. The actual output
            length is determined by the `length` parameter if provided, otherwise `n_timesteps`.
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
        rng (Optional[np.random.RandomState]): Random number generator instance. If None,
            a default `np.random.RandomState()` is created. Typically provided by the
            TimeSeriesBuilder for reproducibility. Defaults to None.
        length (Optional[int]): The exact desired length of the output time series array.
            If provided, this overrides `n_timesteps`. If None, `n_timesteps` is used.
            Typically provided by the TimeSeriesBuilder. Defaults to None.
        **kwargs: Catches unused parameters passed by TimeSeriesBuilder for compatibility.

    Returns:
        np.ndarray: A 1D numpy array of the specified length representing the synthetic ECG signal.

    Raises:
        ValueError: If heart_rate or sampling_rate result in a non-positive period.

    Example:
        >>> rng_ecg = np.random.RandomState(5)
        >>> # Generate 4 seconds of ECG data
        >>> ecg_signal = generate_ecg_like(n_timesteps=1000, sampling_rate=250, rng=rng_ecg, length=1000)
        >>> print(ecg_signal.shape)
        (1000,)
    """
    if rng is None:
        rng = np.random.RandomState()

    output_length = length if length is not None else n_timesteps

    # Calculate period in samples
    period_samples = int(60.0 / heart_rate * sampling_rate)
    if period_samples <= 0:
        raise ValueError("Heart rate and sampling rate result in non-positive period")

    # Number of heartbeats to generate
    n_beats = int(np.ceil(output_length / period_samples)) if period_samples > 0 else 0

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
    full_signal = np.zeros(output_length)
    if n_beats > 0:
        full_signal = np.tile(beat, n_beats)[:output_length]

    # Add heart rate variability
    if n_beats > 1 and hr_variability > 0 and period_samples > 0:
        modified_signal = np.zeros(output_length)
        current_pos = 0

        for i in range(n_beats):
            if i < n_beats - 1:
                period_var = period_samples * (
                    1 + rng.uniform(-hr_variability, hr_variability)
                )
                period_var = int(period_var)
            else:
                # Last beat fills remaining space
                period_var = output_length - current_pos

            # Ensure period is positive and doesn't exceed remaining length
            period_var = max(1, period_var)
            period_var = min(period_var, output_length - current_pos)

            if period_var > 0:
                # Stretch/compress the standard beat to the variable period length
                beat_stretched = np.interp(
                    np.linspace(0, 1, period_var),
                    np.linspace(0, 1, period_samples),
                    beat,
                )
                end_pos = current_pos + period_var
                modified_signal[current_pos:end_pos] = beat_stretched
                current_pos = end_pos

            if current_pos >= output_length:
                break

        full_signal = modified_signal

    # Add baseline wander (low frequency drift)
    if baseline_wander > 0:
        t = np.arange(output_length) / sampling_rate
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
        noise = rng.normal(0, noise_level, size=output_length)

        # Add occasional small artifacts
        if rng.uniform() < 0.3:  # 30% chance of artifact
            if output_length > 10:
                artifact_pos = rng.randint(0, output_length - 10)
                artifact_len = rng.randint(5, 15)
                artifact_amp = noise_level * rng.uniform(2, 4)
                noise[artifact_pos : artifact_pos + artifact_len] += (
                    artifact_amp * rng.normal(0, 1, artifact_len)
                )

        full_signal += noise

    return full_signal


def generate_gaussian_pulse(
    n_timesteps: int,
    amplitude: float = 1.0,
    width_ratio: float = 1.0,
    center: float = 0.5,
    rng: Optional[np.random.RandomState] = None,
    length: Optional[int] = None,
    **kwargs,
) -> np.ndarray:
    """Generate a Gaussian pulse time series.

    The pulse follows a Gaussian curve: amplitude * exp(-0.5 * ((x - center) / sigma)^2)
    The center is automatically snapped to the nearest discrete timestep to ensure
    exact amplitude and symmetric shape.

    Width is controlled using the 6-sigma rule: 99.7% of pulse energy falls within
    the specified width_ratio, with amplitude dropping to ~1% at boundaries.

    Args:
        n_timesteps (int): The nominal length of the time series context. The actual output
            length is determined by the `length` parameter if provided, otherwise `n_timesteps`.
        amplitude (float): Peak amplitude of the Gaussian pulse. The maximum value will be
            exactly this amplitude. Defaults to 1.0.
        width_ratio (float): Pulse width as fraction of the output length. Using the 6-sigma
            rule, 99.7% of the pulse energy will be contained within this fraction of the
            total length, with the pulse amplitude dropping to ~1% at the boundaries.
            Must be between 0.0 and 1.0. Defaults to 1.0.
        center (float): Desired peak position within the output length, ranging from 0.0 (start)
            to 1.0 (end). Will be snapped to the nearest discrete timestep to ensure exact
            amplitude and symmetric shape. Defaults to 0.5 (middle).
        rng (Optional[np.random.RandomState]): Random number generator instance. Included for
            API consistency with other generators but unused here. Defaults to None.
        length (Optional[int]): The exact desired length of the output time series array.
            If provided, this overrides `n_timesteps`. If None, `n_timesteps` is used.
            Typically provided by the TimeSeriesBuilder. Defaults to None.
        **kwargs: Catches unused parameters passed by TimeSeriesBuilder for compatibility.

    Returns:
        np.ndarray: A 1D numpy array containing the Gaussian pulse with exact amplitude.

    Raises:
        ValueError: If width_ratio is not between 0.0 and 1.0, or if center is not between
            0.0 and 1.0.

    Example:
        >>> pulse = generate_gaussian_pulse(n_timesteps=100, amplitude=2.0, center=0.7)
        >>> np.max(pulse)  # Exactly 2.0
        2.0
        >>> np.argmax(pulse)  # At position 70 (nearest to 0.7 * 99 = 69.3)
        70
    """
    # Validate parameters
    if not 0.0 <= width_ratio <= 1.0:
        raise ValueError(f"width_ratio must be between 0.0 and 1.0, got {width_ratio}")
    if not 0.0 <= center <= 1.0:
        raise ValueError(f"center must be between 0.0 and 1.0, got {center}")

    output_length = length if length is not None else n_timesteps

    if output_length <= 0:
        return np.array([])

    # Create time index array
    x = np.arange(output_length)

    # Convert relative center to absolute position and snap to nearest discrete timestep
    # This ensures the peak falls exactly on a timestep for mathematical precision
    center_pos = round(center * (output_length - 1)) if output_length > 1 else 0

    # Convert width_ratio to standard deviation using 6-sigma rule
    # This ensures 99.7% of the pulse energy is within the specified width
    sigma = (width_ratio * output_length) / 6.0

    # Handle edge case where sigma is very small
    if sigma <= 0:
        # If width is essentially zero, create a single spike at the center
        result = np.zeros(output_length)
        center_idx = int(round(center_pos))
        if 0 <= center_idx < output_length:
            result[center_idx] = amplitude
        return result

    # Generate Gaussian pulse
    gaussian_pulse = amplitude * np.exp(-0.5 * ((x - center_pos) / sigma) ** 2)

    return gaussian_pulse


def generate_pseudo_periodic(
    n_timesteps: int,
    period: float = 10.0,
    amplitude: float = 1.0,
    frequency_std: float = 0.05,
    amplitude_std: float = 0.1,
    rng: Optional[np.random.RandomState] = None,
    length: Optional[int] = None,
    **kwargs,
) -> np.ndarray:
    """Generate a pseudo-periodic signal with stochastic amplitude and frequency variations.

    Produces a sine wave where the instantaneous frequency and amplitude vary randomly
    at each timestep. This creates a more realistic background signal than a perfectly
    periodic sine wave — useful when simulating real-world data where oscillations are
    not exactly regular (e.g. respiration, heart rate, seasonal economic cycles).

    Ported and adapted from the ``PseudoPeriodic`` generator in the timesynth package
    (MIT License, https://github.com/TimeSynth/TimeSynth).

    Args:
        n_timesteps (int): The nominal length of the time series context. The actual output
            length is determined by the `length` parameter if provided, otherwise `n_timesteps`.
        period (float): Mean number of timesteps per full cycle. Defaults to 10.0.
        amplitude (float): Mean amplitude of the oscillation. Defaults to 1.0.
        frequency_std (float): Standard deviation of per-step frequency perturbations,
            expressed as a fraction of the base frequency (``1 / period``). Larger values
            produce more irregular periodicity. Defaults to 0.05.
        amplitude_std (float): Standard deviation of per-step amplitude perturbations.
            Larger values produce more variable amplitude. Defaults to 0.1.
        rng (Optional[np.random.RandomState]): Random number generator for reproducibility.
            If None, a fresh unseeded generator is used. Defaults to None.
        length (Optional[int]): The exact desired length of the output array.
            If provided, this overrides `n_timesteps`. Defaults to None.
        **kwargs: Catches unused parameters passed by TimeSeriesBuilder for compatibility.

    Returns:
        np.ndarray: A 1D numpy array of the specified length.

    Example:
        >>> rng = np.random.RandomState(0)
        >>> sig = generate_pseudo_periodic(n_timesteps=100, period=20, rng=rng)
        >>> len(sig)
        100
    """
    output_length = length if length is not None else n_timesteps
    if rng is None:
        rng = np.random.RandomState()

    base_freq = 1.0 / period
    freq_noise = rng.normal(0, frequency_std * base_freq, output_length)
    local_freq = base_freq + freq_noise

    amp_noise = rng.normal(0, amplitude_std, output_length)
    local_amp = amplitude + amp_noise

    cumulative_phase = 2 * np.pi * np.cumsum(local_freq)
    return local_amp * np.sin(cumulative_phase)


# Dictionary mapping component types to generator functions
#
# When adding a new generator:
# 1. Implement generate_XXX() function above
# 2. Add entry to this dictionary: "xxx": generate_xxx
# 3. Create XXX() component function in components.py
# 4. Register component in __init__.py with register_component()
# 5. Add to __all__ exports in __init__.py
# See docs/guides/adding_generators.md for the complete guide
GENERATOR_FUNCS = {
    "constant": generate_constant,
    "random_walk": generate_random_walk,
    "gaussian": generate_gaussian,
    "uniform": generate_uniform,
    "seasonal": generate_seasonal,
    "trend": generate_trend,
    "peak": generate_peak,
    "trough": generate_trough,
    "manual": generate_manual,
    "red_noise": generate_red_noise,
    "ecg_like": generate_ecg_like,
    "gaussian_pulse": generate_gaussian_pulse,
    "pseudo_periodic": generate_pseudo_periodic,
}


def generate_component(
    component_type: str, n_timesteps: int, rng: np.random.RandomState, **kwargs
) -> np.ndarray:
    """Generate a component vector using its registered generator function.

    This acts as a dispatcher to the specific `generate_...` functions based on the
    `component_type`.

    Args:
        component_type (str): The type of component to generate (e.g., 'constant', 'gaussian').
            Must be a key in the `GENERATOR_FUNCS` dictionary.
        n_timesteps (int): The nominal length of the time series context.
        rng (np.random.RandomState): Random number generator instance.
        **kwargs: Component-specific parameters, potentially including 'length', which will be
            passed to the underlying generator function.

    Returns:
        np.ndarray: The generated component vector.

    Raises:
        ValueError: If `component_type` is not found in `GENERATOR_FUNCS`, with a helpful
            message listing all available component types.
    """
    if component_type not in GENERATOR_FUNCS:
        available = ", ".join(sorted(GENERATOR_FUNCS.keys()))
        raise ValueError(
            f"Unknown component type: '{component_type}'. Available types: {available}"
        )

    # Pass n_timesteps positionally, pass rng and other kwargs by keyword
    return GENERATOR_FUNCS[component_type](n_timesteps, rng=rng, **kwargs)
