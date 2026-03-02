# Components

Component functions define the parameters for signal and feature generation. These are used with `add_signal()` and `add_feature()` on the `TimeSeriesBuilder`.

## Signal Components

Components typically used as background signals.

::: xaitimesynth.components.random_walk

::: xaitimesynth.components.gaussian

::: xaitimesynth.components.uniform

::: xaitimesynth.components.red_noise

::: xaitimesynth.components.ecg_like

## Feature Components

Components typically used as discriminative features.

::: xaitimesynth.components.peak

::: xaitimesynth.components.trough

::: xaitimesynth.components.gaussian_pulse

## Versatile Components

Components commonly used as either signals or features.

::: xaitimesynth.components.constant

::: xaitimesynth.components.seasonal

::: xaitimesynth.components.trend

::: xaitimesynth.components.manual
