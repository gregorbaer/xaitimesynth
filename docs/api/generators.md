# Generators

Low-level generator functions that produce the actual time series arrays. These are dispatched automatically by the builder via the registry — most users interact with [components](components.md) instead.

Useful as a reference for the function signature contract when [writing custom generators](../guides/adding_generators.md).

## Dispatch

::: xaitimesynth.generators.generate_component

## Signal Generators

::: xaitimesynth.generators.generate_random_walk

::: xaitimesynth.generators.generate_gaussian_noise

::: xaitimesynth.generators.generate_uniform

::: xaitimesynth.generators.generate_red_noise

::: xaitimesynth.generators.generate_ecg_like

::: xaitimesynth.generators.generate_pseudo_periodic

## Feature Generators

::: xaitimesynth.generators.generate_peak

::: xaitimesynth.generators.generate_trough

::: xaitimesynth.generators.generate_gaussian_pulse

## Versatile Generators

::: xaitimesynth.generators.generate_constant

::: xaitimesynth.generators.generate_seasonal

::: xaitimesynth.generators.generate_trend

::: xaitimesynth.generators.generate_manual
