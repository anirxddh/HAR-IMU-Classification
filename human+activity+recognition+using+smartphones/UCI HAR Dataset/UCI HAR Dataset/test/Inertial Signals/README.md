# Test Inertial Signals

## Purpose

This directory stores raw 128-sample inertial signal windows for the original test partition.

## Channels

| Prefix | Sensor component |
|---|---|
| `total_acc_*` | Total accelerometer acceleration |
| `body_acc_*` | Estimated body acceleration |
| `body_gyro_*` | Estimated body angular velocity |

Each channel file covers X, Y, or Z axis data. Rows correspond to windows and columns correspond to 128 sequential samples.

## Use Case

The workflow stacks the nine channels into tensors shaped `(number_of_windows, 128, 9)` for final CNN evaluation.
