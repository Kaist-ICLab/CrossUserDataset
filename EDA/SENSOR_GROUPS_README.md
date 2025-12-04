# Sensor Column Groups

This document outlines the grouping of sensor columns into 8 high-level categories, based on the user's provided schema.

## Grouping Summary

| Group | Count | Description |
| :--- | :--- | :--- |
| **Motion & Physical Activity** | 1239 | Accelerometer (`ACE`), Activity (`ACT`), Fitbit Steps/Distance/Calories. |
| **Phone Usage & Interaction** | 1696 | App Usage (`APP`), Screen (`SCR`), Notifications, Ringer (`RING`), Installed Apps (`INST`). |
| **Communication & Social Behavior** | 545 | Messages (`MSG`), Calls (`CALL`). |
| **Connectivity Sensors** | 33 | Wireless/Bluetooth (`WLS`). |
| **Device State & System Context** | 1035 | Battery (`BAT`), Charging (`CHG`), Power (`PWR`), Data Traffic (`DATA`). |
| **Location & Mobility** | 284 | Location (`LOC`). |
| **Wearable Physiology** | 113 | Fitbit Heart Rate (`FitbitHeartrate`). |
| **Other** | 78 | Sleep data, Demographics (`PIF`), Time metadata, Dataset metadata. |

## Grouping Logic

The grouping is performed by the `group_columns.py` script using prefix matching:

- **Motion**: `ACE`, `ACT`, `FitbitStepcount`, `Fitbitdistance`, `Fitbitcalorie`
- **Phone Usage**: `APP`, `SCR`, `Notification`, `RING`, `ONOFF`, `INST`
- **Communication**: `MSG`, `CALL`
- **Connectivity**: `WLS`
- **Device State**: `BAT`, `CHG`, `PWR`, `Dozemode`, `DATA`
- **Location**: `LOC`
- **Physiology**: `FitbitHeartrate`
