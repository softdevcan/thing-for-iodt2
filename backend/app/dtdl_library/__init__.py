"""
DTDL (Digital Twins Definition Language) Library

This library contains DTDL v2 interface definitions for TwinScale platform.

Structure:
- base/: Base interfaces (BaseTwin, SensorTwin, ActuatorTwin, etc.)
- domain/: Domain-specific interfaces organized by category
  - environmental/: Temperature, humidity, pressure sensors
  - air_quality/: PM2.5, PM10, CO2 sensors
  - structural/: Strain gauges, accelerometers for structural health
  - industrial/: PLCs, motors, pumps
- extensions/: IoDT2-specific DTDL extensions

DTDL Version: 2 (dtmi:dtdl:context;2)
Namespace: dtmi:iodt2
"""

__version__ = "1.0.0"
