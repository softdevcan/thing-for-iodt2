"""
Test script for DTDL Validator Service

Demonstrates validation of TwinScale Things against DTDL interfaces.
Run this from the backend directory: python test_dtdl_validator.py
"""

import sys
from pathlib import Path

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent))

from app.services.dtdl_validator_service import DTDLValidatorService, ValidationSeverity


def main():
    print("=" * 70)
    print("DTDL Validator Service Test")
    print("=" * 70)
    print()

    # Initialize validator
    print("1. Initializing DTDL Validator...")
    validator = DTDLValidatorService()
    print("   [OK] Validator initialized")
    print()

    # Test Case 1: Perfect match - Temperature Sensor
    print("2. Test Case 1: Perfect match (Temperature Sensor)")
    perfect_thing = {
        "telemetry": {
            "temperature": 22.5
        },
        "properties": {
            "temperatureUnit": "celsius",
            "alertThreshold": 30.0
        }
    }
    result1 = validator.validate_thing_against_interface(
        perfect_thing,
        "dtmi:iodt2:TemperatureSensor;1"
    )
    print(f"   Compatibility Score: {result1.compatibility_score}/100")
    print(f"   Is Compatible: {result1.is_compatible}")
    print(f"   Matched Telemetry: {result1.matched_telemetry}")
    print(f"   Matched Properties: {result1.matched_properties}")
    print(f"   Issues: {len(result1.issues)}")
    print()

    # Test Case 2: Partial match - Missing properties
    print("3. Test Case 2: Partial match (Missing properties)")
    partial_thing = {
        "telemetry": {
            "temperature": 22.5
        },
        "properties": {}
    }
    result2 = validator.validate_thing_against_interface(
        partial_thing,
        "dtmi:iodt2:TemperatureSensor;1"
    )
    print(f"   Compatibility Score: {result2.compatibility_score}/100")
    print(f"   Is Compatible: {result2.is_compatible}")
    print(f"   Missing Properties: {result2.missing_properties}")
    print(f"   Issues:")
    for issue in result2.issues[:3]:  # Show first 3 issues
        print(f"     - [{issue.severity.value.upper()}] {issue.field}: {issue.message}")
    print()

    # Test Case 3: Extra fields
    print("4. Test Case 3: Extra fields (non-strict mode)")
    extra_fields_thing = {
        "telemetry": {
            "temperature": 22.5,
            "pressure": 1013.25  # Not in TemperatureSensor interface
        },
        "properties": {
            "temperatureUnit": "celsius",
            "alertThreshold": 30.0,
            "location": "Room 101"  # Extra field
        }
    }
    result3 = validator.validate_thing_against_interface(
        extra_fields_thing,
        "dtmi:iodt2:TemperatureSensor;1",
        strict=False
    )
    print(f"   Compatibility Score: {result3.compatibility_score}/100")
    print(f"   Extra Fields: {result3.extra_fields}")
    print(f"   Issues:")
    for issue in result3.issues:
        if issue.severity == ValidationSeverity.INFO:
            print(f"     - [INFO] {issue.field}: {issue.message}")
    print()

    # Test Case 4: Type mismatch
    print("5. Test Case 4: Type mismatch")
    wrong_type_thing = {
        "telemetry": {
            "temperature": "twenty-two"  # Should be double/float
        },
        "properties": {
            "temperatureUnit": "celsius",
            "alertThreshold": 30.0
        }
    }
    result4 = validator.validate_thing_against_interface(
        wrong_type_thing,
        "dtmi:iodt2:TemperatureSensor;1"
    )
    print(f"   Compatibility Score: {result4.compatibility_score}/100")
    print(f"   Issues:")
    for issue in result4.issues:
        if "type" in issue.message.lower():
            print(f"     - [{issue.severity.value.upper()}] {issue.field}: {issue.message}")
    print()

    # Test Case 5: Humidity Sensor validation
    print("6. Test Case 5: Humidity Sensor (with alert thresholds)")
    humidity_thing = {
        "telemetry": {
            "humidity": 65.0
        },
        "properties": {
            "humidityAlertMin": 30.0,
            "humidityAlertMax": 80.0
        }
    }
    result5 = validator.validate_thing_against_interface(
        humidity_thing,
        "dtmi:iodt2:HumiditySensor;1"
    )
    print(f"   Compatibility Score: {result5.compatibility_score}/100")
    print(f"   Is Compatible: {result5.is_compatible}")
    print(f"   Matched: {len(result5.matched_telemetry)} telemetry, {len(result5.matched_properties)} properties")
    print()

    # Test Case 6: Find best matching interfaces
    print("7. Test Case 6: Find best matching interfaces")
    unknown_sensor = {
        "telemetry": {
            "temperature": 25.0,
            "humidity": 60.0
        },
        "properties": {}
    }
    matches = validator.find_best_matching_interfaces(
        unknown_sensor,
        thing_type="sensor",
        domain="environmental",
        top_n=3
    )
    print(f"   Found {len(matches)} potential matches:")
    for i, (validation, score) in enumerate(matches, 1):
        print(f"   {i}. {validation.interface_name} (Combined Score: {score:.1f})")
        print(f"      - Compatibility: {validation.compatibility_score:.1f}")
        print(f"      - Matched: {len(validation.matched_telemetry)} telemetry, {len(validation.matched_properties)} properties")
        print(f"      - Missing: {len(validation.missing_telemetry)} telemetry, {len(validation.missing_properties)} properties")
    print()

    # Test Case 7: Get interface requirements
    print("8. Test Case 7: Get interface requirements (PM2.5 Sensor)")
    requirements = validator.get_interface_requirements("dtmi:iodt2:PM25Sensor;1")
    print(f"   Interface: {requirements['displayName']}")
    print(f"   Total Requirements: {requirements['total_requirements']}")
    print(f"   Required Telemetry:")
    for tel in requirements['required_telemetry']:
        print(f"     - {tel['name']} ({tel['schema']}) {tel.get('unit', '')}")
    print(f"   Required Properties:")
    for prop in requirements['required_properties']:
        print(f"     - {prop['name']} ({prop['schema']}) [writable: {prop['writable']}]")
    print()

    # Test Case 8: Weather Station (Component-based)
    print("9. Test Case 8: Weather Station (Component-based)")
    weather_thing = {
        "telemetry": {
            "pressure": 1013.25,
            "windSpeed": 5.5,
            "windDirection": 180,
            "rainfall": 0.0
        },
        "properties": {}
    }
    result8 = validator.validate_thing_against_interface(
        weather_thing,
        "dtmi:iodt2:WeatherStation;1"
    )
    print(f"   Compatibility Score: {result8.compatibility_score}/100")
    print(f"   Is Compatible: {result8.is_compatible}")
    print(f"   Matched Telemetry: {result8.matched_telemetry}")
    print(f"   Note: Components (temperatureSensor, humiditySensor) not validated in this test")
    print()

    # Test Case 9: Multi-sensor device
    print("10. Test Case 9: Multi-sensor device (find best match)")
    multi_sensor = {
        "telemetry": {
            "temperature": 22.0,
            "humidity": 55.0,
            "pressure": 1013.0,
            "pm25": 12.5
        },
        "properties": {
            "temperatureUnit": "celsius"
        }
    }
    matches = validator.find_best_matching_interfaces(
        multi_sensor,
        thing_type="device",
        top_n=5
    )
    print(f"   Best match for multi-sensor device:")
    if matches:
        validation, score = matches[0]
        print(f"   - {validation.interface_name}")
        print(f"   - Combined Score: {score:.1f}")
        print(f"   - Compatibility Score: {validation.compatibility_score:.1f}")
        print(f"   - Matched: {validation.matched_telemetry}")
    else:
        print("   - No suitable device-level interface found")
        print("   - Consider using component-based modeling (WeatherStation)")
    print()

    print("=" * 70)
    print("Test completed successfully!")
    print("=" * 70)


if __name__ == "__main__":
    main()
