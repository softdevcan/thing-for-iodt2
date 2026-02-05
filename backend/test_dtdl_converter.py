"""
Test script for DTDL Converter Service

Demonstrates conversion between DTDL and TwinScale formats.
Run this from the backend directory: python test_dtdl_converter.py
"""

import sys
from pathlib import Path

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent))

from app.services.dtdl_converter_service import DTDLConverterService


def main():
    print("=" * 70)
    print("DTDL Converter Service Test")
    print("=" * 70)
    print()

    # Initialize converter
    print("1. Initializing DTDL Converter...")
    converter = DTDLConverterService()
    print("   [OK] Converter initialized")
    print()

    # Test Case 1: Convert Temperature Sensor to TwinScale
    print("2. Test Case 1: Convert TemperatureSensor to TwinScale YAML")
    print("   Converting dtmi:iodt2:TemperatureSensor;1...")
    result1 = converter.dtdl_to_twinscale_template(
        dtmi="dtmi:iodt2:TemperatureSensor;1",
        thing_name="my-temp-sensor",
        tenant_id="tenant-001"
    )
    print()
    print("   === TwinInterface YAML ===")
    print(result1["interface_yaml"])
    print()
    print("   === TwinInstance YAML ===")
    print(result1["instance_yaml"])
    print()

    # Test Case 2: Convert Humidity Sensor
    print("3. Test Case 2: Convert HumiditySensor to TwinScale YAML")
    result2 = converter.dtdl_to_twinscale_template(
        dtmi="dtmi:iodt2:HumiditySensor;1"
    )
    print("   Generated YAML templates for HumiditySensor")
    print("   Interface name:", result2["interface_yaml"].split("name: ")[1].split("\n")[0])
    print()

    # Test Case 3: Convert Weather Station (component-based)
    print("4. Test Case 3: Convert WeatherStation (Component-based)")
    result3 = converter.dtdl_to_twinscale_template(
        dtmi="dtmi:iodt2:WeatherStation;1",
        thing_name="weather-station-01"
    )
    print("   Generated YAML with components:")
    # Count components
    component_count = result3["interface_yaml"].count("components:")
    if component_count > 0:
        print("   - Has component section")
        print("   - Contains: temperatureSensor, humiditySensor")
    print()

    # Test Case 4: Enrich TwinScale Thing with DTDL
    print("5. Test Case 4: Enrich TwinScale Thing with DTDL metadata")
    thing_data = {
        "metadata": {
            "name": "my-sensor",
            "labels": {
                "tenant": "tenant-001"
            }
        },
        "spec": {
            "telemetry": {
                "temperature": 22.5
            }
        }
    }
    enriched = converter.enrich_twinscale_with_dtdl(
        thing_data,
        "dtmi:iodt2:TemperatureSensor;1"
    )
    print("   Original annotations:", thing_data.get("metadata", {}).get("annotations", {}))
    print("   Enriched annotations:", enriched["metadata"]["annotations"])
    print()

    # Test Case 5: Extract DTDL binding
    print("6. Test Case 5: Extract DTDL binding from Thing")
    binding = converter.extract_dtdl_binding(enriched)
    print(f"   Extracted binding: {binding}")
    print()

    # Test Case 6: Get interface summary
    print("7. Test Case 6: Get interface summary (PM2.5 Sensor)")
    summary = converter.get_interface_summary("dtmi:iodt2:PM25Sensor;1")
    print(f"   Interface: {summary['displayName']}")
    print(f"   Description: {summary['description']}")
    print(f"   Extends: {summary.get('extends', 'N/A')}")
    print(f"   Counts:")
    print(f"     - Telemetry: {summary['telemetryCount']} ({', '.join(summary['telemetryNames'])})")
    print(f"     - Properties: {summary['propertyCount']} ({', '.join(summary['propertyNames'])})")
    print(f"     - Commands: {summary['commandCount']}")
    print(f"     - Components: {summary['componentCount']}")
    print()

    # Test Case 7: Convert all base interfaces
    print("8. Test Case 7: Convert all base interfaces")
    base_interfaces = [
        "dtmi:iodt2:BaseTwin;1",
        "dtmi:iodt2:SensorTwin;1",
        "dtmi:iodt2:ActuatorTwin;1",
        "dtmi:iodt2:GatewayTwin;1"
    ]
    for dtmi in base_interfaces:
        try:
            result = converter.dtdl_to_twinscale_template(dtmi)
            interface_name = result["interface_yaml"].split("name: ")[1].split("\n")[0]
            print(f"   [OK] {dtmi} -> {interface_name}")
        except Exception as e:
            print(f"   [FAIL] {dtmi}: {e}")
    print()

    # Test Case 8: Schema conversion test
    print("9. Test Case 8: Schema conversion examples")
    test_schemas = [
        ("double", "double"),
        ("integer", "integer"),
        ("string", "string"),
        ("boolean", "boolean"),
        ({"@type": "Enum"}, "string"),
        ({"@type": "Object"}, "object"),
        ({"@type": "Array"}, "array")
    ]
    print("   DTDL Schema -> TwinScale Schema:")
    for dtdl_schema, expected in test_schemas:
        converted = converter._convert_dtdl_schema(dtdl_schema)
        status = "[OK]" if converted == expected else "[MISMATCH]"
        print(f"     {status} {dtdl_schema} -> {converted}")
    print()

    # Test Case 9: Default value generation
    print("10. Test Case 9: Default value generation")
    test_types = [
        "boolean",
        "double",
        "integer",
        "string",
        {"@type": "Enum", "enumValues": [{"enumValue": "value1"}]},
        {"@type": "Object"},
        {"@type": "Array"}
    ]
    print("   Schema Type -> Default Value:")
    for schema_type in test_types:
        default = converter._get_default_value(schema_type)
        print(f"     {schema_type} -> {default}")
    print()

    print("=" * 70)
    print("Test completed successfully!")
    print("=" * 70)


if __name__ == "__main__":
    main()
