"""
Test script for DTDL Loader Service

Demonstrates DTDL library loading and searching capabilities.
Run this from the backend directory: python test_dtdl_loader.py
"""

import sys
from pathlib import Path

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent))

from app.services.dtdl_loader_service import DTDLLoaderService


def main():
    print("=" * 70)
    print("DTDL Loader Service Test")
    print("=" * 70)
    print()

    # Initialize loader
    print("1. Initializing DTDL Loader...")
    loader = DTDLLoaderService()
    print(f"   [OK] Loaded {len(loader._interfaces_cache)} interfaces")
    print()

    # List all interfaces
    print("2. Listing all interfaces:")
    all_interfaces = loader.list_all_interfaces()
    for interface in all_interfaces:
        print(f"   - {interface['displayName']} ({interface['dtmi']})")
        print(f"     Category: {interface['category']}, Tags: {', '.join(interface.get('tags', []))}")
    print()

    # Search by thing_type
    print("3. Searching by thing_type='sensor':")
    sensors = loader.search_interfaces(thing_type="sensor")
    print(f"   Found {len(sensors)} sensors:")
    for sensor in sensors:
        print(f"   - {sensor['displayName']} ({sensor['dtmi']})")
    print()

    # Search by domain
    print("4. Searching by domain='environmental':")
    env_interfaces = loader.search_interfaces(domain="environmental")
    print(f"   Found {len(env_interfaces)} environmental interfaces:")
    for interface in env_interfaces:
        print(f"   - {interface['displayName']} ({interface['dtmi']})")
    print()

    # Get interface details
    print("5. Getting details for TemperatureSensor:")
    temp_sensor = loader.get_interface_details("dtmi:iodt2:TemperatureSensor;1")
    if temp_sensor:
        print(f"   Display Name: {temp_sensor.get('displayName')}")
        print(f"   Description: {temp_sensor.get('description')}")
        print(f"   Extends: {temp_sensor.get('extends')}")

        summary = temp_sensor.get('_summary', {})
        print(f"   Contents:")
        print(f"     - Telemetry: {summary.get('telemetryCount', 0)}")
        print(f"     - Properties: {summary.get('propertyCount', 0)}")
        print(f"     - Commands: {summary.get('commandCount', 0)}")
        print(f"     - Components: {summary.get('componentCount', 0)}")
        print(f"     - Total: {summary.get('totalContents', 0)}")
    print()

    # Search with keywords
    print("6. Searching with keywords='weather':")
    weather_interfaces = loader.search_interfaces(keywords="weather")
    print(f"   Found {len(weather_interfaces)} interfaces:")
    for interface in weather_interfaces:
        print(f"   - {interface['displayName']} ({interface['dtmi']})")
    print()

    # Get base for thing types
    print("7. Getting recommended base interfaces:")
    for thing_type in ["sensor", "device", "component"]:
        base_dtmi = loader.get_base_for_thing_type(thing_type)
        print(f"   - {thing_type}: {base_dtmi}")
    print()

    # Validate DTMIs
    print("8. Validating DTMI formats:")
    test_dtmis = [
        "dtmi:iodt2:TemperatureSensor;1",  # Valid
        "dtmi:invalid",  # Missing version
        "notadtmi:test;1",  # Wrong prefix
        "dtmi:test;0",  # Invalid version
    ]
    for dtmi in test_dtmis:
        is_valid = loader.validate_dtmi(dtmi)
        status = "[OK] Valid" if is_valid else "[X] Invalid"
        print(f"   {status}: {dtmi}")
    print()

    # Get component-based interface
    print("9. Component-based interface example (WeatherStation):")
    weather_station = loader.get_interface("dtmi:iodt2:WeatherStation;1")
    if weather_station:
        print(f"   Display Name: {weather_station.get('displayName')}")
        print(f"   Components:")
        contents = weather_station.get('contents', [])
        components = [c for c in contents if c.get('@type') == 'Component']
        for comp in components:
            print(f"     - {comp.get('name')} ({comp.get('schema')})")
    print()

    # Domain and thing type mappings
    print("10. Registry Mappings:")
    print("    Thing Type Mapping:")
    thing_type_mapping = loader._registry_cache.get('thingTypeMapping', {})
    for thing_type, dtmis in thing_type_mapping.items():
        print(f"      - {thing_type}: {len(dtmis)} interfaces")

    print("    Domain Mapping:")
    domain_mapping = loader._registry_cache.get('domainMapping', {})
    for domain, dtmis in domain_mapping.items():
        print(f"      - {domain}: {len(dtmis)} interfaces")
    print()

    print("=" * 70)
    print("Test completed successfully!")
    print("=" * 70)


if __name__ == "__main__":
    main()
