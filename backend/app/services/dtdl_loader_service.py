"""
DTDL Loader Service

Loads and manages DTDL (Digital Twins Definition Language) interface library.
Provides search, filtering, and retrieval capabilities for DTDL interfaces.

Usage:
    loader = DTDLLoaderService()
    interfaces = loader.search_interfaces(thing_type="sensor")
    interface = loader.get_interface("dtmi:iodt2:TemperatureSensor;1")
"""

import json
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional
from functools import lru_cache

logger = logging.getLogger(__name__)


class DTDLLoaderService:
    """Service for loading and managing DTDL interface library"""

    def __init__(self):
        """Initialize DTDL loader"""
        self.library_path = Path(__file__).parent.parent / "dtdl_library"
        self.registry_path = self.library_path / "registry.json"

        # Cache for loaded interfaces
        self._interfaces_cache: Dict[str, Dict[str, Any]] = {}
        self._registry_cache: Optional[Dict[str, Any]] = None

        # Load registry and interfaces on initialization
        self._load_registry()
        self._load_all_interfaces()

        logger.info(f"DTDL Loader initialized with {len(self._interfaces_cache)} interfaces")

    def _load_registry(self) -> Dict[str, Any]:
        """Load registry.json"""
        try:
            if not self.registry_path.exists():
                logger.error(f"Registry not found at: {self.registry_path}")
                self._registry_cache = {
                    "version": "1.0.0",
                    "interfaces": [],
                    "thingTypeMapping": {},
                    "domainMapping": {}
                }
                return self._registry_cache

            with open(self.registry_path, 'r', encoding='utf-8') as f:
                self._registry_cache = json.load(f)
                logger.info(f"Loaded registry v{self._registry_cache['version']} with "
                           f"{len(self._registry_cache['interfaces'])} interface definitions")
                return self._registry_cache

        except Exception as e:
            logger.error(f"Failed to load registry: {e}")
            self._registry_cache = {
                "version": "1.0.0",
                "interfaces": [],
                "thingTypeMapping": {},
                "domainMapping": {}
            }
            return self._registry_cache

    def _load_all_interfaces(self):
        """Load all DTDL interface files from library"""
        if not self._registry_cache:
            logger.warning("Registry not loaded, skipping interface loading")
            return

        for interface_def in self._registry_cache.get("interfaces", []):
            dtmi = interface_def.get("dtmi")
            file_path = interface_def.get("filePath")

            if not dtmi or not file_path:
                continue

            try:
                full_path = self.library_path / file_path
                if not full_path.exists():
                    logger.warning(f"Interface file not found: {full_path}")
                    continue

                with open(full_path, 'r', encoding='utf-8') as f:
                    interface_json = json.load(f)

                # Merge registry metadata with interface JSON
                interface_json["_registry"] = interface_def
                self._interfaces_cache[dtmi] = interface_json

            except Exception as e:
                logger.error(f"Failed to load interface {dtmi} from {file_path}: {e}")

        logger.info(f"Successfully loaded {len(self._interfaces_cache)} DTDL interfaces")

    def get_interface(self, dtmi: str) -> Optional[Dict[str, Any]]:
        """
        Get DTDL interface by DTMI

        Args:
            dtmi: Digital Twin Model Identifier (e.g., "dtmi:iodt2:TemperatureSensor;1")

        Returns:
            DTDL interface JSON or None if not found
        """
        return self._interfaces_cache.get(dtmi)

    def list_all_interfaces(self) -> List[Dict[str, Any]]:
        """
        List all available DTDL interfaces

        Returns:
            List of interface metadata from registry
        """
        if not self._registry_cache:
            return []

        return self._registry_cache.get("interfaces", [])

    def search_interfaces(
        self,
        thing_type: Optional[str] = None,
        domain: Optional[str] = None,
        category: Optional[str] = None,
        tags: Optional[List[str]] = None,
        keywords: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Search interfaces by criteria

        Args:
            thing_type: Filter by thing type (device, sensor, component)
            domain: Filter by domain (environmental, air_quality, etc.)
            category: Filter by category (base, environmental, etc.)
            tags: Filter by tags (must match all provided tags)
            keywords: Search in displayName and description

        Returns:
            List of matching interface metadata
        """
        results = []

        for interface_def in self.list_all_interfaces():
            # Filter by thing_type
            if thing_type:
                interface_thing_type = interface_def.get("thingType")
                # Check direct thingType match
                if interface_thing_type == thing_type:
                    pass  # Direct match, include this interface
                # Check thingTypeMapping
                elif self._is_in_thing_type_mapping(interface_def["dtmi"], thing_type):
                    pass  # Found in mapping, include this interface
                else:
                    # No match found, skip this interface
                    continue

            # Filter by domain
            if domain:
                if not self._is_in_domain_mapping(interface_def["dtmi"], domain):
                    continue

            # Filter by category
            if category:
                if interface_def.get("category") != category:
                    continue

            # Filter by tags (AND logic - must have all tags)
            if tags:
                interface_tags = interface_def.get("tags", [])
                if not all(tag in interface_tags for tag in tags):
                    continue

            # Filter by keywords
            if keywords:
                keywords_lower = keywords.lower()
                display_name = interface_def.get("displayName", "").lower()
                description = interface_def.get("description", "").lower()

                if keywords_lower not in display_name and keywords_lower not in description:
                    continue

            results.append(interface_def)

        return results

    def _is_in_thing_type_mapping(self, dtmi: str, thing_type: str) -> bool:
        """Check if DTMI is in thingTypeMapping for given thing_type"""
        if not self._registry_cache:
            return False

        mapping = self._registry_cache.get("thingTypeMapping", {})
        dtmi_list = mapping.get(thing_type, [])
        return dtmi in dtmi_list

    def _is_in_domain_mapping(self, dtmi: str, domain: str) -> bool:
        """Check if DTMI is in domainMapping for given domain"""
        if not self._registry_cache:
            return False

        mapping = self._registry_cache.get("domainMapping", {})
        dtmi_list = mapping.get(domain, [])
        return dtmi in dtmi_list

    def get_base_for_thing_type(self, thing_type: str) -> Optional[str]:
        """
        Get recommended base interface DTMI for thing_type

        Args:
            thing_type: 'device', 'sensor', or 'component'

        Returns:
            DTMI of recommended base interface
        """
        mapping = {
            "sensor": "dtmi:iodt2:SensorTwin;1",
            "device": "dtmi:iodt2:BaseTwin;1",  # or ActuatorTwin/GatewayTwin
            "component": "dtmi:iodt2:BaseTwin;1"
        }
        return mapping.get(thing_type)

    def get_interfaces_by_thing_type(self, thing_type: str) -> List[Dict[str, Any]]:
        """
        Get all interfaces recommended for a thing_type

        Args:
            thing_type: 'device', 'sensor', or 'component'

        Returns:
            List of interface metadata
        """
        return self.search_interfaces(thing_type=thing_type)

    def get_interfaces_by_domain(self, domain: str) -> List[Dict[str, Any]]:
        """
        Get all interfaces in a domain

        Args:
            domain: Domain name (environmental, air_quality, etc.)

        Returns:
            List of interface metadata
        """
        return self.search_interfaces(domain=domain)

    def validate_dtmi(self, dtmi: str) -> bool:
        """
        Validate DTMI format

        Args:
            dtmi: Digital Twin Model Identifier

        Returns:
            True if valid DTMI format
        """
        # Basic DTMI validation: dtmi:<scheme>:<path>;<version>
        if not dtmi.startswith("dtmi:"):
            return False

        if ";" not in dtmi:
            return False

        parts = dtmi.split(";")
        if len(parts) != 2:
            return False

        try:
            version = int(parts[1])
            if version < 1:
                return False
        except ValueError:
            return False

        return True

    def get_interface_details(self, dtmi: str) -> Optional[Dict[str, Any]]:
        """
        Get detailed interface information including contents

        Args:
            dtmi: Digital Twin Model Identifier

        Returns:
            Full interface JSON with metadata
        """
        interface = self.get_interface(dtmi)
        if not interface:
            return None

        # Count contents by type
        contents = interface.get("contents", [])
        telemetry_count = len([c for c in contents if c.get("@type") == "Telemetry"])
        property_count = len([c for c in contents if c.get("@type") == "Property"])
        command_count = len([c for c in contents if c.get("@type") == "Command"])
        relationship_count = len([c for c in contents if c.get("@type") == "Relationship"])
        component_count = len([c for c in contents if c.get("@type") == "Component"])

        # Add summary
        interface["_summary"] = {
            "telemetryCount": telemetry_count,
            "propertyCount": property_count,
            "commandCount": command_count,
            "relationshipCount": relationship_count,
            "componentCount": component_count,
            "totalContents": len(contents)
        }

        return interface

    def reload(self):
        """Reload registry and all interfaces (for development/testing)"""
        logger.info("Reloading DTDL library...")
        self._interfaces_cache.clear()
        self._registry_cache = None
        self._load_registry()
        self._load_all_interfaces()
        logger.info("DTDL library reloaded successfully")


# Singleton instance
_loader_instance: Optional[DTDLLoaderService] = None


def get_dtdl_loader() -> DTDLLoaderService:
    """
    Get singleton instance of DTDLLoaderService

    Returns:
        DTDLLoaderService instance
    """
    global _loader_instance
    if _loader_instance is None:
        _loader_instance = DTDLLoaderService()
    return _loader_instance


__all__ = ["DTDLLoaderService", "get_dtdl_loader"]
