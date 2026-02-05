"""
TwinScale YAML Generator Service

Converts W3C WoT Thing Descriptions to TwinScale Framework YAML format.
Generates both TwinInterface (blueprint) and TwinInstance (concrete instance) YAML files.

Usage:
    generator = TwinScaleGeneratorService()
    interface_yaml = generator.generate_twin_interface_yaml(thing_description_dict)
    instance_yaml = generator.generate_twin_instance_yaml(thing_description_dict, interface_name)
"""
import re
import yaml
from typing import Dict, List, Optional, Any
from datetime import datetime

try:
    # Relative import (when used as part of app)
    from ..models.twinscale_models import (
        TwinInterfaceCR,
        TwinInstanceCR,
        TwinScaleMetadata,
        TwinInterfaceSpec,
        TwinInstanceSpec,
        TwinScaleProperty,
        TwinScaleRelationship,
        TwinScaleCommand,
        ServiceSpec,
        ServiceResources,
        ServiceAutoscaling,
        EventStoreSpec,
        HistoricalStoreSpec,
        TwinInstanceRelationship,
        ValidationResult,
    )
except ImportError:
    # Absolute import (when used standalone for testing)
    from app.models.twinscale_models import (
        TwinInterfaceCR,
        TwinInstanceCR,
        TwinScaleMetadata,
        TwinInterfaceSpec,
        TwinInstanceSpec,
        TwinScaleProperty,
        TwinScaleRelationship,
        TwinScaleCommand,
        ServiceSpec,
        ServiceResources,
        ServiceAutoscaling,
        EventStoreSpec,
        HistoricalStoreSpec,
        TwinInstanceRelationship,
        ValidationResult,
    )


class TwinScaleGeneratorService:
    """Service for generating TwinScale YAML from WoT Thing Descriptions"""

    NAMESPACE_PREFIX = "ems-iodt2"

    def __init__(self):
        """Initialize the generator service"""
        pass

    # ========================================================================
    # Public API
    # ========================================================================

    def generate_twin_interface_yaml(
        self,
        thing_description: Dict[str, Any],
        include_service_spec: bool = True,
        thing_type: str = "device",
        domain_metadata: Optional[Dict[str, str]] = None,
        dtdl_interface: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Generate TwinInterface YAML from WoT Thing Description

        Args:
            thing_description: W3C WoT Thing Description dict
            include_service_spec: Whether to include service/container spec
            thing_type: Thing modeling type ('device', 'sensor', 'component')
            domain_metadata: Domain metadata (manufacturer, model, serial_number, firmware_version)
            dtdl_interface: Optional DTDL interface metadata (dtmi, displayName, etc.)

        Returns:
            YAML string representing TwinInterface

        Raises:
            ValueError: If thing_description is invalid
        """
        # Extract metadata
        thing_id = thing_description.get("@id") or thing_description.get("id")
        if not thing_id:
            raise ValueError("Thing Description must have an @id or id field")

        interface_name = self._normalize_name(thing_id)

        # Build labels
        labels = {
            "generated-by": "iodt2-platform",
            "generated-at": datetime.utcnow().isoformat(),
            "thing-type": thing_type,  # NEW: Add thing type
        }

        # Build annotations
        annotations = {
            "source": "wot-thing-description",
            "original-id": thing_id,
        }

        # Add domain metadata to annotations if provided
        if domain_metadata:
            if domain_metadata.get("manufacturer"):
                annotations["manufacturer"] = domain_metadata["manufacturer"]
            if domain_metadata.get("model"):
                annotations["model"] = domain_metadata["model"]
            if domain_metadata.get("serial_number"):
                annotations["serialNumber"] = domain_metadata["serial_number"]

        # Add DTDL interface metadata if provided
        if dtdl_interface:
            annotations["dtdl-interface"] = dtdl_interface.get("dtmi", "")
            annotations["dtdl-interface-name"] = dtdl_interface.get("displayName", "")
            if dtdl_interface.get("category"):
                annotations["dtdl-category"] = dtdl_interface["category"]
            if domain_metadata.get("firmware_version"):
                annotations["firmwareVersion"] = domain_metadata["firmware_version"]

        # Build TwinInterface CR
        interface_cr = TwinInterfaceCR(
            metadata=TwinScaleMetadata(
                name=interface_name,
                labels=labels,
                annotations=annotations,
            ),
            spec=TwinInterfaceSpec(
                name=interface_name,
                properties=self._extract_properties(thing_description),
                relationships=self._extract_relationships(thing_description),
                commands=self._extract_commands(thing_description),
                service=self._build_service_spec() if include_service_spec else None,
                eventStore=EventStoreSpec(persistRealEvent=True),
                historicalStore=HistoricalStoreSpec(persistRealEvent=True),
            )
        )

        # Convert to YAML
        return self._to_yaml(interface_cr.model_dump(by_alias=True, exclude_none=True))

    def generate_twin_instance_yaml(
        self,
        thing_description: Dict[str, Any],
        interface_name: Optional[str] = None
    ) -> str:
        """
        Generate TwinInstance YAML from WoT Thing Description

        Args:
            thing_description: W3C WoT Thing Description dict
            interface_name: Name of the TwinInterface (auto-generated if not provided)

        Returns:
            YAML string representing TwinInstance

        Raises:
            ValueError: If thing_description is invalid
        """
        # Extract metadata
        thing_id = thing_description.get("@id") or thing_description.get("id")
        if not thing_id:
            raise ValueError("Thing Description must have an @id or id field")

        instance_name = self._normalize_name(thing_id)
        if not interface_name:
            interface_name = instance_name

        # Build TwinInstance CR
        instance_cr = TwinInstanceCR(
            metadata=TwinScaleMetadata(
                name=instance_name,
                labels={
                    "generated-by": "iodt2-platform",
                    "generated-at": datetime.utcnow().isoformat(),
                },
                annotations={
                    "source": "wot-thing-description",
                    "original-id": thing_id,
                }
            ),
            spec=TwinInstanceSpec(
                name=instance_name,
                interface=interface_name,
                twinInstanceRelationships=self._extract_instance_relationships(thing_description),
            )
        )

        # Convert to YAML
        return self._to_yaml(instance_cr.model_dump(by_alias=True, exclude_none=True))

    def generate_location_instance_yaml(
        self,
        location_data: Dict[str, Any]
    ) -> Optional[str]:
        """
        Generate TwinInstance YAML for a Location entity (optional)

        Args:
            location_data: Location information from Thing Description

        Returns:
            YAML string or None if location_data is incomplete
        """
        if not location_data or "name" not in location_data:
            return None

        location_name = self._normalize_name(location_data["name"])

        location_cr = TwinInstanceCR(
            metadata=TwinScaleMetadata(
                name=f"{location_name}-location",
                labels={
                    "type": "location",
                    "generated-by": "iodt2-platform",
                },
            ),
            spec=TwinInstanceSpec(
                name=f"{location_name}-location",
                interface="ems-iodt2-location",
                twinInstanceRelationships=[],
            )
        )

        return self._to_yaml(location_cr.model_dump(by_alias=True, exclude_none=True))

    def validate_twinscale_yaml(
        self,
        yaml_content: str,
        kind: str
    ) -> ValidationResult:
        """
        Validate TwinScale YAML content

        Args:
            yaml_content: YAML string to validate
            kind: Expected kind (TwinInterface or TwinInstance)

        Returns:
            ValidationResult with valid flag and errors/warnings
        """
        errors = []
        warnings = []

        try:
            # Parse YAML
            data = yaml.safe_load(yaml_content)

            # Check structure
            if not isinstance(data, dict):
                errors.append("YAML must be a dictionary")
                return ValidationResult(valid=False, errors=errors)

            # Check required fields
            if data.get("apiVersion") != "dtd.twinscale/v0":
                errors.append(f"Invalid apiVersion: {data.get('apiVersion')}")

            if data.get("kind") != kind:
                errors.append(f"Expected kind '{kind}', got '{data.get('kind')}'")

            if "metadata" not in data:
                errors.append("Missing 'metadata' field")
            elif "name" not in data.get("metadata", {}):
                errors.append("Missing 'metadata.name' field")

            if "spec" not in data:
                errors.append("Missing 'spec' field")

            # Validate with Pydantic
            if kind == "TwinInterface":
                TwinInterfaceCR(**data)
            elif kind == "TwinInstance":
                TwinInstanceCR(**data)
            else:
                errors.append(f"Unknown kind: {kind}")

        except yaml.YAMLError as e:
            errors.append(f"YAML parsing error: {str(e)}")
        except Exception as e:
            errors.append(f"Validation error: {str(e)}")

        return ValidationResult(
            valid=len(errors) == 0,
            errors=errors,
            warnings=warnings
        )

    # ========================================================================
    # Private Helper Methods
    # ========================================================================

    def _normalize_name(self, thing_id: str) -> str:
        """
        Normalize thing ID to TwinScale naming convention

        Examples:
            urn:iodt2:sensor:temp-001 -> ems-iodt2-temp-001
            sensor-temp-001 -> ems-iodt2-sensor-temp-001
        """
        # Remove URN prefix if present
        name = thing_id.split(":")[-1]

        # Convert to lowercase and replace invalid chars
        name = re.sub(r"[^a-z0-9-]", "-", name.lower())

        # Remove consecutive dashes
        name = re.sub(r"-+", "-", name).strip("-")

        # Add prefix
        return f"{self.NAMESPACE_PREFIX}-{name}"

    def _extract_properties(
        self,
        thing_description: Dict[str, Any]
    ) -> List[TwinScaleProperty]:
        """Extract properties from WoT Thing Description"""
        properties = []
        wot_properties = thing_description.get("properties", {})

        for prop_name, prop_def in wot_properties.items():
            # Map WoT type to TwinScale type
            wot_type = prop_def.get("type", "string")
            twinscale_type = self._map_wot_type_to_twinscale(wot_type)

            property_obj = TwinScaleProperty(
                name=prop_name,
                type=twinscale_type,
                description=prop_def.get("description") or prop_def.get("title"),
                x_writable=prop_def.get("writable", False),
                x_minimum=prop_def.get("minimum"),
                x_maximum=prop_def.get("maximum"),
                x_unit=prop_def.get("unit"),
            )
            properties.append(property_obj)

        return properties

    def _extract_relationships(
        self,
        thing_description: Dict[str, Any]
    ) -> List[TwinScaleRelationship]:
        """Extract relationships from WoT Thing Description links"""
        relationships = []
        links = thing_description.get("links", [])

        for link in links:
            rel = link.get("rel")
            if not rel or rel in ["self", "type"]:
                continue

            # Extract target interface from href
            href = link.get("href", "")
            target_interface = self._extract_interface_from_href(href)

            if target_interface:
                relationship = TwinScaleRelationship(
                    name=rel,
                    interface=target_interface,
                    description=link.get("title"),
                )
                relationships.append(relationship)

        return relationships

    def _extract_commands(
        self,
        thing_description: Dict[str, Any]
    ) -> List[TwinScaleCommand]:
        """Extract commands/actions from WoT Thing Description"""
        commands = []
        wot_actions = thing_description.get("actions", {})

        for action_name, action_def in wot_actions.items():
            command = TwinScaleCommand(
                name=action_name,
                description=action_def.get("description") or action_def.get("title"),
                schema=action_def.get("input", {}),
            )
            commands.append(command)

        return commands

    def _extract_instance_relationships(
        self,
        thing_description: Dict[str, Any]
    ) -> List[TwinInstanceRelationship]:
        """Extract concrete instance relationships from WoT links"""
        relationships = []
        links = thing_description.get("links", [])

        for link in links:
            rel = link.get("rel")
            if not rel or rel in ["self", "type"]:
                continue

            href = link.get("href", "")
            target_interface = self._extract_interface_from_href(href)
            target_instance = self._extract_instance_from_href(href)

            if target_interface and target_instance:
                relationship = TwinInstanceRelationship(
                    name=rel,
                    interface=target_interface,
                    instance=target_instance,
                )
                relationships.append(relationship)

        return relationships

    def _build_service_spec(self) -> ServiceSpec:
        """Build default service specification"""
        return ServiceSpec(
            image="iodt2/twin-service:latest",
            resources=ServiceResources(
                cpu="500m",
                memory="512Mi"
            ),
            autoscaling=ServiceAutoscaling(
                min=1,
                max=10
            )
        )

    def _map_wot_type_to_twinscale(self, wot_type: str) -> str:
        """Map WoT data type to TwinScale type"""
        type_mapping = {
            "number": "float",
            "integer": "integer",
            "string": "string",
            "boolean": "boolean",
            "object": "object",
            "array": "array",
        }
        return type_mapping.get(wot_type.lower(), "string")

    def _extract_interface_from_href(self, href: str) -> Optional[str]:
        """Extract interface name from link href"""
        if not href:
            return None

        # Try to extract interface name from URN or path
        # Example: urn:iodt2:interface:location -> ems-iodt2-location
        parts = href.split("/")[-1].split(":")
        if parts:
            return self._normalize_name(parts[-1])

        return None

    def _extract_instance_from_href(self, href: str) -> Optional[str]:
        """Extract instance name from link href"""
        if not href:
            return None

        # Similar to interface extraction
        parts = href.split("/")[-1].split(":")
        if parts:
            return self._normalize_name(parts[-1])

        return None

    def _to_yaml(self, data: Dict[str, Any]) -> str:
        """
        Convert dictionary to YAML string with proper formatting

        Args:
            data: Dictionary to convert

        Returns:
            Formatted YAML string
        """
        return yaml.dump(
            data,
            default_flow_style=False,
            sort_keys=False,
            allow_unicode=True,
            indent=2,
        )


# ============================================================================
# Convenience Functions
# ============================================================================

def generate_twinscale_yaml_files(
    thing_description: Dict[str, Any]
) -> Dict[str, str]:
    """
    Generate both TwinInterface and TwinInstance YAML files

    Args:
        thing_description: W3C WoT Thing Description dict

    Returns:
        Dictionary with keys 'interface' and 'instance' containing YAML strings
    """
    generator = TwinScaleGeneratorService()

    interface_yaml = generator.generate_twin_interface_yaml(thing_description)
    instance_yaml = generator.generate_twin_instance_yaml(thing_description)

    return {
        "interface": interface_yaml,
        "instance": instance_yaml,
    }


__all__ = [
    "TwinScaleGeneratorService",
    "generate_twinscale_yaml_files",
]
