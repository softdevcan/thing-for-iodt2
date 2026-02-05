"""
DTDL Converter Service

Bidirectional converter between DTDL interfaces and TwinScale YAML format.
Converts DTDL to TwinScale templates and enriches TwinScale Things with DTDL metadata.

Usage:
    converter = DTDLConverterService()
    yaml_template = converter.dtdl_to_twinscale_template(dtmi)
    enriched_thing = converter.enrich_twinscale_with_dtdl(thing_data, dtmi)
"""

import logging
from typing import Dict, Any, List, Optional
from datetime import datetime

from app.services.dtdl_loader_service import get_dtdl_loader

logger = logging.getLogger(__name__)


class DTDLConverterService:
    """Service for converting between DTDL and TwinScale formats"""

    def __init__(self):
        """Initialize converter with DTDL loader"""
        self.loader = get_dtdl_loader()

    def dtdl_to_twinscale_template(
        self,
        dtmi: str,
        thing_name: Optional[str] = None,
        tenant_id: Optional[str] = None
    ) -> Dict[str, str]:
        """
        Convert DTDL interface to TwinScale YAML template

        Args:
            dtmi: DTDL interface identifier
            thing_name: Optional Thing name (default: interface displayName)
            tenant_id: Optional tenant ID

        Returns:
            Dictionary with 'interface_yaml' and 'instance_yaml' keys
        """
        interface = self.loader.get_interface_details(dtmi)
        if not interface:
            raise ValueError(f"Interface not found: {dtmi}")

        display_name = interface.get("displayName", "Unknown")
        description = interface.get("description", "")

        if not thing_name:
            thing_name = display_name.lower().replace(" ", "-")

        # Extract contents
        contents = interface.get("contents", [])
        telemetry_fields = [c for c in contents if c.get("@type") == "Telemetry"]
        property_fields = [c for c in contents if c.get("@type") == "Property"]
        command_fields = [c for c in contents if c.get("@type") == "Command"]
        component_fields = [c for c in contents if c.get("@type") == "Component"]

        # Generate TwinInterface YAML
        interface_yaml = self._generate_interface_yaml(
            thing_name=thing_name,
            display_name=display_name,
            description=description,
            dtmi=dtmi,
            telemetry_fields=telemetry_fields,
            property_fields=property_fields,
            command_fields=command_fields,
            component_fields=component_fields,
            tenant_id=tenant_id
        )

        # Generate TwinInstance YAML
        instance_yaml = self._generate_instance_yaml(
            thing_name=thing_name,
            display_name=display_name,
            description=description,
            dtmi=dtmi,
            telemetry_fields=telemetry_fields,
            property_fields=property_fields,
            tenant_id=tenant_id
        )

        return {
            "interface_yaml": interface_yaml,
            "instance_yaml": instance_yaml
        }

    def _generate_interface_yaml(
        self,
        thing_name: str,
        display_name: str,
        description: str,
        dtmi: str,
        telemetry_fields: List[Dict],
        property_fields: List[Dict],
        command_fields: List[Dict],
        component_fields: List[Dict],
        tenant_id: Optional[str]
    ) -> str:
        """Generate TwinInterface YAML from DTDL"""
        yaml_lines = []

        # Metadata
        yaml_lines.append("apiVersion: twinscale.io/v1")
        yaml_lines.append("kind: TwinInterface")
        yaml_lines.append("metadata:")
        yaml_lines.append(f"  name: {thing_name}-interface")
        yaml_lines.append("  labels:")
        yaml_lines.append("    generated-by: dtdl-converter")
        yaml_lines.append(f"    generated-at: {datetime.utcnow().isoformat()}")
        if tenant_id:
            yaml_lines.append(f"    tenant: {tenant_id}")
        yaml_lines.append("  annotations:")
        yaml_lines.append(f'    description: "{description}"')
        yaml_lines.append(f'    dtdl-interface: "{dtmi}"')
        yaml_lines.append(f'    source: "DTDL v2"')
        yaml_lines.append("")

        # Spec
        yaml_lines.append("spec:")
        yaml_lines.append(f'  displayName: "{display_name}"')
        yaml_lines.append("")

        # Telemetry
        if telemetry_fields:
            yaml_lines.append("  telemetry:")
            for tel in telemetry_fields:
                name = tel.get("name")
                tel_display = tel.get("displayName", name)
                tel_schema = self._convert_dtdl_schema(tel.get("schema"))
                tel_unit = tel.get("unit", "")

                yaml_lines.append(f"    - name: {name}")
                yaml_lines.append(f'      displayName: "{tel_display}"')
                yaml_lines.append(f"      schema: {tel_schema}")
                if tel_unit:
                    yaml_lines.append(f'      unit: "{tel_unit}"')
            yaml_lines.append("")

        # Properties
        if property_fields:
            yaml_lines.append("  properties:")
            for prop in property_fields:
                name = prop.get("name")
                prop_display = prop.get("displayName", name)
                prop_schema = self._convert_dtdl_schema(prop.get("schema"))
                writable = prop.get("writable", False)

                yaml_lines.append(f"    - name: {name}")
                yaml_lines.append(f'      displayName: "{prop_display}"')
                yaml_lines.append(f"      schema: {prop_schema}")
                yaml_lines.append(f"      writable: {str(writable).lower()}")
            yaml_lines.append("")

        # Commands
        if command_fields:
            yaml_lines.append("  commands:")
            for cmd in command_fields:
                name = cmd.get("name")
                cmd_display = cmd.get("displayName", name)

                yaml_lines.append(f"    - name: {name}")
                yaml_lines.append(f'      displayName: "{cmd_display}"')
            yaml_lines.append("")

        # Components
        if component_fields:
            yaml_lines.append("  components:")
            for comp in component_fields:
                name = comp.get("name")
                comp_display = comp.get("displayName", name)
                schema = comp.get("schema", "")

                yaml_lines.append(f"    - name: {name}")
                yaml_lines.append(f'      displayName: "{comp_display}"')
                yaml_lines.append(f'      schema: "{schema}"')
            yaml_lines.append("")

        return "\n".join(yaml_lines)

    def _generate_instance_yaml(
        self,
        thing_name: str,
        display_name: str,
        description: str,
        dtmi: str,
        telemetry_fields: List[Dict],
        property_fields: List[Dict],
        tenant_id: Optional[str]
    ) -> str:
        """Generate TwinInstance YAML from DTDL"""
        yaml_lines = []

        # Metadata
        yaml_lines.append("apiVersion: twinscale.io/v1")
        yaml_lines.append("kind: TwinInstance")
        yaml_lines.append("metadata:")
        yaml_lines.append(f"  name: {thing_name}-001")
        yaml_lines.append("  labels:")
        yaml_lines.append("    generated-by: dtdl-converter")
        yaml_lines.append(f"    generated-at: {datetime.utcnow().isoformat()}")
        if tenant_id:
            yaml_lines.append(f"    tenant: {tenant_id}")
        yaml_lines.append("  annotations:")
        yaml_lines.append(f'    description: "{description}"')
        yaml_lines.append(f'    dtdl-interface: "{dtmi}"')
        yaml_lines.append("")

        # Spec
        yaml_lines.append("spec:")
        yaml_lines.append(f"  interfaceRef: {thing_name}-interface")
        yaml_lines.append(f'  displayName: "{display_name} Instance 001"')
        yaml_lines.append("")

        # Initial telemetry (with placeholder values)
        if telemetry_fields:
            yaml_lines.append("  telemetry:")
            for tel in telemetry_fields:
                name = tel.get("name")
                default_value = self._get_default_value(tel.get("schema"))
                yaml_lines.append(f"    {name}: {default_value}")
            yaml_lines.append("")

        # Initial properties (with placeholder values)
        if property_fields:
            yaml_lines.append("  properties:")
            for prop in property_fields:
                name = prop.get("name")
                default_value = self._get_default_value(prop.get("schema"))
                yaml_lines.append(f"    {name}: {default_value}")
            yaml_lines.append("")

        return "\n".join(yaml_lines)

    def _convert_dtdl_schema(self, schema: Any) -> str:
        """
        Convert DTDL schema to TwinScale schema string

        Args:
            schema: DTDL schema (string or dict)

        Returns:
            TwinScale schema string
        """
        if isinstance(schema, str):
            # Simple primitive types
            type_mapping = {
                "boolean": "boolean",
                "date": "string",
                "dateTime": "string",
                "double": "double",
                "duration": "string",
                "float": "float",
                "integer": "integer",
                "long": "long",
                "string": "string",
                "time": "string",
            }
            return type_mapping.get(schema, "string")

        elif isinstance(schema, dict):
            schema_type = schema.get("@type")

            if schema_type == "Enum":
                # Return as string for TwinScale (enum validation in app logic)
                return "string"
            elif schema_type == "Object":
                return "object"
            elif schema_type == "Array":
                return "array"
            else:
                return "string"

        return "string"

    def _get_default_value(self, schema: Any) -> Any:
        """
        Get default placeholder value for schema

        Args:
            schema: DTDL schema

        Returns:
            Default value appropriate for the schema type
        """
        if isinstance(schema, str):
            defaults = {
                "boolean": "false",
                "double": "0.0",
                "float": "0.0",
                "integer": "0",
                "long": "0",
                "string": '""',
            }
            return defaults.get(schema, '""')

        elif isinstance(schema, dict):
            schema_type = schema.get("@type")

            if schema_type == "Enum":
                # Get first enum value
                enum_values = schema.get("enumValues", [])
                if enum_values:
                    first_value = enum_values[0].get("enumValue", "")
                    return f'"{first_value}"'
                return '""'
            elif schema_type == "Object":
                return "{}"
            elif schema_type == "Array":
                return "[]"

        return '""'

    def enrich_twinscale_with_dtdl(
        self,
        thing_data: Dict[str, Any],
        dtmi: str
    ) -> Dict[str, Any]:
        """
        Enrich TwinScale Thing data with DTDL metadata

        Args:
            thing_data: Existing TwinScale Thing data
            dtmi: DTDL interface identifier

        Returns:
            Enriched Thing data with DTDL metadata
        """
        interface = self.loader.get_interface(dtmi)
        if not interface:
            raise ValueError(f"Interface not found: {dtmi}")

        enriched = thing_data.copy()

        # Add DTDL metadata to annotations
        if "metadata" not in enriched:
            enriched["metadata"] = {}
        if "annotations" not in enriched["metadata"]:
            enriched["metadata"]["annotations"] = {}

        enriched["metadata"]["annotations"]["dtdl-interface"] = dtmi
        enriched["metadata"]["annotations"]["dtdl-version"] = "2"
        enriched["metadata"]["annotations"]["interface-name"] = interface.get("displayName", "")

        # Extract interface context
        extends = interface.get("extends")
        if extends:
            if isinstance(extends, list):
                enriched["metadata"]["annotations"]["dtdl-extends"] = ",".join(extends)
            else:
                enriched["metadata"]["annotations"]["dtdl-extends"] = extends

        return enriched

    def extract_dtdl_binding(self, thing_data: Dict[str, Any]) -> Optional[str]:
        """
        Extract DTDL interface binding from Thing metadata

        Args:
            thing_data: TwinScale Thing data

        Returns:
            DTMI if found, None otherwise
        """
        try:
            annotations = thing_data.get("metadata", {}).get("annotations", {})
            return annotations.get("dtdl-interface")
        except (KeyError, AttributeError):
            return None

    def get_interface_summary(self, dtmi: str) -> Dict[str, Any]:
        """
        Get a summary of DTDL interface for UI display

        Args:
            dtmi: DTDL interface identifier

        Returns:
            Summary dictionary with counts and lists
        """
        interface = self.loader.get_interface_details(dtmi)
        if not interface:
            return {}

        contents = interface.get("contents", [])
        summary = interface.get("_summary", {})

        return {
            "dtmi": dtmi,
            "displayName": interface.get("displayName"),
            "description": interface.get("description"),
            "extends": interface.get("extends"),
            "telemetryCount": summary.get("telemetryCount", 0),
            "propertyCount": summary.get("propertyCount", 0),
            "commandCount": summary.get("commandCount", 0),
            "componentCount": summary.get("componentCount", 0),
            "telemetryNames": [c["name"] for c in contents if c.get("@type") == "Telemetry"],
            "propertyNames": [c["name"] for c in contents if c.get("@type") == "Property"],
            "commandNames": [c["name"] for c in contents if c.get("@type") == "Command"],
            "componentNames": [c["name"] for c in contents if c.get("@type") == "Component"]
        }


# Singleton instance
_converter_instance: Optional[DTDLConverterService] = None


def get_dtdl_converter() -> DTDLConverterService:
    """
    Get singleton instance of DTDLConverterService

    Returns:
        DTDLConverterService instance
    """
    global _converter_instance
    if _converter_instance is None:
        _converter_instance = DTDLConverterService()
    return _converter_instance


__all__ = ["DTDLConverterService", "get_dtdl_converter"]
