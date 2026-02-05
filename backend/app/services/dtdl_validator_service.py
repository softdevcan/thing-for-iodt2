"""
DTDL Validator Service

Validates TwinScale Thing compatibility with DTDL interfaces.
Provides compatibility scoring and recommendations for missing/incompatible fields.

Usage:
    validator = DTDLValidatorService()
    result = validator.validate_thing_against_interface(thing_data, dtmi)
    score = result.compatibility_score
    issues = result.issues
"""

import logging
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum

from app.services.dtdl_loader_service import get_dtdl_loader

logger = logging.getLogger(__name__)


class ValidationSeverity(str, Enum):
    """Validation issue severity levels"""
    ERROR = "error"      # Blocking issue, thing cannot use this interface
    WARNING = "warning"  # Non-blocking, but recommended to fix
    INFO = "info"        # Informational, suggestions for improvement


@dataclass
class ValidationIssue:
    """Represents a validation issue"""
    severity: ValidationSeverity
    field: str
    message: str
    suggestion: Optional[str] = None


@dataclass
class ValidationResult:
    """Result of DTDL validation"""
    is_compatible: bool
    compatibility_score: float  # 0-100
    dtmi: str
    interface_name: str
    issues: List[ValidationIssue] = field(default_factory=list)
    matched_telemetry: List[str] = field(default_factory=list)
    matched_properties: List[str] = field(default_factory=list)
    missing_telemetry: List[str] = field(default_factory=list)
    missing_properties: List[str] = field(default_factory=list)
    extra_fields: List[str] = field(default_factory=list)


class DTDLValidatorService:
    """Service for validating TwinScale Things against DTDL interfaces"""

    def __init__(self):
        """Initialize validator with DTDL loader"""
        self.loader = get_dtdl_loader()

    def validate_thing_against_interface(
        self,
        thing_data: Dict[str, Any],
        dtmi: str,
        strict: bool = False
    ) -> ValidationResult:
        """
        Validate a TwinScale Thing against a DTDL interface

        Args:
            thing_data: TwinScale Thing data (properties and telemetry)
            dtmi: DTDL interface identifier
            strict: If True, extra fields are treated as errors

        Returns:
            ValidationResult with compatibility score and issues
        """
        # Get interface details
        interface = self.loader.get_interface_details(dtmi)
        if not interface:
            return ValidationResult(
                is_compatible=False,
                compatibility_score=0.0,
                dtmi=dtmi,
                interface_name="Unknown",
                issues=[ValidationIssue(
                    severity=ValidationSeverity.ERROR,
                    field="dtmi",
                    message=f"Interface not found: {dtmi}"
                )]
            )

        interface_name = interface.get("displayName", "Unknown")
        issues = []
        matched_telemetry = []
        matched_properties = []
        missing_telemetry = []
        missing_properties = []
        extra_fields = []

        # Extract DTDL contents
        contents = interface.get("contents", [])
        dtdl_telemetry = {
            c["name"]: c for c in contents if c.get("@type") == "Telemetry"
        }
        dtdl_properties = {
            c["name"]: c for c in contents if c.get("@type") == "Property"
        }

        # Extract Thing data
        thing_telemetry = thing_data.get("telemetry", {})
        thing_properties = thing_data.get("properties", {})

        # Validate telemetry
        for tel_name, tel_def in dtdl_telemetry.items():
            if tel_name in thing_telemetry:
                # Check type compatibility
                schema_issues = self._validate_schema(
                    tel_name,
                    thing_telemetry[tel_name],
                    tel_def.get("schema"),
                    "telemetry"
                )
                if schema_issues:
                    issues.extend(schema_issues)
                else:
                    matched_telemetry.append(tel_name)
            else:
                missing_telemetry.append(tel_name)
                issues.append(ValidationIssue(
                    severity=ValidationSeverity.WARNING,
                    field=f"telemetry.{tel_name}",
                    message=f"Missing telemetry: {tel_name}",
                    suggestion=f"Add telemetry field '{tel_name}' with schema {tel_def.get('schema')}"
                ))

        # Validate properties
        for prop_name, prop_def in dtdl_properties.items():
            if prop_name in thing_properties:
                # Check type compatibility
                schema_issues = self._validate_schema(
                    prop_name,
                    thing_properties[prop_name],
                    prop_def.get("schema"),
                    "property"
                )
                if schema_issues:
                    issues.extend(schema_issues)
                else:
                    matched_properties.append(prop_name)
            else:
                # Check if property is writable (required)
                is_writable = prop_def.get("writable", False)
                severity = ValidationSeverity.ERROR if is_writable else ValidationSeverity.WARNING

                missing_properties.append(prop_name)
                issues.append(ValidationIssue(
                    severity=severity,
                    field=f"property.{prop_name}",
                    message=f"Missing property: {prop_name}",
                    suggestion=f"Add property field '{prop_name}' with schema {prop_def.get('schema')}"
                ))

        # Check for extra fields
        for tel_name in thing_telemetry.keys():
            if tel_name not in dtdl_telemetry:
                extra_fields.append(f"telemetry.{tel_name}")
                severity = ValidationSeverity.ERROR if strict else ValidationSeverity.INFO
                issues.append(ValidationIssue(
                    severity=severity,
                    field=f"telemetry.{tel_name}",
                    message=f"Extra telemetry not defined in interface: {tel_name}",
                    suggestion="Remove this field or extend the interface to include it"
                ))

        for prop_name in thing_properties.keys():
            if prop_name not in dtdl_properties:
                extra_fields.append(f"property.{prop_name}")
                severity = ValidationSeverity.ERROR if strict else ValidationSeverity.INFO
                issues.append(ValidationIssue(
                    severity=severity,
                    field=f"property.{prop_name}",
                    message=f"Extra property not defined in interface: {prop_name}",
                    suggestion="Remove this field or extend the interface to include it"
                ))

        # Calculate compatibility score
        score = self._calculate_compatibility_score(
            matched_telemetry=len(matched_telemetry),
            matched_properties=len(matched_properties),
            missing_telemetry=len(missing_telemetry),
            missing_properties=len(missing_properties),
            extra_fields=len(extra_fields),
            total_errors=len([i for i in issues if i.severity == ValidationSeverity.ERROR])
        )

        # Determine compatibility
        is_compatible = score >= 60 and not any(
            i.severity == ValidationSeverity.ERROR for i in issues
        )

        return ValidationResult(
            is_compatible=is_compatible,
            compatibility_score=score,
            dtmi=dtmi,
            interface_name=interface_name,
            issues=issues,
            matched_telemetry=matched_telemetry,
            matched_properties=matched_properties,
            missing_telemetry=missing_telemetry,
            missing_properties=missing_properties,
            extra_fields=extra_fields
        )

    def _validate_schema(
        self,
        field_name: str,
        value: Any,
        schema: Any,
        field_type: str
    ) -> List[ValidationIssue]:
        """
        Validate a field value against DTDL schema

        Args:
            field_name: Name of the field
            value: Field value
            schema: DTDL schema definition
            field_type: "telemetry" or "property"

        Returns:
            List of validation issues (empty if valid)
        """
        issues = []

        # Handle simple schema types
        if isinstance(schema, str):
            expected_type = self._map_dtdl_type_to_python(schema)
            actual_type = type(value).__name__

            if expected_type and actual_type != expected_type:
                issues.append(ValidationIssue(
                    severity=ValidationSeverity.WARNING,
                    field=f"{field_type}.{field_name}",
                    message=f"Type mismatch: expected {expected_type}, got {actual_type}",
                    suggestion=f"Convert value to {expected_type}"
                ))

        # Handle complex schema (enum, object, etc.)
        elif isinstance(schema, dict):
            schema_type = schema.get("@type")

            if schema_type == "Enum":
                # Validate enum value
                enum_values = [ev.get("enumValue") for ev in schema.get("enumValues", [])]
                if value not in enum_values:
                    issues.append(ValidationIssue(
                        severity=ValidationSeverity.ERROR,
                        field=f"{field_type}.{field_name}",
                        message=f"Invalid enum value: {value}",
                        suggestion=f"Use one of: {', '.join(str(ev) for ev in enum_values)}"
                    ))

            elif schema_type == "Object":
                # Validate object structure (simplified)
                if not isinstance(value, dict):
                    issues.append(ValidationIssue(
                        severity=ValidationSeverity.ERROR,
                        field=f"{field_type}.{field_name}",
                        message=f"Expected object, got {type(value).__name__}",
                        suggestion="Provide a dictionary/object value"
                    ))

        return issues

    def _map_dtdl_type_to_python(self, dtdl_type: str) -> Optional[str]:
        """Map DTDL primitive types to Python types"""
        type_mapping = {
            "boolean": "bool",
            "date": "str",
            "dateTime": "str",
            "double": "float",
            "duration": "str",
            "float": "float",
            "integer": "int",
            "long": "int",
            "string": "str",
            "time": "str",
        }
        return type_mapping.get(dtdl_type)

    def _calculate_compatibility_score(
        self,
        matched_telemetry: int,
        matched_properties: int,
        missing_telemetry: int,
        missing_properties: int,
        extra_fields: int,
        total_errors: int
    ) -> float:
        """
        Calculate compatibility score (0-100)

        Scoring algorithm:
        - Base: 50 points
        - Matched telemetry: +5 points each
        - Matched properties: +3 points each
        - Missing telemetry: -3 points each
        - Missing properties: -2 points each
        - Extra fields: -1 point each
        - Errors: -10 points each
        """
        score = 50.0

        score += matched_telemetry * 5
        score += matched_properties * 3
        score -= missing_telemetry * 3
        score -= missing_properties * 2
        score -= extra_fields * 1
        score -= total_errors * 10

        # Clamp to 0-100
        return max(0.0, min(100.0, score))

    def find_best_matching_interfaces(
        self,
        thing_data: Dict[str, Any],
        thing_type: Optional[str] = None,
        domain: Optional[str] = None,
        top_n: int = 5
    ) -> List[Tuple[ValidationResult, float]]:
        """
        Find best matching DTDL interfaces for a Thing

        Args:
            thing_data: TwinScale Thing data
            thing_type: Optional thing type filter
            domain: Optional domain filter
            top_n: Number of top results to return

        Returns:
            List of (ValidationResult, combined_score) tuples, sorted by score
        """
        # Search for candidate interfaces
        candidates = self.loader.search_interfaces(
            thing_type=thing_type,
            domain=domain
        )

        if not candidates:
            logger.warning(f"No candidate interfaces found for thing_type={thing_type}, domain={domain}")
            return []

        # Validate against each candidate
        results = []
        for candidate in candidates:
            dtmi = candidate["dtmi"]
            validation = self.validate_thing_against_interface(thing_data, dtmi, strict=False)

            # Calculate combined score (validation + metadata match)
            metadata_score = 0
            if thing_type and candidate.get("thingType") == thing_type:
                metadata_score += 10
            if domain:
                if self.loader._is_in_domain_mapping(dtmi, domain):
                    metadata_score += 10

            combined_score = (validation.compatibility_score * 0.8) + (metadata_score * 0.2)
            results.append((validation, combined_score))

        # Sort by combined score (descending)
        results.sort(key=lambda x: x[1], reverse=True)

        return results[:top_n]

    def get_interface_requirements(self, dtmi: str) -> Dict[str, Any]:
        """
        Get requirements summary for an interface

        Args:
            dtmi: DTDL interface identifier

        Returns:
            Dictionary with required/optional telemetry and properties
        """
        interface = self.loader.get_interface_details(dtmi)
        if not interface:
            return {}

        contents = interface.get("contents", [])

        required_telemetry = []
        optional_telemetry = []
        required_properties = []
        optional_properties = []

        for content in contents:
            content_type = content.get("@type")
            name = content.get("name")

            if content_type == "Telemetry":
                required_telemetry.append({
                    "name": name,
                    "displayName": content.get("displayName", name),
                    "schema": content.get("schema"),
                    "unit": content.get("unit")
                })
            elif content_type == "Property":
                is_writable = content.get("writable", False)
                prop_info = {
                    "name": name,
                    "displayName": content.get("displayName", name),
                    "schema": content.get("schema"),
                    "writable": is_writable
                }
                if is_writable:
                    required_properties.append(prop_info)
                else:
                    optional_properties.append(prop_info)

        return {
            "dtmi": dtmi,
            "displayName": interface.get("displayName"),
            "description": interface.get("description"),
            "required_telemetry": required_telemetry,
            "optional_telemetry": optional_telemetry,
            "required_properties": required_properties,
            "optional_properties": optional_properties,
            "total_requirements": len(required_telemetry) + len(required_properties)
        }


# Singleton instance
_validator_instance: Optional[DTDLValidatorService] = None


def get_dtdl_validator() -> DTDLValidatorService:
    """
    Get singleton instance of DTDLValidatorService

    Returns:
        DTDLValidatorService instance
    """
    global _validator_instance
    if _validator_instance is None:
        _validator_instance = DTDLValidatorService()
    return _validator_instance


__all__ = [
    "DTDLValidatorService",
    "get_dtdl_validator",
    "ValidationResult",
    "ValidationIssue",
    "ValidationSeverity"
]
