"""
DTDL API Router

Endpoints for browsing, searching, and retrieving DTDL interfaces from the library.
Supports interface suggestion based on TwinScale Thing data.
"""

from fastapi import APIRouter, HTTPException, Query, Depends
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field
import logging

from app.services.dtdl_loader_service import get_dtdl_loader
from app.services.dtdl_validator_service import get_dtdl_validator, ValidationSeverity
from app.services.dtdl_converter_service import get_dtdl_converter
from app.api.deps import get_current_user

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/dtdl", tags=["DTDL"])


# Request/Response Models
class InterfaceListResponse(BaseModel):
    """Response model for interface listing"""
    total: int = Field(description="Total number of interfaces found")
    interfaces: List[Dict[str, Any]] = Field(description="List of interface metadata")


class InterfaceDetailResponse(BaseModel):
    """Response model for interface details"""
    interface: Dict[str, Any] = Field(description="Full DTDL interface JSON")
    summary: Optional[Dict[str, Any]] = Field(description="Content summary (counts)")


class InterfaceSuggestionRequest(BaseModel):
    """Request model for interface suggestion"""
    thing_type: str = Field(description="Thing type: device, sensor, or component")
    domain: Optional[str] = Field(None, description="Domain name (e.g., environmental, air_quality)")
    properties: Optional[List[str]] = Field(None, description="List of property names")
    telemetry: Optional[List[str]] = Field(None, description="List of telemetry names")
    keywords: Optional[str] = Field(None, description="Keywords to search in interface descriptions")


class InterfaceSuggestionResponse(BaseModel):
    """Response model for interface suggestion"""
    suggested: List[Dict[str, Any]] = Field(description="Suggested interfaces with match scores")
    base_interface: Optional[str] = Field(None, description="Recommended base interface DTMI")


@router.get("/interfaces", response_model=InterfaceListResponse)
async def list_interfaces(
    thing_type: Optional[str] = Query(None, description="Filter by thing type (device, sensor, component)"),
    domain: Optional[str] = Query(None, description="Filter by domain (environmental, air_quality, etc.)"),
    category: Optional[str] = Query(None, description="Filter by category (base, environmental, etc.)"),
    tags: Optional[str] = Query(None, description="Filter by tags (comma-separated, AND logic)"),
    keywords: Optional[str] = Query(None, description="Search in displayName and description"),
    current_user: Dict = Depends(get_current_user)
):
    """
    List all DTDL interfaces with optional filtering

    Supports filtering by:
    - thing_type: device, sensor, component
    - domain: environmental, air_quality, etc.
    - category: base, environmental, etc.
    - tags: comma-separated list (must match all)
    - keywords: search in displayName and description
    """
    try:
        loader = get_dtdl_loader()

        # Parse tags if provided
        tag_list = None
        if tags:
            tag_list = [t.strip() for t in tags.split(",") if t.strip()]

        # Search interfaces
        results = loader.search_interfaces(
            thing_type=thing_type,
            domain=domain,
            category=category,
            tags=tag_list,
            keywords=keywords
        )

        logger.info(f"Found {len(results)} interfaces matching filters (user: {current_user.get('sub')})")

        return InterfaceListResponse(
            total=len(results),
            interfaces=results
        )

    except Exception as e:
        logger.error(f"Failed to list interfaces: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to list interfaces: {str(e)}")


@router.get("/interfaces/{dtmi:path}", response_model=InterfaceDetailResponse)
async def get_interface_details(
    dtmi: str,
    current_user: Dict = Depends(get_current_user)
):
    """
    Get detailed information for a specific DTDL interface

    Returns:
    - Full DTDL interface JSON
    - Content summary (telemetry, property, command counts)
    - Registry metadata
    """
    try:
        loader = get_dtdl_loader()

        # Validate DTMI format
        if not loader.validate_dtmi(dtmi):
            raise HTTPException(status_code=400, detail=f"Invalid DTMI format: {dtmi}")

        # Get interface details
        interface = loader.get_interface_details(dtmi)

        if not interface:
            raise HTTPException(status_code=404, detail=f"Interface not found: {dtmi}")

        logger.info(f"Retrieved interface details for {dtmi} (user: {current_user.get('sub')})")

        # Extract summary if available
        summary = interface.pop("_summary", None)

        return InterfaceDetailResponse(
            interface=interface,
            summary=summary
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get interface details for {dtmi}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get interface details: {str(e)}")


@router.post("/suggest", response_model=InterfaceSuggestionResponse)
async def suggest_interfaces(
    request: InterfaceSuggestionRequest,
    current_user: Dict = Depends(get_current_user)
):
    """
    Suggest DTDL interfaces based on TwinScale Thing data

    Analyzes thing_type, domain, properties, telemetry, and keywords
    to recommend appropriate DTDL interfaces.

    Returns:
    - List of suggested interfaces with match scores
    - Recommended base interface DTMI
    """
    try:
        loader = get_dtdl_loader()

        # Get base interface recommendation
        base_interface = loader.get_base_for_thing_type(request.thing_type)

        # Search for matching interfaces
        results = loader.search_interfaces(
            thing_type=request.thing_type,
            domain=request.domain,
            keywords=request.keywords
        )

        # Calculate match scores for each interface
        suggested = []
        for interface_def in results:
            score = 0
            reasons = []

            # Match thing_type
            if interface_def.get("thingType") == request.thing_type:
                score += 10
                reasons.append("thing_type match")

            # Match domain
            if request.domain and loader._is_in_domain_mapping(interface_def["dtmi"], request.domain):
                score += 10
                reasons.append("domain match")

            # Match telemetry
            if request.telemetry:
                interface_telemetry = interface_def.get("telemetry", [])
                matching_telemetry = set(request.telemetry) & set(interface_telemetry)
                if matching_telemetry:
                    score += len(matching_telemetry) * 5
                    reasons.append(f"{len(matching_telemetry)} telemetry match(es)")

            # Match properties
            if request.properties:
                interface_properties = interface_def.get("properties", [])
                matching_properties = set(request.properties) & set(interface_properties)
                if matching_properties:
                    score += len(matching_properties) * 3
                    reasons.append(f"{len(matching_properties)} property match(es)")

            # Match keywords
            if request.keywords:
                keywords_lower = request.keywords.lower()
                display_name = interface_def.get("displayName", "").lower()
                description = interface_def.get("description", "").lower()

                if keywords_lower in display_name:
                    score += 5
                    reasons.append("keyword in displayName")
                elif keywords_lower in description:
                    score += 3
                    reasons.append("keyword in description")

            suggested.append({
                **interface_def,
                "matchScore": score,
                "matchReasons": reasons
            })

        # Sort by match score (descending)
        suggested.sort(key=lambda x: x["matchScore"], reverse=True)

        logger.info(f"Suggested {len(suggested)} interfaces for thing_type={request.thing_type}, "
                   f"domain={request.domain} (user: {current_user.get('sub')})")

        return InterfaceSuggestionResponse(
            suggested=suggested,
            base_interface=base_interface
        )

    except Exception as e:
        logger.error(f"Failed to suggest interfaces: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to suggest interfaces: {str(e)}")


@router.get("/domains", response_model=Dict[str, List[str]])
async def list_domains(
    current_user: Dict = Depends(get_current_user)
):
    """
    List all available domains with their associated DTMIs

    Returns:
    - Dictionary mapping domain names to list of DTMIs
    """
    try:
        loader = get_dtdl_loader()

        if not loader._registry_cache:
            return {}

        domain_mapping = loader._registry_cache.get("domainMapping", {})

        logger.info(f"Retrieved {len(domain_mapping)} domains (user: {current_user.get('sub')})")

        return domain_mapping

    except Exception as e:
        logger.error(f"Failed to list domains: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to list domains: {str(e)}")


@router.get("/thing-types", response_model=Dict[str, List[str]])
async def list_thing_types(
    current_user: Dict = Depends(get_current_user)
):
    """
    List all thing types with their associated DTMIs

    Returns:
    - Dictionary mapping thing types to list of DTMIs
    """
    try:
        loader = get_dtdl_loader()

        if not loader._registry_cache:
            return {}

        thing_type_mapping = loader._registry_cache.get("thingTypeMapping", {})

        logger.info(f"Retrieved {len(thing_type_mapping)} thing types (user: {current_user.get('sub')})")

        return thing_type_mapping

    except Exception as e:
        logger.error(f"Failed to list thing types: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to list thing types: {str(e)}")


@router.post("/validate")
async def validate_thing(
    request: Dict[str, Any],
    current_user: Dict = Depends(get_current_user)
):
    """
    Validate a TwinScale Thing against a DTDL interface

    Request Body:
    - thing_data: Thing properties and telemetry
    - dtmi: DTDL interface identifier
    - strict: Whether to treat extra fields as errors (optional, default: false)

    Returns:
    - Validation result with compatibility score and issues
    """
    try:
        validator = get_dtdl_validator()

        thing_data = request.get("thing_data")
        dtmi = request.get("dtmi")
        strict = request.get("strict", False)

        if not thing_data:
            raise HTTPException(status_code=400, detail="thing_data is required")
        if not dtmi:
            raise HTTPException(status_code=400, detail="dtmi is required")

        result = validator.validate_thing_against_interface(thing_data, dtmi, strict)

        logger.info(f"Validated thing against {dtmi}, score: {result.compatibility_score} "
                   f"(user: {current_user.get('sub')})")

        return {
            "is_compatible": result.is_compatible,
            "compatibility_score": result.compatibility_score,
            "dtmi": result.dtmi,
            "interface_name": result.interface_name,
            "issues": [
                {
                    "severity": issue.severity.value,
                    "field": issue.field,
                    "message": issue.message,
                    "suggestion": issue.suggestion
                }
                for issue in result.issues
            ],
            "matched_telemetry": result.matched_telemetry,
            "matched_properties": result.matched_properties,
            "missing_telemetry": result.missing_telemetry,
            "missing_properties": result.missing_properties,
            "extra_fields": result.extra_fields
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to validate thing: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to validate thing: {str(e)}")


@router.post("/find-best-match")
async def find_best_match(
    request: Dict[str, Any],
    current_user: Dict = Depends(get_current_user)
):
    """
    Find best matching DTDL interfaces for a Thing

    Request Body:
    - thing_data: Thing properties and telemetry
    - thing_type: Optional thing type filter (device, sensor, component)
    - domain: Optional domain filter (environmental, air_quality, etc.)
    - top_n: Number of top results (optional, default: 5)

    Returns:
    - List of validation results sorted by combined score
    """
    try:
        validator = get_dtdl_validator()

        thing_data = request.get("thing_data")
        thing_type = request.get("thing_type")
        domain = request.get("domain")
        top_n = request.get("top_n", 5)

        if not thing_data:
            raise HTTPException(status_code=400, detail="thing_data is required")

        results = validator.find_best_matching_interfaces(
            thing_data=thing_data,
            thing_type=thing_type,
            domain=domain,
            top_n=top_n
        )

        logger.info(f"Found {len(results)} matching interfaces for thing_type={thing_type}, "
                   f"domain={domain} (user: {current_user.get('sub')})")

        return {
            "matches": [
                {
                    "validation": {
                        "is_compatible": validation.is_compatible,
                        "compatibility_score": validation.compatibility_score,
                        "dtmi": validation.dtmi,
                        "interface_name": validation.interface_name,
                        "matched_telemetry": validation.matched_telemetry,
                        "matched_properties": validation.matched_properties,
                        "missing_telemetry": validation.missing_telemetry,
                        "missing_properties": validation.missing_properties,
                        "issues_count": len(validation.issues)
                    },
                    "combined_score": combined_score
                }
                for validation, combined_score in results
            ]
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to find best match: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to find best match: {str(e)}")


@router.get("/interfaces/{dtmi:path}/requirements")
async def get_interface_requirements(
    dtmi: str,
    current_user: Dict = Depends(get_current_user)
):
    """
    Get requirements summary for a DTDL interface

    Returns:
    - Required and optional telemetry fields
    - Required and optional property fields
    - Schema information for each field
    """
    try:
        validator = get_dtdl_validator()

        requirements = validator.get_interface_requirements(dtmi)

        if not requirements:
            raise HTTPException(status_code=404, detail=f"Interface not found: {dtmi}")

        logger.info(f"Retrieved requirements for {dtmi} (user: {current_user.get('sub')})")

        return requirements

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get interface requirements: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get interface requirements: {str(e)}")


@router.post("/convert/to-twinscale")
async def convert_to_twinscale(
    request: Dict[str, Any],
    current_user: Dict = Depends(get_current_user)
):
    """
    Convert DTDL interface to TwinScale YAML template

    Request Body:
    - dtmi: DTDL interface identifier (required)
    - thing_name: Optional Thing name (default: interface displayName)
    - tenant_id: Optional tenant ID

    Returns:
    - interface_yaml: TwinInterface YAML template
    - instance_yaml: TwinInstance YAML template
    """
    try:
        converter = get_dtdl_converter()

        dtmi = request.get("dtmi")
        thing_name = request.get("thing_name")
        tenant_id = request.get("tenant_id")

        if not dtmi:
            raise HTTPException(status_code=400, detail="dtmi is required")

        result = converter.dtdl_to_twinscale_template(dtmi, thing_name, tenant_id)

        logger.info(f"Converted {dtmi} to TwinScale template (user: {current_user.get('sub')})")

        return result

    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to convert to TwinScale: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to convert to TwinScale: {str(e)}")


@router.post("/enrich")
async def enrich_with_dtdl(
    request: Dict[str, Any],
    current_user: Dict = Depends(get_current_user)
):
    """
    Enrich TwinScale Thing with DTDL metadata

    Request Body:
    - thing_data: TwinScale Thing data (required)
    - dtmi: DTDL interface identifier (required)

    Returns:
    - Enriched Thing data with DTDL annotations
    """
    try:
        converter = get_dtdl_converter()

        thing_data = request.get("thing_data")
        dtmi = request.get("dtmi")

        if not thing_data:
            raise HTTPException(status_code=400, detail="thing_data is required")
        if not dtmi:
            raise HTTPException(status_code=400, detail="dtmi is required")

        enriched = converter.enrich_twinscale_with_dtdl(thing_data, dtmi)

        logger.info(f"Enriched Thing with {dtmi} metadata (user: {current_user.get('sub')})")

        return enriched

    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to enrich with DTDL: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to enrich with DTDL: {str(e)}")


@router.get("/interfaces/{dtmi:path}/summary")
async def get_interface_summary(
    dtmi: str,
    current_user: Dict = Depends(get_current_user)
):
    """
    Get interface summary for UI display

    Returns:
    - Summary with counts and field names
    """
    try:
        converter = get_dtdl_converter()

        summary = converter.get_interface_summary(dtmi)

        if not summary:
            raise HTTPException(status_code=404, detail=f"Interface not found: {dtmi}")

        logger.info(f"Retrieved summary for {dtmi} (user: {current_user.get('sub')})")

        return summary

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get interface summary: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get interface summary: {str(e)}")


@router.post("/reload")
async def reload_library(
    current_user: Dict = Depends(get_current_user)
):
    """
    Reload DTDL library (for development/testing)

    Clears cache and reloads registry and all interface files.
    Requires authentication.
    """
    try:
        loader = get_dtdl_loader()
        loader.reload()

        logger.info(f"DTDL library reloaded (user: {current_user.get('sub')})")

        return {
            "status": "success",
            "message": "DTDL library reloaded successfully",
            "interfaces_count": len(loader._interfaces_cache)
        }

    except Exception as e:
        logger.error(f"Failed to reload library: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to reload library: {str(e)}")
