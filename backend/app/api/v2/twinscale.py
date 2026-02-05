"""
TwinScale-Lite API Endpoints

Direct form-to-YAML-to-RDF workflow without WoT/Ditto dependencies.
"""
import logging
from typing import Optional, Dict, Any, Literal, List
from fastapi import APIRouter, Depends, HTTPException, status, Response, Query, Path
from fastapi.responses import StreamingResponse
import io
import zipfile

from app.services.twinscale_generator_service import TwinScaleGeneratorService
from app.services.twinscale_rdf_service import TwinScaleRDFService
from app.models.twinscale_models import ValidationResult
from app.api.dependencies import get_tenant_id
from pydantic import BaseModel


router = APIRouter()
logger = logging.getLogger(__name__)


# ============================================================================
# Request/Response Models
# ============================================================================

class TwinScaleQueryRequest(BaseModel):
    """Request model for SPARQL queries"""
    query: str
    limit: Optional[int] = 100


class InterfaceListResponse(BaseModel):
    """Response model for interface list"""
    interfaces: List[Dict[str, Any]]
    count: int


class InstanceListResponse(BaseModel):
    """Response model for instance list"""
    instances: List[Dict[str, Any]]
    count: int


# ============================================================================
# Direct Creation Models (Form → TwinScale YAML)
# ============================================================================

class TwinScalePropertyInput(BaseModel):
    """Property input from form"""
    name: str
    type: Literal["float", "integer", "string", "boolean", "object", "array"] = "string"
    description: Optional[str] = None
    writable: bool = False
    minimum: Optional[float] = None
    maximum: Optional[float] = None
    unit: Optional[str] = None


class TwinScaleRelationshipInput(BaseModel):
    """Relationship input from form"""
    name: str
    target_interface: str
    target_instance: Optional[str] = None
    description: Optional[str] = None


class TwinScaleCommandInput(BaseModel):
    """Command input from form"""
    name: str
    description: Optional[str] = None
    input_schema: Optional[Dict[str, Any]] = None


class TwinScaleCreateRequest(BaseModel):
    """
    Request model for creating TwinScale Thing directly from form data.
    """
    # Basic info
    id: str
    name: str
    description: Optional[str] = None

    # Properties
    properties: List[TwinScalePropertyInput] = []

    # Relationships
    relationships: List[TwinScaleRelationshipInput] = []

    # Location (Optional)
    latitude: Optional[float] = None
    longitude: Optional[float] = None

    # Commands/Actions
    commands: List[TwinScaleCommandInput] = []

    # Service configuration (optional)
    include_service_spec: bool = True
    service_image: Optional[str] = None

    # Store options
    store_in_rdf: bool = True

    # NEW: Thing Type (Phase 1)
    thing_type: Literal["device", "sensor", "component"] = "device"

    # NEW: Domain Metadata (Phase 1)
    manufacturer: Optional[str] = None
    model: Optional[str] = None
    serial_number: Optional[str] = None
    firmware_version: Optional[str] = None

    # NEW: DTDL Integration (Phase 2)
    dtdl_interface: Optional[Dict[str, Any]] = None  # Selected DTDL interface metadata


class TwinScaleCreateResponse(BaseModel):
    """Response model for TwinScale creation"""
    success: bool
    interface_name: str
    instance_name: str
    interface_yaml: str
    instance_yaml: str
    stored_in_rdf: bool
    message: str


class ValidateYamlRequest(BaseModel):
    """Request model for YAML validation"""
    yaml_content: str
    kind: Literal["TwinInterface", "TwinInstance"]


# ============================================================================
# Endpoints
# ============================================================================

@router.post(
    "/create",
    response_model=TwinScaleCreateResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Create TwinScale Thing directly from form data",
    description="Create TwinInterface and TwinInstance YAML from form input and store in RDF",
)
async def create_twinscale_thing(
    request: TwinScaleCreateRequest,
    tenant_id: str = Depends(get_tenant_id),
):
    """
    Create TwinScale Thing directly from form data.

    Creates both TwinInterface (blueprint) and TwinInstance (concrete) YAML files
    and optionally stores them in Fuseki RDF database.

    Flow:
    1. Receive form data (id, name, properties, relationships, commands)
    2. Generate TwinInterface YAML
    3. Generate TwinInstance YAML
    4. Store in Fuseki RDF (if store_in_rdf=True)
    5. Return YAML content
    """
    try:
        logger.info(f"Creating TwinScale Thing: {request.id}")

        # Build thing_description dict from form data
        thing_description = {
            "@id": request.id,
            "title": request.name,
            "description": request.description,
            "latitude": request.latitude,
            "longitude": request.longitude,
            "properties": {},
            "actions": {},
            "links": []
        }

        # Convert properties
        for prop in request.properties:
            thing_description["properties"][prop.name] = {
                "type": prop.type,
                "description": prop.description,
                "writable": prop.writable,
                "minimum": prop.minimum,
                "maximum": prop.maximum,
                "unit": prop.unit
            }

        # Convert commands/actions
        for cmd in request.commands:
            thing_description["actions"][cmd.name] = {
                "description": cmd.description,
                "input": cmd.input_schema or {}
            }

        # Convert relationships to links
        for rel in request.relationships:
            thing_description["links"].append({
                "rel": rel.name,
                "href": rel.target_interface,
                "title": rel.description
            })

        # Initialize generator
        generator = TwinScaleGeneratorService()

        # Generate YAML files with new parameters
        interface_yaml = generator.generate_twin_interface_yaml(
            thing_description,
            include_service_spec=request.include_service_spec,
            thing_type=request.thing_type,
            domain_metadata={
                "manufacturer": request.manufacturer,
                "model": request.model,
                "serial_number": request.serial_number,
                "firmware_version": request.firmware_version,
            },
            dtdl_interface=request.dtdl_interface
        )

        instance_yaml = generator.generate_twin_instance_yaml(thing_description)

        # Get normalized names
        interface_name = generator._normalize_name(request.id)
        instance_name = interface_name

        # Store in RDF if requested
        stored_in_rdf = False
        if request.store_in_rdf:
            try:
                logger.info(f"Storing TwinScale RDF for: {request.id}")
                rdf_service = TwinScaleRDFService()

                # Prepare metadata
                metadata = {
                    "tenant_id": tenant_id,
                    "name": request.name,
                    "description": request.description,
                    # NEW: Thing Type and Domain Metadata
                    "thing_type": request.thing_type,
                    "manufacturer": request.manufacturer,
                    "model": request.model,
                    "serial_number": request.serial_number,
                    "firmware_version": request.firmware_version,
                }

                # Add DTDL interface metadata if provided
                if request.dtdl_interface:
                    metadata["dtdl_interface"] = request.dtdl_interface.get("dtmi")
                    metadata["dtdl_interface_name"] = request.dtdl_interface.get("displayName")

                await rdf_service.store_twinscale_yaml(
                    interface_yaml=interface_yaml,
                    instance_yaml=instance_yaml,
                    thing_id=request.id,
                    metadata=metadata
                )
                stored_in_rdf = True
                logger.info(f"Successfully stored TwinScale RDF for: {request.id}")
            except Exception as rdf_error:
                logger.warning(f"Failed to store in RDF: {rdf_error}")

        return TwinScaleCreateResponse(
            success=True,
            interface_name=interface_name,
            instance_name=instance_name,
            interface_yaml=interface_yaml,
            instance_yaml=instance_yaml,
            stored_in_rdf=stored_in_rdf,
            message=f"TwinScale Thing '{request.name}' created successfully"
        )

    except ValueError as e:
        logger.error(f"Validation error: {e}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid input: {str(e)}"
        )
    except Exception as e:
        logger.error(f"Error creating TwinScale Thing: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to create TwinScale Thing: {str(e)}"
        )


@router.get(
    "/export/{interface_name}",
    summary="Export TwinScale YAML as ZIP",
    description="Download TwinInterface and TwinInstance YAML files as ZIP",
)
async def export_twinscale_zip(
    interface_name: str = Path(..., description="Name of the TwinInterface to export"),
    tenant_id: str = Depends(get_tenant_id),
):
    """
    Export TwinScale YAML files from RDF as ZIP.
    """
    try:
        rdf_service = TwinScaleRDFService()

        # Get interface details from RDF
        interface = await rdf_service.get_interface_details(interface_name, tenant_id=tenant_id)
        if not interface:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Interface '{interface_name}' not found"
            )

        # Regenerate YAML from stored data
        generator = TwinScaleGeneratorService()

        # Build thing_description from stored data
        thing_description = {
            "@id": interface.get("name", interface_name),
            "title": interface.get("name"),
            "description": interface.get("description"),
            "properties": {
                p["name"]: {
                    "type": p.get("type", "string"),
                    "description": p.get("description"),
                    "writable": p.get("writable", False)
                }
                for p in interface.get("properties", [])
            },
            "actions": {
                c["name"]: {"description": c.get("description")}
                for c in interface.get("commands", [])
            },
            "links": [
                {"rel": r["name"], "href": r.get("targetInterface", "")}
                for r in interface.get("relationships", [])
            ]
        }

        interface_yaml = generator.generate_twin_interface_yaml(thing_description)
        instance_yaml = generator.generate_twin_instance_yaml(thing_description)

        # Create ZIP
        zip_buffer = io.BytesIO()
        with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
            zip_file.writestr(f"{interface_name}_interface.yaml", interface_yaml)
            zip_file.writestr(f"{interface_name}_instance.yaml", instance_yaml)

        zip_buffer.seek(0)

        return StreamingResponse(
            zip_buffer,
            media_type="application/zip",
            headers={
                "Content-Disposition": f'attachment; filename="{interface_name}_twinscale.zip"'
            }
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error exporting TwinScale: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to export: {str(e)}"
        )


@router.post(
    "/validate",
    response_model=ValidationResult,
    summary="Validate TwinScale YAML",
    description="Validate a TwinScale YAML file (TwinInterface or TwinInstance)",
)
async def validate_twinscale_yaml(
    request: ValidateYamlRequest,
    tenant_id: str = Depends(get_tenant_id),
):
    """
    Validate TwinScale YAML content.
    """
    try:
        generator = TwinScaleGeneratorService()
        validation_result = generator.validate_twinscale_yaml(request.yaml_content, request.kind)
        return validation_result

    except Exception as e:
        logger.error(f"Error validating TwinScale YAML: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Validation error: {str(e)}"
        )


# ============================================================================
# RDF Query Endpoints
# ============================================================================

@router.get(
    "/rdf/interfaces",
    response_model=InterfaceListResponse,
    summary="Query TwinScale Interfaces from RDF",
)
async def query_interfaces(
    name_filter: Optional[str] = Query(None, description="Filter by interface name"),
    limit: int = Query(100, ge=1, le=1000, description="Maximum results"),
    tenant_id: str = Depends(get_tenant_id),
):
    """Query TwinInterfaces from RDF database."""
    try:
        rdf_service = TwinScaleRDFService()
        interfaces = await rdf_service.query_interfaces(
            name_filter=name_filter,
            limit=limit,
            tenant_id=tenant_id
        )

        return InterfaceListResponse(
            interfaces=interfaces,
            count=len(interfaces)
        )

    except Exception as e:
        logger.warning(f"Fuseki not available or error querying interfaces: {e}")
        # Return empty list if Fuseki is not available
        return InterfaceListResponse(
            interfaces=[],
            count=0
        )


@router.get(
    "/rdf/instances",
    response_model=InstanceListResponse,
    summary="Query TwinScale Instances from RDF",
)
async def query_instances(
    interface_name: Optional[str] = Query(None, description="Filter by interface"),
    limit: int = Query(100, ge=1, le=1000, description="Maximum results"),
    tenant_id: str = Depends(get_tenant_id),
):
    """Query TwinInstances from RDF database."""
    try:
        rdf_service = TwinScaleRDFService()
        instances = await rdf_service.query_instances(
            interface_name=interface_name,
            limit=limit,
            tenant_id=tenant_id
        )

        return InstanceListResponse(
            instances=instances,
            count=len(instances)
        )

    except Exception as e:
        logger.warning(f"Fuseki not available or error querying instances: {e}")
        # Return empty list if Fuseki is not available
        return InstanceListResponse(
            instances=[],
            count=0
        )


@router.get(
    "/rdf/interfaces/{interface_name}",
    summary="Get TwinScale Interface Details",
)
async def get_interface_details(
    interface_name: str = Path(..., description="Name of the TwinInterface"),
    tenant_id: str = Depends(get_tenant_id),
):
    """Get detailed information about a TwinInterface."""
    try:
        rdf_service = TwinScaleRDFService()
        interface = await rdf_service.get_interface_details(
            interface_name,
            tenant_id=tenant_id
        )

        if not interface:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Interface '{interface_name}' not found"
            )

        return interface

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting interface details: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get interface details: {str(e)}"
        )


@router.get(
    "/rdf/instances/{instance_name}/relationships",
    summary="Get TwinInstance Relationships",
)
async def get_instance_relationships(
    instance_name: str = Path(..., description="Name of the TwinInstance"),
    tenant_id: str = Depends(get_tenant_id),
):
    """Get all relationships for a TwinInstance."""
    try:
        rdf_service = TwinScaleRDFService()
        relationships = await rdf_service.get_instance_relationships(
            instance_name,
            tenant_id=tenant_id
        )

        return {
            "instance": instance_name,
            "relationships": relationships,
            "count": len(relationships)
        }

    except Exception as e:
        logger.error(f"Error getting instance relationships: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get instance relationships: {str(e)}"
        )


@router.post(
    "/rdf/query",
    summary="Execute Custom SPARQL Query",
)
async def execute_sparql_query(
    request: TwinScaleQueryRequest,
    tenant_id: str = Depends(get_tenant_id),
):
    """Execute a custom SPARQL SELECT query."""
    try:
        if not request.query.strip().upper().startswith("SELECT"):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Only SELECT queries are allowed"
            )

        rdf_service = TwinScaleRDFService()
        results = await rdf_service._execute_query(request.query)
        parsed_results = rdf_service._parse_sparql_results(results)

        if request.limit:
            parsed_results = parsed_results[:request.limit]

        return {
            "query": request.query,
            "results": parsed_results,
            "count": len(parsed_results)
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error executing SPARQL query: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to execute SPARQL query: {str(e)}"
        )


@router.delete(
    "/rdf/interfaces/{interface_name}",
    summary="Delete TwinScale Interface from RDF",
)
async def delete_twinscale_interface(
    interface_name: str = Path(..., description="Name of the TwinInterface to delete"),
    tenant_id: str = Depends(get_tenant_id),
):
    """Delete a TwinInterface and all its instances from RDF."""
    try:
        rdf_service = TwinScaleRDFService()
        success = await rdf_service.delete_twinscale(interface_name, tenant_id=tenant_id)

        if success:
            return {
                "message": f"Interface '{interface_name}' deleted successfully",
                "interface": interface_name
            }
        else:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to delete interface"
            )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting interface: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to delete interface: {str(e)}"
        )


__all__ = ["router"]
