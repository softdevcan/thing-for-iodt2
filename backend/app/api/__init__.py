"""
TwinScale-Lite API Router

Simplified API with TwinScale, Tenant, and DTDL endpoints.
"""

from fastapi import APIRouter
from .v2 import twinscale, tenants, dtdl

# Create main API router
api_router = APIRouter()

# Include TwinScale routes
api_router.include_router(
    twinscale.router,
    prefix="/v2/twinscale",
    tags=["twinscale"]
)

# Include Tenant routes
api_router.include_router(
    tenants.router,
    prefix="/v2/tenants",
    tags=["tenants"]
)

# Include DTDL routes
api_router.include_router(
    dtdl.router,
    prefix="/v2"
)


@api_router.get("/test")
async def test_api():
    return {"message": "TwinScale-Lite API is working"}
