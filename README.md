# IoDT2 Thing Description

A DTDL (Digital Twins Definition Language) based digital twin management system. Create, manage, and query digital twins using human-readable YAML format.

## Quick Start

```bash
# 1. Clone the project
git clone <repository-url>
cd thing-for-twinscale

# 2. Start with Docker (Recommended)
docker-compose up -d

# 3. Open in your browser
# Frontend: http://localhost:3005
# API Docs: http://localhost:3015/docs
```

---

## Table of Contents

- [Features](#features)
- [Architecture](#architecture)
- [Installation](#installation)
  - [Docker Installation (Recommended)](#docker-installation-recommended)
  - [Manual Installation](#manual-installation)
- [Usage](#usage)
  - [DTDL Operations](#dtdl-operations)
  - [Thing Management](#thing-management)
  - [YAML Queries](#yaml-queries)
- [DTDL Library](#dtdl-library)
- [API Documentation](#api-documentation)
- [Testing](#testing)
- [Troubleshooting](#troubleshooting)
- [Project Structure](#project-structure)

---

## Features

- **DTDL Support**: Compatible with standard DTDL (v3) interfaces
- **YAML Format**: Human-readable YAML format for digital twin definitions
- **Modular Library**: Environmental sensors, seismic detectors, air quality sensors
- **Validation**: Automatic validation against DTDL schemas
- **Conversion**: Bidirectional DTDL <-> YAML conversion
- **REST API**: Modern REST API powered by FastAPI
- **React Frontend**: User-friendly web interface

---

## Architecture

### Backend (Python/FastAPI)

```
backend/
├── app/
│   ├── api/              # API endpoints
│   │   └── v2/
│   │       ├── dtdl.py   # DTDL interface management
│   │       ├── twin.py   # Thing management
│   │       ├── fuseki.py # Fuseki RDF operations
│   │       └── tenants.py # Tenant management
│   ├── services/         # Business logic services
│   │   ├── dtdl_loader_service.py      # DTDL loading and caching
│   │   ├── dtdl_converter_service.py   # DTDL <-> YAML conversion
│   │   ├── dtdl_validator_service.py   # DTDL validation
│   │   ├── twin_generator_service.py   # Thing generation
│   │   ├── twin_rdf_service.py         # RDF operations
│   │   ├── tenant_manager.py           # Tenant management
│   │   └── location_service.py         # Location service
│   ├── dtdl_library/     # DTDL interface library
│   │   ├── base/         # Base interfaces (BaseTwin, SensorTwin, etc.)
│   │   ├── domain/       # Domain-specific interfaces
│   │   │   ├── environmental/  # Environmental sensors
│   │   │   ├── air_quality/    # Air quality sensors
│   │   │   └── seismic/        # Seismic detectors
│   │   └── registry.json # Interface registry
│   ├── models/           # SQLAlchemy models
│   ├── schemas/          # Pydantic schemas
│   └── core/             # Core configuration
├── tests/                # Test files
├── scripts/              # Utility scripts
└── main.py               # Application entry point
```

### Frontend (React/Vite)

```
frontend/
├── src/
│   ├── api/              # API clients
│   ├── components/       # React components
│   │   ├── dtdl/         # DTDL components
│   │   └── twin/         # Thing components
│   ├── pages/            # Page components
│   │   └── twin/         # Thing pages
│   ├── services/         # Service layer
│   └── locales/          # i18n translations
└── ...
```

---

## Installation

### Docker Installation (Recommended)

#### Prerequisites
- **Docker** 20.10+
- **Docker Compose** 2.0+

#### Steps

```bash
# 1. Start all services (Backend, Frontend, Fuseki)
docker-compose up -d

# 2. Watch logs (optional)
docker-compose logs -f

# 3. Check status
docker-compose ps
```

Services will be available at:

- **Frontend**: http://localhost:3005
- **Backend API**: http://localhost:3015
- **Fuseki (RDF Store)**: http://localhost:3030
- **API Docs**: http://localhost:3015/docs

#### Docker Commands

```bash
# Stop services
docker-compose down

# Stop services and remove volumes
docker-compose down -v

# Restart services
docker-compose restart

# Restart a specific service
docker-compose restart backend

# View logs
docker-compose logs backend
docker-compose logs frontend
docker-compose logs fuseki

# Attach to a container (debugging)
docker-compose exec backend bash
docker-compose exec frontend sh

# Rebuild services (after code changes)
docker-compose up -d --build
```

#### Health Check

All services come with automatic health check configuration:

```bash
docker-compose ps

# Detailed health status
docker inspect iodt2-thing-backend | grep -A 10 "Health"
docker inspect iodt2-thing-frontend | grep -A 10 "Health"
docker inspect iodt2-thing-fuseki | grep -A 10 "Health"
```

---

### Manual Installation

For local development:

#### Prerequisites

- **Python** 3.9+
- **Node.js** 16+
- **npm** or **yarn**

#### Backend Setup

```bash
cd backend

# Create virtual environment
python -m venv venv

# Activate virtual environment
# Windows:
venv\Scripts\activate
# Linux/Mac:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Configure environment variables
cp .env.example .env
# Edit .env as needed
```

#### Frontend Setup

```bash
cd frontend

# Install dependencies
npm install

# Configure environment variables
cp .env.example .env
# Edit .env as needed
```

#### Running Locally

```bash
# Backend (http://localhost:3015)
cd backend
python main.py

# Frontend (http://localhost:5173)
cd frontend
npm run dev
```

---

## DTDL Operations

### 1. List DTDL Interfaces

**Endpoint:** `GET /api/v2/dtdl/interfaces`

```bash
curl http://localhost:3015/api/v2/dtdl/interfaces
```

**Response:**
```json
[
  {
    "dtmi": "dtmi:twinscale:BaseTwin;1",
    "displayName": "Base Digital Twin",
    "description": "Base interface for all digital twins",
    "category": "base"
  },
  {
    "dtmi": "dtmi:twinscale:SensorTwin;1",
    "displayName": "Sensor Twin",
    "description": "Base interface for sensor digital twins",
    "category": "base"
  }
]
```

### 2. Get Interface Details

**Endpoint:** `GET /api/v2/dtdl/interfaces/{dtmi}`

```bash
curl "http://localhost:3015/api/v2/dtdl/interfaces/dtmi:twinscale:environmental:TemperatureSensor;1"
```

### 3. Get Interface Requirements

**Endpoint:** `GET /api/v2/dtdl/interfaces/{dtmi}/requirements`

```bash
curl "http://localhost:3015/api/v2/dtdl/interfaces/dtmi:twinscale:environmental:TemperatureSensor;1/requirements"
```

**Response:**
```json
{
  "requiredProperties": [
    {
      "name": "temperature",
      "schema": "double",
      "description": "Current temperature reading in Celsius"
    }
  ],
  "requiredTelemetry": [
    {
      "name": "temperatureReading",
      "schema": "double"
    }
  ],
  "requiredCommands": [],
  "inheritedFrom": ["dtmi:twinscale:SensorTwin;1"]
}
```

### 4. Generate YAML Template from DTDL

**Endpoint:** `POST /api/v2/dtdl/convert/to-twinscale`

```bash
curl -X POST "http://localhost:3015/api/v2/dtdl/convert/to-twinscale" \
  -H "Content-Type: application/json" \
  -d '{
    "dtmi": "dtmi:twinscale:environmental:TemperatureSensor;1",
    "thing_name": "OfficeTemperatureSensor",
    "tenant_id": "my-tenant"
  }'
```

---

## Thing Management

### 1. Create a Thing (YAML)

**Endpoint:** `POST /api/v2/twin/things`

**YAML Thing Definition:**
```yaml
thing:
  name: MyOfficeTemperatureSensor
  interface: TemperatureSensor
  version: "1.0"
  tenant: office-building
  properties:
    temperature: 23.5
    unit: celsius
    location: "Office Room 101"
  metadata:
    dtdl:
      dtmi: dtmi:twinscale:environmental:TemperatureSensor;1
    created_by: user@example.com
```

```bash
curl -X POST "http://localhost:3015/api/v2/twin/things" \
  -H "Content-Type: application/x-yaml" \
  --data-binary @thing.yaml
```

### 2. List Things

**Endpoint:** `GET /api/v2/twin/things`

**Query Parameters:**
- `tenant`: Filter by tenant ID
- `interface`: Filter by interface name
- `limit`: Limit number of results (default: 100)
- `offset`: Pagination offset

```bash
# All things
curl "http://localhost:3015/api/v2/twin/things"

# Filter by tenant
curl "http://localhost:3015/api/v2/twin/things?tenant=office-building"

# Filter by interface
curl "http://localhost:3015/api/v2/twin/things?interface=TemperatureSensor"
```

### 3. Get Thing Details

**Endpoint:** `GET /api/v2/twin/things/{thing_id}`

```bash
curl "http://localhost:3015/api/v2/twin/things/1"
```

### 4. Update a Thing

**Endpoint:** `PUT /api/v2/twin/things/{thing_id}`

```bash
curl -X PUT "http://localhost:3015/api/v2/twin/things/1" \
  -H "Content-Type: application/x-yaml" \
  --data-binary @updated-thing.yaml
```

### 5. Delete a Thing

**Endpoint:** `DELETE /api/v2/twin/things/{thing_id}`

```bash
curl -X DELETE "http://localhost:3015/api/v2/twin/things/1"
```

---

## YAML Queries

Search key-value pairs within YAML content:

**Endpoint:** `GET /api/v2/twin/things/search`

```bash
# Find all sensors with a specific temperature value
curl "http://localhost:3015/api/v2/twin/things/search?query=properties.temperature&value=23.5"

# Find all devices at a specific location
curl "http://localhost:3015/api/v2/twin/things/search?query=properties.location&value=Office%20Room%20101"
```

---

## DTDL Library

### Available Interfaces

#### Base Interfaces
- **BaseTwin** (`dtmi:twinscale:BaseTwin;1`) - Base interface for all digital twins
- **SensorTwin** (`dtmi:twinscale:SensorTwin;1`) - Base interface for sensors
- **ActuatorTwin** (`dtmi:twinscale:ActuatorTwin;1`) - Base interface for actuators
- **GatewayTwin** (`dtmi:twinscale:GatewayTwin;1`) - Base interface for gateways

#### Environmental Sensors
- **TemperatureSensor** (`dtmi:twinscale:environmental:TemperatureSensor;1`)
- **HumiditySensor** (`dtmi:twinscale:environmental:HumiditySensor;1`)
- **WeatherStation** (`dtmi:twinscale:environmental:WeatherStation;1`)

#### Air Quality Sensors
- **PM25Sensor** (`dtmi:twinscale:air_quality:PM25Sensor;1`)

#### Seismic Sensors
- **Building** (`dtmi:twinscale:seismic:Building;1`)
- **Street** (`dtmi:twinscale:seismic:Street;1`)
- **BaseStation** (`dtmi:twinscale:seismic:BaseStation;1`)
- **SeismicSensor** (`dtmi:twinscale:seismic:SeismicSensor;1`)

### Adding a New DTDL Interface

1. **Create the DTDL JSON file:**

```json
{
  "@context": "dtmi:dtdl:context;3",
  "@id": "dtmi:twinscale:domain:MySensor;1",
  "@type": "Interface",
  "displayName": "My Custom Sensor",
  "description": "Description of my sensor",
  "extends": "dtmi:twinscale:SensorTwin;1",
  "contents": [
    {
      "@type": "Property",
      "name": "myProperty",
      "schema": "double",
      "description": "My custom property"
    }
  ]
}
```

2. **Save it to the appropriate directory:**
```
backend/app/dtdl_library/domain/my_domain/MySensor.json
```

3. **Update `registry.json`:**
```json
{
  "dtmi": "dtmi:twinscale:domain:MySensor;1",
  "displayName": "My Custom Sensor",
  "description": "Description of my sensor",
  "filePath": "domain/my_domain/MySensor.json",
  "category": "domain",
  "tags": ["sensor", "custom"]
}
```

4. **Restart the backend:**
```bash
python main.py
```

---

## API Documentation

When the backend is running, interactive API documentation is available at:

- **Swagger UI**: http://localhost:3015/docs
- **ReDoc**: http://localhost:3015/redoc

---

## Testing

```bash
cd backend

# Run all tests
pytest tests/

# Run a specific test file
pytest tests/test_dtdl_loader.py

# Verbose output
pytest tests/ -v

# With coverage report
pytest tests/ --cov=app
```

### Testing in Docker

```bash
docker-compose exec backend pytest tests/ -v
```

---

## Troubleshooting

### Containers not starting

```bash
# Check logs
docker-compose logs

# Check a specific service
docker-compose logs backend

# Clean up and restart
docker-compose down -v
docker-compose up -d --build
```

### Port already in use

```bash
# Windows:
netstat -ano | findstr :3015
netstat -ano | findstr :3005
netstat -ano | findstr :3030

# Linux/Mac:
lsof -i :3015
lsof -i :3005
lsof -i :3030
```

### Backend cannot connect to Fuseki

```bash
# Fuseki health check
curl http://localhost:3030/$/ping

# Check Fuseki logs
docker-compose logs fuseki

# Network inspection
docker network inspect iodt2-network
```

### Frontend cannot reach backend

```bash
# Test backend connection from container
docker-compose exec frontend curl http://backend:3015/health

# Rebuild frontend
docker-compose up -d --build frontend
```

### DTDL interface not loading

```bash
# Check registry file
cat backend/app/dtdl_library/registry.json

# Verify DTDL files exist
ls -la backend/app/dtdl_library/domain/

# Check backend logs
docker-compose logs backend | grep -i dtdl
```

### CORS error

```bash
# Check backend .env
grep CORS backend/.env

# Update CORS_ORIGINS environment variable
CORS_ORIGINS=http://localhost,http://localhost:3005,http://localhost:5173
```

---

## Project Structure

```
thing-for-twinscale/
├── backend/                      # Python/FastAPI backend
│   ├── app/
│   │   ├── api/v2/              # API endpoints
│   │   │   ├── dtdl.py          # DTDL API
│   │   │   ├── twin.py          # Thing API
│   │   │   ├── fuseki.py        # Fuseki API
│   │   │   └── tenants.py       # Tenant API
│   │   ├── core/                # Core configuration
│   │   │   ├── config.py
│   │   │   ├── database.py
│   │   │   └── twin_ontology.py
│   │   ├── dtdl_library/        # DTDL interface library
│   │   │   ├── base/            # Base interfaces
│   │   │   ├── domain/          # Domain-specific interfaces
│   │   │   └── registry.json
│   │   ├── models/              # SQLAlchemy ORM models
│   │   ├── schemas/             # Pydantic schemas
│   │   └── services/            # Business logic services
│   ├── tests/
│   ├── scripts/
│   ├── main.py
│   ├── requirements.txt
│   └── Dockerfile
│
├── frontend/                    # React/Vite frontend
│   ├── src/
│   │   ├── api/                 # API clients
│   │   ├── components/          # React components
│   │   │   ├── dtdl/
│   │   │   └── twin/
│   │   ├── pages/twin/          # Thing pages
│   │   ├── services/
│   │   └── locales/             # i18n translations
│   ├── package.json
│   ├── vite.config.js
│   ├── Dockerfile
│   └── nginx.conf
│
├── docker-compose.yml
└── README.md
```

### Docker Services

Docker Compose runs 3 services:

| Service | Port | Description |
|---------|------|-------------|
| **fuseki** | 3030 | Apache Jena Fuseki RDF Triple Store (SPARQL endpoint, 2GB JVM heap) |
| **backend** | 3015 | FastAPI backend (REST API, DTDL management, YAML processing, SQLite) |
| **frontend** | 3005 | React + Nginx (SPA web interface, reverse proxy) |

---

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

---

## License

This project is licensed under the MIT License.

---

## Acknowledgments

- [FastAPI](https://fastapi.tiangolo.com/) - Modern Python web framework
- [React](https://react.dev/) - UI library
- [DTDL](https://github.com/Azure/opendigitaltwins-dtdl) - Digital Twins Definition Language
