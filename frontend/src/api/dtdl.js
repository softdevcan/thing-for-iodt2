/**
 * DTDL API Client
 *
 * API client for DTDL interface browsing, searching, validation, and conversion.
 */

import api from './index'

/**
 * List all DTDL interfaces with optional filtering
 * @param {Object} params - Filter parameters
 * @param {string} params.thing_type - Filter by thing type (device, sensor, component)
 * @param {string} params.domain - Filter by domain (environmental, air_quality, etc.)
 * @param {string} params.category - Filter by category (base, environmental, etc.)
 * @param {string} params.tags - Filter by tags (comma-separated)
 * @param {string} params.keywords - Search in displayName and description
 * @returns {Promise} Response with total count and interfaces list
 */
export const listInterfaces = async (params = {}) => {
  const response = await api.get('/v2/dtdl/interfaces', { params })
  return response.data
}

/**
 * Get detailed information for a specific DTDL interface
 * @param {string} dtmi - DTDL interface identifier (e.g., "dtmi:iodt2:TemperatureSensor;1")
 * @returns {Promise} Interface details with full DTDL JSON and summary
 */
export const getInterfaceDetails = async (dtmi) => {
  const response = await api.get(`/v2/dtdl/interfaces/${encodeURIComponent(dtmi)}`)
  return response.data
}

/**
 * Get interface summary for UI display
 * @param {string} dtmi - DTDL interface identifier
 * @returns {Promise} Summary with counts and field names
 */
export const getInterfaceSummary = async (dtmi) => {
  const response = await api.get(`/v2/dtdl/interfaces/${encodeURIComponent(dtmi)}/summary`)
  return response.data
}

/**
 * Get interface requirements (required/optional telemetry and properties)
 * @param {string} dtmi - DTDL interface identifier
 * @returns {Promise} Requirements with schema information
 */
export const getInterfaceRequirements = async (dtmi) => {
  const response = await api.get(`/v2/dtdl/interfaces/${encodeURIComponent(dtmi)}/requirements`)
  return response.data
}

/**
 * Suggest DTDL interfaces based on Thing data
 * @param {Object} data - Suggestion request
 * @param {string} data.thing_type - Thing type (device, sensor, component)
 * @param {string} data.domain - Optional domain filter
 * @param {Array<string>} data.properties - Optional property names
 * @param {Array<string>} data.telemetry - Optional telemetry names
 * @param {string} data.keywords - Optional keywords
 * @returns {Promise} Suggested interfaces with match scores
 */
export const suggestInterfaces = async (data) => {
  const response = await api.post('/v2/dtdl/suggest', data)
  return response.data
}

/**
 * Validate a TwinScale Thing against a DTDL interface
 * @param {Object} data - Validation request
 * @param {Object} data.thing_data - Thing properties and telemetry
 * @param {string} data.dtmi - DTDL interface identifier
 * @param {boolean} data.strict - Whether to treat extra fields as errors
 * @returns {Promise} Validation result with compatibility score and issues
 */
export const validateThing = async (data) => {
  const response = await api.post('/v2/dtdl/validate', data)
  return response.data
}

/**
 * Find best matching DTDL interfaces for a Thing
 * @param {Object} data - Find best match request
 * @param {Object} data.thing_data - Thing properties and telemetry
 * @param {string} data.thing_type - Optional thing type filter
 * @param {string} data.domain - Optional domain filter
 * @param {number} data.top_n - Number of top results (default: 5)
 * @returns {Promise} List of matches sorted by combined score
 */
export const findBestMatch = async (data) => {
  const response = await api.post('/v2/dtdl/find-best-match', data)
  return response.data
}

/**
 * Convert DTDL interface to TwinScale YAML template
 * @param {Object} data - Conversion request
 * @param {string} data.dtmi - DTDL interface identifier
 * @param {string} data.thing_name - Optional Thing name
 * @param {string} data.tenant_id - Optional tenant ID
 * @returns {Promise} TwinInterface and TwinInstance YAML templates
 */
export const convertToTwinScale = async (data) => {
  const response = await api.post('/v2/dtdl/convert/to-twinscale', data)
  return response.data
}

/**
 * Enrich TwinScale Thing with DTDL metadata
 * @param {Object} data - Enrich request
 * @param {Object} data.thing_data - TwinScale Thing data
 * @param {string} data.dtmi - DTDL interface identifier
 * @returns {Promise} Enriched Thing data with DTDL annotations
 */
export const enrichWithDTDL = async (data) => {
  const response = await api.post('/v2/dtdl/enrich', data)
  return response.data
}

/**
 * List all available domains with their associated DTMIs
 * @returns {Promise} Dictionary mapping domain names to DTMI lists
 */
export const listDomains = async () => {
  const response = await api.get('/v2/dtdl/domains')
  return response.data
}

/**
 * List all thing types with their associated DTMIs
 * @returns {Promise} Dictionary mapping thing types to DTMI lists
 */
export const listThingTypes = async () => {
  const response = await api.get('/v2/dtdl/thing-types')
  return response.data
}

/**
 * Reload DTDL library (for development/testing)
 * @returns {Promise} Success status with interface count
 */
export const reloadLibrary = async () => {
  const response = await api.post('/v2/dtdl/reload')
  return response.data
}

export default {
  listInterfaces,
  getInterfaceDetails,
  getInterfaceSummary,
  getInterfaceRequirements,
  suggestInterfaces,
  validateThing,
  findBestMatch,
  convertToTwinScale,
  enrichWithDTDL,
  listDomains,
  listThingTypes,
  reloadLibrary
}
