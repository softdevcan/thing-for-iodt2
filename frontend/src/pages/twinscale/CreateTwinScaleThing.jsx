import { useState, useEffect } from 'react'
import { useNavigate } from 'react-router-dom'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import { Textarea } from '@/components/ui/textarea'
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs'
import Editor from '@monaco-editor/react'
import { useToast } from '@/hooks/use-toast'
import { Plus, Loader2, Check, Trash2, MapPin, Building2, Layers, Info, FileCode } from 'lucide-react'
import { RadioGroup, RadioGroupItem } from '@/components/ui/radio-group'
import { Label } from '@/components/ui/label'
import { Badge } from '@/components/ui/badge'
import TwinScaleService from '@/services/twinscaleService'
import MapComponent from '@/components/map/MapComponent'
import { useTranslation } from 'react-i18next'
import useTenantStore from '@/store/useTenantStore'
import { Alert, AlertDescription } from '@/components/ui/alert'
import DTDLSelectionModal from '@/components/dtdl/DTDLSelectionModal'

const CreateTwinScaleThing = () => {
  const { t } = useTranslation()
  const navigate = useNavigate()
  const { toast } = useToast()
  const [isLoading, setIsLoading] = useState(false)

  // Tenant store
  const { currentTenant, fetchTenants } = useTenantStore()

  // Form state
  const [formData, setFormData] = useState({
    id: '',
    name: '',
    description: '',
    properties: [],
    relationships: [],
    commands: [],
    include_service_spec: true,
    store_in_rdf: true,
    latitude: 39.9334, // Default Ankara
    longitude: 32.8597,
    // NEW: Thing Type
    thing_type: 'device', // 'device', 'sensor', 'component'
    // NEW: Domain Metadata
    manufacturer: '',
    model: '',
    serial_number: '',
    firmware_version: '',
    // NEW: DTDL Integration
    dtdl_interface: null, // Selected DTDL interface
    dtdl_interface_summary: null, // Interface summary for UI
  })

  // DTDL Modal state
  const [showDTDLModal, setShowDTDLModal] = useState(false)

  // YAML preview state
  const [interfaceYaml, setInterfaceYaml] = useState('')
  const [instanceYaml, setInstanceYaml] = useState('')

  // Initialize tenant on mount
  useEffect(() => {
    if (!currentTenant) {
      fetchTenants(true, true) // Use public endpoint for unauthenticated access
    }
  }, [currentTenant, fetchTenants])

  // Auto-prefix Thing ID with tenant when user types
  const handleIdChange = (value) => {
    let finalId = value

    // If tenant exists and input doesn't already contain tenant prefix
    if (currentTenant && value && !value.includes(':')) {
      // Auto-format: tenant_id:value
      finalId = value
    }

    setFormData({ ...formData, id: finalId })
  }

  // Generate YAML preview when form changes
  useEffect(() => {
    if (formData.id && formData.name) {
      // Get the normalized ID (without tenant prefix for display)
      const cleanId = formData.id.includes(':')
        ? formData.id.split(':')[1]
        : formData.id

      const normalizedId = cleanId.toLowerCase().replace(/[^a-z0-9-]/g, '-')

      // Build labels section
      const labelsSection = []
      if (currentTenant) {
        labelsSection.push(`    tenant: ${currentTenant.tenant_id}`)
      }
      labelsSection.push(`    thing-type: ${formData.thing_type}`)

      // Build annotations section
      const annotationsSection = []
      if (formData.manufacturer) {
        annotationsSection.push(`    manufacturer: "${formData.manufacturer}"`)
      }
      if (formData.model) {
        annotationsSection.push(`    model: "${formData.model}"`)
      }
      if (formData.serial_number) {
        annotationsSection.push(`    serialNumber: "${formData.serial_number}"`)
      }
      if (formData.firmware_version) {
        annotationsSection.push(`    firmwareVersion: "${formData.firmware_version}"`)
      }

      // Simple YAML preview generation
      const interfacePreview = `apiVersion: dtd.twinscale/v0
kind: TwinInterface
metadata:
  name: ems-iodt2-${normalizedId}
  labels:
${labelsSection.join('\n')}${annotationsSection.length > 0 ? `\n  annotations:\n${annotationsSection.join('\n')}` : ''}
spec:
  name: ems-iodt2-${normalizedId}
  properties:
${formData.properties.map(p => `    - name: ${p.name}
      type: ${p.type}
      description: ${p.description || ''}
      x-writable: ${p.writable || false}`).join('\n') || '    []'}
  relationships:
${formData.relationships.map(r => `    - name: ${r.name}
      interface: ${r.target_interface}`).join('\n') || '    []'}
  commands:
${formData.commands.map(c => `    - name: ${c.name}
      description: ${c.description || ''}`).join('\n') || '    []'}`

      setInterfaceYaml(interfacePreview)

      // Instance YAML with labels
      const instanceLabelsSection = []
      if (currentTenant) {
        instanceLabelsSection.push(`    tenant: ${currentTenant.tenant_id}`)
      }
      instanceLabelsSection.push(`    thing-type: ${formData.thing_type}`)

      setInstanceYaml(`apiVersion: dtd.twinscale/v0
kind: TwinInstance
metadata:
  name: ems-iodt2-${normalizedId}
  labels:
${instanceLabelsSection.join('\n')}
spec:
  name: ems-iodt2-${normalizedId}
  interface: ems-iodt2-${normalizedId}
  twinInstanceRelationships: []`)
    }
  }, [formData, currentTenant])

  const handleSubmit = async () => {
    // Tenant validation
    if (!currentTenant) {
      toast({
        variant: 'destructive',
        title: t('common.error'),
        description: 'Please select a tenant before creating a Thing',
      })
      return
    }

    if (!formData.id || !formData.name) {
      toast({
        variant: 'destructive',
        title: 'Validation Error',
        description: 'ID and Name are required',
      })
      return
    }

    setIsLoading(true)
    try {
      const result = await TwinScaleService.createTwinScaleThing(formData)
      toast({
        title: t('common.success'),
        description: t('createThing.createSuccess', { name: formData.name }),
      })
      navigate('/things')
    } catch (error) {
      toast({
        variant: 'destructive',
        title: t('common.error'),
        description: error.message,
      })
    } finally {
      setIsLoading(false)
    }
  }

  const addProperty = () => {
    setFormData({
      ...formData,
      properties: [
        ...formData.properties,
        { name: '', type: 'string', description: '', writable: false, unit: '', minimum: null, maximum: null },
      ],
    })
  }

  const removeProperty = (index) => {
    setFormData({
      ...formData,
      properties: formData.properties.filter((_, i) => i !== index),
    })
  }

  const updateProperty = (index, field, value) => {
    const newProperties = [...formData.properties]
    newProperties[index] = { ...newProperties[index], [field]: value }
    setFormData({ ...formData, properties: newProperties })
  }

  // DTDL Handlers
  const handleDTDLSelect = (selectedInterface, interfaceSummary) => {
    setFormData({
      ...formData,
      dtdl_interface: selectedInterface,
      dtdl_interface_summary: interfaceSummary,
    })
    toast({
      title: t('common.success'),
      description: t('dtdl.interfaceSelected', { name: selectedInterface.displayName }),
    })
  }

  const handleRemoveDTDL = () => {
    setFormData({
      ...formData,
      dtdl_interface: null,
      dtdl_interface_summary: null,
    })
  }

  const handleAutoFillFromDTDL = () => {
    if (!formData.dtdl_interface_summary) return

    const summary = formData.dtdl_interface_summary

    // Auto-fill properties from DTDL
    const dtdlProperties = summary.propertyNames?.map((name) => ({
      name,
      type: 'string', // Default type, would need full interface to get actual type
      description: '',
      writable: true,
      unit: '',
      minimum: null,
      maximum: null,
    })) || []

    setFormData({
      ...formData,
      properties: [...formData.properties, ...dtdlProperties],
    })

    toast({
      title: t('common.success'),
      description: t('dtdl.fieldsAutoFilled'),
    })
  }

  const addRelationship = () => {
    setFormData({
      ...formData,
      relationships: [
        ...formData.relationships,
        { name: '', target_interface: '', description: '' },
      ],
    })
  }

  const removeRelationship = (index) => {
    setFormData({
      ...formData,
      relationships: formData.relationships.filter((_, i) => i !== index),
    })
  }

  const updateRelationship = (index, field, value) => {
    const newRelationships = [...formData.relationships]
    newRelationships[index] = { ...newRelationships[index], [field]: value }
    setFormData({ ...formData, relationships: newRelationships })
  }

  const addCommand = () => {
    setFormData({
      ...formData,
      commands: [
        ...formData.commands,
        { name: '', description: '' },
      ],
    })
  }

  const removeCommand = (index) => {
    setFormData({
      ...formData,
      commands: formData.commands.filter((_, i) => i !== index),
    })
  }

  const updateCommand = (index, field, value) => {
    const newCommands = [...formData.commands]
    newCommands[index] = { ...newCommands[index], [field]: value }
    setFormData({ ...formData, commands: newCommands })
  }

  const handleMapClick = (event) => {
    const { lngLat } = event
    setFormData({
      ...formData,
      latitude: lngLat.lat,
      longitude: lngLat.lng
    })
  }

  return (
    <div className="space-y-6">
      <div className="flex justify-between items-center">
        <h1 className="text-2xl font-bold">{t('createThing.title')}</h1>
        <Button onClick={handleSubmit} disabled={isLoading || !currentTenant}>
          {isLoading ? (
            <>
              <Loader2 className="mr-2 h-4 w-4 animate-spin" />
              {t('createThing.creating')}
            </>
          ) : (
            <>
              <Check className="mr-2 h-4 w-4" />
              {t('createThing.createButton')}
            </>
          )}
        </Button>
      </div>

      {/* Tenant Information Banner */}
      {currentTenant ? (
        <Alert className="bg-blue-50 dark:bg-blue-950/50 border-blue-200 dark:border-blue-800">
          <Building2 className="h-4 w-4 text-blue-600 dark:text-blue-400" />
          <AlertDescription className="flex items-center gap-2">
            <span className="font-medium text-blue-800 dark:text-blue-200">
              Creating Thing for Tenant:
            </span>
            <span className="font-semibold text-blue-900 dark:text-blue-100">
              {currentTenant.name || currentTenant.tenant_id}
            </span>
            <span className="text-xs text-blue-600 dark:text-blue-400 bg-blue-100 dark:bg-blue-900/30 px-2 py-1 rounded">
              ID: {currentTenant.tenant_id}
            </span>
          </AlertDescription>
        </Alert>
      ) : (
        <Alert variant="destructive">
          <Building2 className="h-4 w-4" />
          <AlertDescription>
            No tenant selected. Please select a tenant from the navbar to create a Thing.
          </AlertDescription>
        </Alert>
      )}

      {/* Thing Type Selector */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Layers className="h-5 w-5" />
            {t('createThing.thingType')}
          </CardTitle>
          <p className="text-sm text-muted-foreground">
            {t('createThing.thingTypeDescription')}
          </p>
        </CardHeader>
        <CardContent>
          <RadioGroup
            value={formData.thing_type}
            onValueChange={(value) => setFormData({ ...formData, thing_type: value })}
            className="space-y-3"
          >
            {/* Device Option */}
            <div className="flex items-start space-x-3 border rounded-lg p-4 hover:bg-accent transition-colors cursor-pointer">
              <RadioGroupItem value="device" id="device" className="mt-1" />
              <Label htmlFor="device" className="flex-1 cursor-pointer">
                <div className="font-semibold text-base mb-1">
                  📦 {t('createThing.typeDevice')}
                </div>
                <div className="text-sm text-muted-foreground space-y-1">
                  <p>{t('createThing.typeDeviceDesc')}</p>
                  <p className="text-xs italic">
                    <strong>Example:</strong> Weather station with temperature, humidity, pressure sensors
                  </p>
                  <div className="flex gap-2 mt-2">
                    <Badge variant="secondary" className="text-xs">DTDL: Multiple Telemetry</Badge>
                    <Badge variant="secondary" className="text-xs">Ditto: Multiple Features</Badge>
                  </div>
                </div>
              </Label>
            </div>

            {/* Sensor Option */}
            <div className="flex items-start space-x-3 border rounded-lg p-4 hover:bg-accent transition-colors cursor-pointer">
              <RadioGroupItem value="sensor" id="sensor" className="mt-1" />
              <Label htmlFor="sensor" className="flex-1 cursor-pointer">
                <div className="font-semibold text-base mb-1">
                  🌡️ {t('createThing.typeSensor')}
                </div>
                <div className="text-sm text-muted-foreground space-y-1">
                  <p>{t('createThing.typeSensorDesc')}</p>
                  <p className="text-xs italic">
                    <strong>Example:</strong> Single DHT22 temperature sensor
                  </p>
                  <div className="flex gap-2 mt-2">
                    <Badge variant="secondary" className="text-xs">DTDL: Simple Interface</Badge>
                    <Badge variant="secondary" className="text-xs">Ditto: Single Feature</Badge>
                  </div>
                </div>
              </Label>
            </div>

            {/* Component Option */}
            <div className="flex items-start space-x-3 border rounded-lg p-4 hover:bg-accent transition-colors cursor-pointer">
              <RadioGroupItem value="component" id="component" className="mt-1" />
              <Label htmlFor="component" className="flex-1 cursor-pointer">
                <div className="font-semibold text-base mb-1">
                  🏗️ {t('createThing.typeComponent')}
                </div>
                <div className="text-sm text-muted-foreground space-y-1">
                  <p>{t('createThing.typeComponentDesc')}</p>
                  <p className="text-xs italic">
                    <strong>Example:</strong> Building floor with multiple child sensors
                  </p>
                  <div className="flex gap-2 mt-2">
                    <Badge variant="secondary" className="text-xs">DTDL: Components (Ideal!)</Badge>
                    <Badge variant="secondary" className="text-xs">Uses Relationships</Badge>
                  </div>
                </div>
              </Label>
            </div>
          </RadioGroup>

          {/* Info Alert */}
          <Alert className="mt-4">
            <Info className="h-4 w-4" />
            <AlertDescription className="text-xs">
              <strong>Tip:</strong> Choose "Device" for physical units with multiple sensors.
              Choose "Component" for logical groupings using relationships.
            </AlertDescription>
          </Alert>
        </CardContent>
      </Card>

      {/* DTDL Interface Selection */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <FileCode className="h-5 w-5" />
            {t('dtdl.useDTDL')}
            <Badge variant="outline" className="ml-auto">Optional</Badge>
          </CardTitle>
          <p className="text-sm text-muted-foreground">
            {t('dtdl.useDTDLDescription')}
          </p>
        </CardHeader>
        <CardContent className="space-y-4">
          {!formData.dtdl_interface ? (
            <div className="border-2 border-dashed rounded-lg p-6 text-center">
              <FileCode className="h-12 w-12 mx-auto text-muted-foreground mb-3" />
              <p className="text-sm text-muted-foreground mb-4">
                {t('dtdl.noInterfaceSelected')}
              </p>
              <Button
                onClick={() => setShowDTDLModal(true)}
                variant="outline"
                className="gap-2"
              >
                <FileCode className="h-4 w-4" />
                {t('dtdl.selectInterface')}
              </Button>
            </div>
          ) : (
            <div className="space-y-4">
              <div className="border rounded-lg p-4 bg-accent/50">
                <div className="flex items-start justify-between mb-3">
                  <div className="flex-1">
                    <h4 className="font-semibold text-base">{formData.dtdl_interface.displayName}</h4>
                    <p className="text-xs text-muted-foreground font-mono mt-1">
                      {formData.dtdl_interface.dtmi}
                    </p>
                  </div>
                  <Button
                    onClick={handleRemoveDTDL}
                    variant="ghost"
                    size="sm"
                    className="h-8 w-8 p-0"
                  >
                    <Trash2 className="h-4 w-4" />
                  </Button>
                </div>
                <p className="text-sm text-muted-foreground mb-3">
                  {formData.dtdl_interface.description}
                </p>

                {formData.dtdl_interface_summary && (
                  <div className="grid grid-cols-4 gap-2 mb-3">
                    <div className="bg-background rounded p-2 text-center">
                      <div className="text-lg font-bold text-blue-600">
                        {formData.dtdl_interface_summary.telemetryCount}
                      </div>
                      <div className="text-xs text-muted-foreground">{t('dtdl.telemetry')}</div>
                    </div>
                    <div className="bg-background rounded p-2 text-center">
                      <div className="text-lg font-bold text-green-600">
                        {formData.dtdl_interface_summary.propertyCount}
                      </div>
                      <div className="text-xs text-muted-foreground">{t('dtdl.properties')}</div>
                    </div>
                    <div className="bg-background rounded p-2 text-center">
                      <div className="text-lg font-bold text-purple-600">
                        {formData.dtdl_interface_summary.commandCount}
                      </div>
                      <div className="text-xs text-muted-foreground">{t('dtdl.commands')}</div>
                    </div>
                    <div className="bg-background rounded p-2 text-center">
                      <div className="text-lg font-bold text-orange-600">
                        {formData.dtdl_interface_summary.componentCount}
                      </div>
                      <div className="text-xs text-muted-foreground">{t('dtdl.components')}</div>
                    </div>
                  </div>
                )}

                <div className="flex gap-2">
                  <Button
                    onClick={() => setShowDTDLModal(true)}
                    variant="outline"
                    size="sm"
                    className="flex-1"
                  >
                    {t('dtdl.changeInterface')}
                  </Button>
                  <Button
                    onClick={handleAutoFillFromDTDL}
                    variant="secondary"
                    size="sm"
                    className="flex-1"
                    disabled={!formData.dtdl_interface_summary?.propertyNames?.length}
                  >
                    {t('dtdl.autoFill')}
                  </Button>
                </div>
              </div>

              <Alert>
                <Info className="h-4 w-4" />
                <AlertDescription className="text-xs">
                  {t('dtdl.autoFillDescription')}
                </AlertDescription>
              </Alert>
            </div>
          )}
        </CardContent>
      </Card>

      <div className="grid grid-cols-2 gap-6">
        {/* Left: Form */}
        <Card>
          <CardHeader>
            <CardTitle>{t('createThing.thingDefinition')}</CardTitle>
          </CardHeader>
          <CardContent className="space-y-4">
            <div className="space-y-2">
              <label className="text-sm font-medium">{t('createThing.idRequired')}</label>
              <Input
                placeholder={currentTenant ? `${currentTenant.tenant_id}:thing1 or just thing1` : t('createThing.idPlaceholder')}
                value={formData.id}
                onChange={(e) => handleIdChange(e.target.value)}
              />
              {currentTenant && formData.id && !formData.id.includes(':') && (
                <p className="text-xs text-muted-foreground">
                  Will be saved as: <span className="font-mono font-semibold">{currentTenant.tenant_id}:{formData.id}</span>
                </p>
              )}
            </div>

            <div className="space-y-2">
              <label className="text-sm font-medium">{t('createThing.nameRequired')}</label>
              <Input
                placeholder={t('createThing.namePlaceholder')}
                value={formData.name}
                onChange={(e) => setFormData({ ...formData, name: e.target.value })}
              />
            </div>

            <div className="space-y-2">
              <label className="text-sm font-medium">{t('createThing.description')}</label>
              <Textarea
                placeholder={t('createThing.descriptionPlaceholder')}
                value={formData.description}
                onChange={(e) => setFormData({ ...formData, description: e.target.value })}
              />
            </div>

            {/* Domain Metadata Section */}
            <div className="border-t pt-4 mt-4">
              <h3 className="text-sm font-semibold mb-3 flex items-center gap-2">
                <Building2 className="h-4 w-4" />
                {t('createThing.domainMetadata')}
              </h3>
              <div className="grid grid-cols-2 gap-4">
                <div className="space-y-2">
                  <label className="text-sm font-medium">{t('createThing.manufacturer')}</label>
                  <Input
                    placeholder={t('createThing.manufacturerPlaceholder')}
                    value={formData.manufacturer}
                    onChange={(e) => setFormData({ ...formData, manufacturer: e.target.value })}
                  />
                </div>

                <div className="space-y-2">
                  <label className="text-sm font-medium">{t('createThing.model')}</label>
                  <Input
                    placeholder={t('createThing.modelPlaceholder')}
                    value={formData.model}
                    onChange={(e) => setFormData({ ...formData, model: e.target.value })}
                  />
                </div>

                <div className="space-y-2">
                  <label className="text-sm font-medium">{t('createThing.serialNumber')}</label>
                  <Input
                    placeholder={t('createThing.serialNumberPlaceholder')}
                    value={formData.serial_number}
                    onChange={(e) => setFormData({ ...formData, serial_number: e.target.value })}
                  />
                </div>

                <div className="space-y-2">
                  <label className="text-sm font-medium">
                    {t('createThing.firmwareVersion')}
                    <span className="text-muted-foreground ml-1">({t('common.optional')})</span>
                  </label>
                  <Input
                    placeholder={t('createThing.firmwareVersionPlaceholder')}
                    value={formData.firmware_version}
                    onChange={(e) => setFormData({ ...formData, firmware_version: e.target.value })}
                  />
                </div>
              </div>
            </div>

            <Tabs defaultValue="properties">
              <TabsList className="w-full grid grid-cols-4">
                <TabsTrigger value="properties">
                  {t('createThing.properties')} ({formData.properties.length})
                </TabsTrigger>
                <TabsTrigger value="relationships">
                  Relationships ({formData.relationships.length})
                </TabsTrigger>
                <TabsTrigger value="commands">
                  {t('createThing.commands')} ({formData.commands.length})
                </TabsTrigger>
                <TabsTrigger value="location">
                  {t('createThing.location')}
                </TabsTrigger>
              </TabsList>

              <TabsContent value="properties" className="space-y-4">
                {formData.properties.map((prop, index) => (
                  <div key={index} className="border rounded-md p-3 space-y-2">
                    <div className="flex justify-between items-center">
                      <span className="text-sm font-medium">{t('createThing.property')} {index + 1}</span>
                      <Button
                        variant="ghost"
                        size="sm"
                        onClick={() => removeProperty(index)}
                      >
                        <Trash2 className="h-4 w-4 text-destructive" />
                      </Button>
                    </div>
                    <Input
                      placeholder={t('createThing.propertyName')}
                      value={prop.name}
                      onChange={(e) => updateProperty(index, 'name', e.target.value)}
                    />
                    <select
                      className="w-full px-3 py-2 border rounded-md"
                      value={prop.type}
                      onChange={(e) => updateProperty(index, 'type', e.target.value)}
                    >
                      <option value="string">{t('createThing.typeString')}</option>
                      <option value="integer">{t('createThing.typeInteger')}</option>
                      <option value="float">{t('createThing.typeFloat')}</option>
                      <option value="boolean">{t('createThing.typeBoolean')}</option>
                      <option value="object">{t('createThing.typeObject')}</option>
                      <option value="array">{t('createThing.typeArray')}</option>
                    </select>
                    <Input
                      placeholder={t('createThing.propertyDescription')}
                      value={prop.description}
                      onChange={(e) => updateProperty(index, 'description', e.target.value)}
                    />
                    <div className="flex items-center gap-2">
                      <label className="flex items-center gap-2 text-sm">
                        <input
                          type="checkbox"
                          checked={prop.writable || false}
                          onChange={(e) => updateProperty(index, 'writable', e.target.checked)}
                          className="rounded"
                        />
                        Writable
                      </label>
                    </div>
                    <div className="grid grid-cols-3 gap-2">
                      <Input
                        type="number"
                        placeholder="Min"
                        value={prop.minimum || ''}
                        onChange={(e) => updateProperty(index, 'minimum', e.target.value ? parseFloat(e.target.value) : null)}
                      />
                      <Input
                        type="number"
                        placeholder="Max"
                        value={prop.maximum || ''}
                        onChange={(e) => updateProperty(index, 'maximum', e.target.value ? parseFloat(e.target.value) : null)}
                      />
                      <Input
                        placeholder="Unit"
                        value={prop.unit || ''}
                        onChange={(e) => updateProperty(index, 'unit', e.target.value)}
                      />
                    </div>
                  </div>
                ))}
                <Button variant="outline" onClick={addProperty} className="w-full">
                  <Plus className="mr-2 h-4 w-4" />
                  {t('createThing.addProperty')}
                </Button>
              </TabsContent>

              <TabsContent value="relationships" className="space-y-4">
                {formData.relationships.map((rel, index) => (
                  <div key={index} className="border rounded-md p-3 space-y-2">
                    <div className="flex justify-between items-center">
                      <span className="text-sm font-medium">Relationship {index + 1}</span>
                      <Button
                        variant="ghost"
                        size="sm"
                        onClick={() => removeRelationship(index)}
                      >
                        <Trash2 className="h-4 w-4 text-destructive" />
                      </Button>
                    </div>
                    <Input
                      placeholder="Relationship Name"
                      value={rel.name}
                      onChange={(e) => updateRelationship(index, 'name', e.target.value)}
                    />
                    <Input
                      placeholder="Target Interface (e.g., ems-iodt2-location)"
                      value={rel.target_interface}
                      onChange={(e) => updateRelationship(index, 'target_interface', e.target.value)}
                    />
                    <Input
                      placeholder="Description (optional)"
                      value={rel.description || ''}
                      onChange={(e) => updateRelationship(index, 'description', e.target.value)}
                    />
                  </div>
                ))}
                <Button variant="outline" onClick={addRelationship} className="w-full">
                  <Plus className="mr-2 h-4 w-4" />
                  Add Relationship
                </Button>
              </TabsContent>

              <TabsContent value="commands" className="space-y-4">
                {formData.commands.map((cmd, index) => (
                  <div key={index} className="border rounded-md p-3 space-y-2">
                    <div className="flex justify-between items-center">
                      <span className="text-sm font-medium">{t('createThing.command')} {index + 1}</span>
                      <Button
                        variant="ghost"
                        size="sm"
                        onClick={() => removeCommand(index)}
                      >
                        <Trash2 className="h-4 w-4 text-destructive" />
                      </Button>
                    </div>
                    <Input
                      placeholder={t('createThing.commandName')}
                      value={cmd.name}
                      onChange={(e) => updateCommand(index, 'name', e.target.value)}
                    />
                    <Input
                      placeholder={t('createThing.commandDescription')}
                      value={cmd.description}
                      onChange={(e) => updateCommand(index, 'description', e.target.value)}
                    />
                  </div>
                ))}
                <Button variant="outline" onClick={addCommand} className="w-full">
                  <Plus className="mr-2 h-4 w-4" />
                  {t('createThing.addCommand')}
                </Button>
              </TabsContent>

              <TabsContent value="location" className="space-y-4">
                <div className="h-[400px] w-full border rounded-md overflow-hidden relative">
                  <MapComponent
                    center={{ lat: formData.latitude || 39.9334, lng: formData.longitude || 32.8597 }}
                    zoom={15}
                    onMapClick={handleMapClick}
                    sensors={[{
                      id: 'new-location',
                      latitude: formData.latitude || 39.9334,
                      longitude: formData.longitude || 32.8597,
                      type: 'side_kozalak',
                      status: 'online',
                      name: t('createThing.selectedLocation')
                    }]}
                  />
                </div>
                <div className="grid grid-cols-2 gap-4">
                  <div className="space-y-2">
                    <label className="text-sm font-medium">{t('createThing.latitude')}</label>
                    <Input
                      type="number"
                      step="any"
                      value={formData.latitude}
                      onChange={(e) => setFormData({ ...formData, latitude: parseFloat(e.target.value) })}
                    />
                  </div>
                  <div className="space-y-2">
                    <label className="text-sm font-medium">{t('createThing.longitude')}</label>
                    <Input
                      type="number"
                      step="any"
                      value={formData.longitude}
                      onChange={(e) => setFormData({ ...formData, longitude: parseFloat(e.target.value) })}
                    />
                  </div>
                </div>
              </TabsContent>
            </Tabs>
          </CardContent>
        </Card>

        {/* Right: YAML Preview */}
        <Card>
          <CardHeader>
            <CardTitle>{t('createThing.yamlPreview')}</CardTitle>
          </CardHeader>
          <CardContent>
            <Tabs defaultValue="interface">
              <TabsList className="w-full mb-4">
                <TabsTrigger value="interface" className="flex-1">
                  {t('createThing.interfaceYaml')}
                </TabsTrigger>
                <TabsTrigger value="instance" className="flex-1">
                  {t('createThing.instanceYaml')}
                </TabsTrigger>
              </TabsList>

              <TabsContent value="interface">
                <div className="border rounded-md overflow-hidden">
                  <Editor
                    height="500px"
                    defaultLanguage="yaml"
                    value={interfaceYaml}
                    options={{
                      minimap: { enabled: false },
                      readOnly: true,
                      wordWrap: 'on',
                      lineNumbers: 'on',
                    }}
                  />
                </div>
              </TabsContent>

              <TabsContent value="instance">
                <div className="border rounded-md overflow-hidden">
                  <Editor
                    height="500px"
                    defaultLanguage="yaml"
                    value={instanceYaml}
                    options={{
                      minimap: { enabled: false },
                      readOnly: true,
                      wordWrap: 'on',
                      lineNumbers: 'on',
                    }}
                  />
                </div>
              </TabsContent>
            </Tabs>
          </CardContent>
        </Card>
      </div>

      {/* DTDL Selection Modal */}
      <DTDLSelectionModal
        isOpen={showDTDLModal}
        onClose={() => setShowDTDLModal(false)}
        onSelect={handleDTDLSelect}
        thingType={formData.thing_type}
        domain={null}
      />
    </div>
  )
}

export default CreateTwinScaleThing
