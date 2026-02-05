/**
 * DTDL Selection Modal Component
 *
 * Modal for browsing, searching, and selecting DTDL interfaces.
 * Includes filtering, search, and interface detail view.
 */

import React, { useState, useEffect } from 'react'
import { useTranslation } from 'react-i18next'
import { X, Search, Filter, Info, CheckCircle } from 'lucide-react'
import { listInterfaces, getInterfaceSummary } from '../../api/dtdl'

const DTDLSelectionModal = ({ isOpen, onClose, onSelect, thingType = null, domain = null }) => {
  const { t } = useTranslation()
  const [interfaces, setInterfaces] = useState([])
  const [filteredInterfaces, setFilteredInterfaces] = useState([])
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)

  // Filters
  const [searchKeyword, setSearchKeyword] = useState('')
  const [selectedCategory, setSelectedCategory] = useState('all')
  const [selectedDomain, setSelectedDomain] = useState(domain || 'all')

  // Selected interface
  const [selectedInterface, setSelectedInterface] = useState(null)
  const [interfaceSummary, setInterfaceSummary] = useState(null)
  const [loadingSummary, setLoadingSummary] = useState(false)

  // Load interfaces on mount or when filters change
  useEffect(() => {
    if (isOpen) {
      loadInterfaces()
    }
  }, [isOpen, thingType])

  // Apply local filters
  useEffect(() => {
    applyFilters()
  }, [interfaces, searchKeyword, selectedCategory, selectedDomain])

  const loadInterfaces = async () => {
    setLoading(true)
    setError(null)
    try {
      const params = {}
      if (thingType && thingType !== 'all') {
        params.thing_type = thingType
      }
      const data = await listInterfaces(params)
      setInterfaces(data.interfaces || [])
    } catch (err) {
      console.error('Failed to load DTDL interfaces:', err)
      setError(t('dtdl.errorLoadingInterfaces'))
    } finally {
      setLoading(false)
    }
  }

  const applyFilters = () => {
    let filtered = [...interfaces]

    // Filter by keyword
    if (searchKeyword) {
      const keyword = searchKeyword.toLowerCase()
      filtered = filtered.filter(
        (iface) =>
          iface.displayName?.toLowerCase().includes(keyword) ||
          iface.description?.toLowerCase().includes(keyword) ||
          iface.dtmi?.toLowerCase().includes(keyword)
      )
    }

    // Filter by category
    if (selectedCategory !== 'all') {
      filtered = filtered.filter((iface) => iface.category === selectedCategory)
    }

    // Filter by domain
    if (selectedDomain !== 'all') {
      filtered = filtered.filter((iface) => {
        const tags = iface.tags || []
        return tags.includes(selectedDomain)
      })
    }

    setFilteredInterfaces(filtered)
  }

  const handleSelectInterface = async (iface) => {
    setSelectedInterface(iface)
    setLoadingSummary(true)
    try {
      const summary = await getInterfaceSummary(iface.dtmi)
      setInterfaceSummary(summary)
    } catch (err) {
      console.error('Failed to load interface summary:', err)
    } finally {
      setLoadingSummary(false)
    }
  }

  const handleConfirmSelection = () => {
    if (selectedInterface) {
      onSelect(selectedInterface, interfaceSummary)
      handleClose()
    }
  }

  const handleClose = () => {
    setSelectedInterface(null)
    setInterfaceSummary(null)
    setSearchKeyword('')
    setSelectedCategory('all')
    setSelectedDomain(domain || 'all')
    onClose()
  }

  // Extract unique categories and domains
  const categories = ['all', ...new Set(interfaces.map((i) => i.category).filter(Boolean))]
  const domains = [
    'all',
    ...new Set(interfaces.flatMap((i) => i.tags || []).filter((tag) => tag.includes('_') || tag === 'environmental' || tag === 'air-quality'))
  ]

  if (!isOpen) return null

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black bg-opacity-50">
      <div className="bg-white rounded-lg shadow-xl w-full max-w-5xl max-h-[90vh] flex flex-col">
        {/* Header */}
        <div className="flex items-center justify-between p-6 border-b">
          <div>
            <h2 className="text-2xl font-semibold text-gray-900">
              {t('dtdl.selectInterface')}
            </h2>
            <p className="text-sm text-gray-500 mt-1">
              {t('dtdl.selectInterfaceDescription')}
            </p>
          </div>
          <button
            onClick={handleClose}
            className="text-gray-400 hover:text-gray-600 transition-colors"
          >
            <X size={24} />
          </button>
        </div>

        {/* Filters */}
        <div className="p-6 border-b bg-gray-50">
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            {/* Search */}
            <div className="relative">
              <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 text-gray-400" size={20} />
              <input
                type="text"
                placeholder={t('dtdl.searchPlaceholder')}
                value={searchKeyword}
                onChange={(e) => setSearchKeyword(e.target.value)}
                className="w-full pl-10 pr-4 py-2 border border-gray-300 rounded-md focus:ring-2 focus:ring-blue-500 focus:border-transparent"
              />
            </div>

            {/* Category Filter */}
            <div>
              <select
                value={selectedCategory}
                onChange={(e) => setSelectedCategory(e.target.value)}
                className="w-full px-4 py-2 border border-gray-300 rounded-md focus:ring-2 focus:ring-blue-500 focus:border-transparent"
              >
                {categories.map((cat) => (
                  <option key={cat} value={cat}>
                    {cat === 'all' ? t('dtdl.allCategories') : cat}
                  </option>
                ))}
              </select>
            </div>

            {/* Domain Filter */}
            <div>
              <select
                value={selectedDomain}
                onChange={(e) => setSelectedDomain(e.target.value)}
                className="w-full px-4 py-2 border border-gray-300 rounded-md focus:ring-2 focus:ring-blue-500 focus:border-transparent"
              >
                {domains.map((dom) => (
                  <option key={dom} value={dom}>
                    {dom === 'all' ? t('dtdl.allDomains') : dom}
                  </option>
                ))}
              </select>
            </div>
          </div>

          {/* Results count */}
          <div className="mt-3 text-sm text-gray-600">
            {t('dtdl.showingResults', { count: filteredInterfaces.length, total: interfaces.length })}
          </div>
        </div>

        {/* Content */}
        <div className="flex-1 overflow-hidden flex">
          {/* Interface List */}
          <div className="w-1/2 border-r overflow-y-auto">
            {loading ? (
              <div className="flex items-center justify-center h-full">
                <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-blue-600"></div>
              </div>
            ) : error ? (
              <div className="p-6 text-center text-red-600">
                {error}
              </div>
            ) : filteredInterfaces.length === 0 ? (
              <div className="p-6 text-center text-gray-500">
                {t('dtdl.noInterfacesFound')}
              </div>
            ) : (
              <div className="divide-y">
                {filteredInterfaces.map((iface) => (
                  <div
                    key={iface.dtmi}
                    onClick={() => handleSelectInterface(iface)}
                    className={`p-4 cursor-pointer hover:bg-gray-50 transition-colors ${
                      selectedInterface?.dtmi === iface.dtmi ? 'bg-blue-50 border-l-4 border-blue-600' : ''
                    }`}
                  >
                    <div className="flex items-start justify-between">
                      <div className="flex-1">
                        <h3 className="font-medium text-gray-900">{iface.displayName}</h3>
                        <p className="text-xs text-gray-500 mt-1">{iface.dtmi}</p>
                        <p className="text-sm text-gray-600 mt-2 line-clamp-2">{iface.description}</p>
                        <div className="flex flex-wrap gap-2 mt-2">
                          <span className="inline-flex items-center px-2 py-1 rounded text-xs font-medium bg-gray-100 text-gray-800">
                            {iface.category}
                          </span>
                          {iface.tags?.slice(0, 3).map((tag) => (
                            <span key={tag} className="inline-flex items-center px-2 py-1 rounded text-xs font-medium bg-blue-100 text-blue-800">
                              {tag}
                            </span>
                          ))}
                        </div>
                      </div>
                      {selectedInterface?.dtmi === iface.dtmi && (
                        <CheckCircle className="text-blue-600 ml-2 flex-shrink-0" size={20} />
                      )}
                    </div>
                  </div>
                ))}
              </div>
            )}
          </div>

          {/* Interface Details */}
          <div className="w-1/2 overflow-y-auto bg-gray-50">
            {!selectedInterface ? (
              <div className="flex flex-col items-center justify-center h-full text-gray-400">
                <Info size={48} />
                <p className="mt-4 text-sm">{t('dtdl.selectInterfaceToView')}</p>
              </div>
            ) : loadingSummary ? (
              <div className="flex items-center justify-center h-full">
                <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-blue-600"></div>
              </div>
            ) : (
              <div className="p-6">
                <h3 className="text-xl font-semibold text-gray-900 mb-2">
                  {selectedInterface.displayName}
                </h3>
                <p className="text-xs text-gray-500 mb-4 font-mono">{selectedInterface.dtmi}</p>
                <p className="text-sm text-gray-700 mb-6">{selectedInterface.description}</p>

                {interfaceSummary && (
                  <>
                    {/* Extends */}
                    {interfaceSummary.extends && (
                      <div className="mb-6">
                        <h4 className="text-sm font-semibold text-gray-900 mb-2">{t('dtdl.extends')}</h4>
                        <p className="text-xs text-gray-600 font-mono bg-white p-2 rounded border">
                          {interfaceSummary.extends}
                        </p>
                      </div>
                    )}

                    {/* Contents Summary */}
                    <div className="mb-6">
                      <h4 className="text-sm font-semibold text-gray-900 mb-2">{t('dtdl.contents')}</h4>
                      <div className="grid grid-cols-2 gap-3">
                        <div className="bg-white p-3 rounded border">
                          <div className="text-2xl font-bold text-blue-600">{interfaceSummary.telemetryCount}</div>
                          <div className="text-xs text-gray-600">{t('dtdl.telemetry')}</div>
                        </div>
                        <div className="bg-white p-3 rounded border">
                          <div className="text-2xl font-bold text-green-600">{interfaceSummary.propertyCount}</div>
                          <div className="text-xs text-gray-600">{t('dtdl.properties')}</div>
                        </div>
                        <div className="bg-white p-3 rounded border">
                          <div className="text-2xl font-bold text-purple-600">{interfaceSummary.commandCount}</div>
                          <div className="text-xs text-gray-600">{t('dtdl.commands')}</div>
                        </div>
                        <div className="bg-white p-3 rounded border">
                          <div className="text-2xl font-bold text-orange-600">{interfaceSummary.componentCount}</div>
                          <div className="text-xs text-gray-600">{t('dtdl.components')}</div>
                        </div>
                      </div>
                    </div>

                    {/* Telemetry List */}
                    {interfaceSummary.telemetryNames?.length > 0 && (
                      <div className="mb-6">
                        <h4 className="text-sm font-semibold text-gray-900 mb-2">{t('dtdl.telemetryFields')}</h4>
                        <ul className="space-y-1">
                          {interfaceSummary.telemetryNames.map((name) => (
                            <li key={name} className="text-sm text-gray-700 bg-white px-3 py-2 rounded border">
                              <code className="text-blue-600">{name}</code>
                            </li>
                          ))}
                        </ul>
                      </div>
                    )}

                    {/* Property List */}
                    {interfaceSummary.propertyNames?.length > 0 && (
                      <div className="mb-6">
                        <h4 className="text-sm font-semibold text-gray-900 mb-2">{t('dtdl.propertyFields')}</h4>
                        <ul className="space-y-1">
                          {interfaceSummary.propertyNames.map((name) => (
                            <li key={name} className="text-sm text-gray-700 bg-white px-3 py-2 rounded border">
                              <code className="text-green-600">{name}</code>
                            </li>
                          ))}
                        </ul>
                      </div>
                    )}

                    {/* Component List */}
                    {interfaceSummary.componentNames?.length > 0 && (
                      <div className="mb-6">
                        <h4 className="text-sm font-semibold text-gray-900 mb-2">{t('dtdl.componentFields')}</h4>
                        <ul className="space-y-1">
                          {interfaceSummary.componentNames.map((name) => (
                            <li key={name} className="text-sm text-gray-700 bg-white px-3 py-2 rounded border">
                              <code className="text-orange-600">{name}</code>
                            </li>
                          ))}
                        </ul>
                      </div>
                    )}
                  </>
                )}
              </div>
            )}
          </div>
        </div>

        {/* Footer */}
        <div className="flex items-center justify-between p-6 border-t bg-gray-50">
          <div className="text-sm text-gray-600">
            {selectedInterface ? (
              <span>
                {t('dtdl.selected')}: <strong>{selectedInterface.displayName}</strong>
              </span>
            ) : (
              <span>{t('dtdl.noInterfaceSelected')}</span>
            )}
          </div>
          <div className="flex gap-3">
            <button
              onClick={handleClose}
              className="px-4 py-2 border border-gray-300 rounded-md text-gray-700 hover:bg-gray-100 transition-colors"
            >
              {t('common.cancel')}
            </button>
            <button
              onClick={handleConfirmSelection}
              disabled={!selectedInterface}
              className="px-4 py-2 bg-blue-600 text-white rounded-md hover:bg-blue-700 transition-colors disabled:bg-gray-300 disabled:cursor-not-allowed"
            >
              {t('common.select')}
            </button>
          </div>
        </div>
      </div>
    </div>
  )
}

export default DTDLSelectionModal
