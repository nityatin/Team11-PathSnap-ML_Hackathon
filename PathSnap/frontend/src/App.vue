<script setup>
import { ref, onMounted, onBeforeUnmount, nextTick, computed } from 'vue'
import axios from 'axios'
import L from 'leaflet'
import 'leaflet/dist/leaflet.css'

const API = import.meta.env.VITE_API_URL || 'http://localhost:8000'

const trips = ref([])
const selectedTrip = ref('')
const result = ref(null)
const loading = ref(false)
const error = ref('')
const useHMM = ref(true)
const apiOnline = ref(false)

let map = null
let mapLayers = []

const gpsCount = computed(() => result.value?.gps_points?.length || 0)
const segmentCount = computed(() => result.value?.predicted_segments?.length || 0)

async function loadTrips() {
  try {
    const response = await axios.get(`${API}/api/v1/trips`)
    trips.value = response.data.trips || []
    selectedTrip.value = trips.value[0] || ''
    apiOnline.value = true
    error.value = ''
  } catch (e) {
    apiOnline.value = false
    error.value = 'Cannot connect to the ML server. Check that FastAPI is running.'
  }
}

async function predict() {
  if (!selectedTrip.value || loading.value) return

  loading.value = true
  error.value = ''
  result.value = null

  try {
    const response = await axios.post(
      `${API}/api/v1/map-match/${selectedTrip.value}`,
      null,
      { params: { use_hmm: useHMM.value } }
    )

    result.value = response.data

    await nextTick()
    drawMap()
  } catch (e) {
    error.value = e.response?.data?.detail || 'Map matching failed. Please try again.'
  } finally {
    loading.value = false
  }
}

function drawMap() {
  const mapElement = document.getElementById('map')
  if (!mapElement || !result.value) return

  if (map) {
    map.remove()
    map = null
  }

  mapLayers = []

  map = L.map(mapElement, {
    zoomControl: true,
    scrollWheelZoom: true
  })

  L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
    maxZoom: 19,
    attribution: '&copy; OpenStreetMap contributors'
  }).addTo(map)

  const gps = (result.value.gps_points || []).map(p => [
    p.latitude,
    p.longitude
  ])

  if (gps.length) {
    const gpsLayer = L.polyline(gps, {
      color: '#2563eb',
      weight: 3,
      opacity: 0.85
    }).addTo(map)

    mapLayers.push(gpsLayer)

    L.circleMarker(gps[0], {
      radius: 6,
      color: '#ffffff',
      weight: 2,
      fillColor: '#16a34a',
      fillOpacity: 1
    }).addTo(map)

    L.circleMarker(gps[gps.length - 1], {
      radius: 6,
      color: '#ffffff',
      weight: 2,
      fillColor: '#dc2626',
      fillOpacity: 1
    }).addTo(map)
  }

  const roadLines = result.value.route_geometry?.coordinates || []

  roadLines.forEach(line => {
    const points = line.map(coord => [coord[1], coord[0]])

    if (points.length >= 2) {
      const layer = L.polyline(points, {
        color: '#ef4444',
        weight: 5,
        opacity: 0.9
      }).addTo(map)

      mapLayers.push(layer)
    }
  })

  const bounds = L.featureGroup(mapLayers).getBounds()

  if (bounds.isValid()) {
    map.fitBounds(bounds, { padding: [35, 35], maxZoom: 16 })
  } else {
    map.setView([0, 0], 2)
  }

  setTimeout(() => map?.invalidateSize(), 150)
}

function resetMap() {
  result.value = null
  error.value = ''
  if (map) {
    map.remove()
    map = null
  }
}

onMounted(loadTrips)

onBeforeUnmount(() => {
  if (map) map.remove()
})
</script>

<template>
  <div class="app-shell">
    <aside class="sidebar">
      <div class="brand">
        <div class="brand-icon">
          <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
            <path d="M3 17l6-6 4 4 8-9"/>
            <path d="M15 6h6v6"/>
          </svg>
        </div>
        <div>
          <h1>PathSnap</h1>
          <span>INTELLIGENT MAP MATCHING</span>
        </div>
      </div>

      <div class="sidebar-section">
        <span class="sidebar-label">WORKSPACE</span>
        <div class="nav-item active">
          <span class="nav-symbol">⌖</span>
          Map Matching
        </div>
        <div class="nav-item">
          <span class="nav-symbol">▤</span>
          Model Evaluation
          <span class="nav-soon">RESULTS</span>
        </div>
      </div>

      <div class="sidebar-bottom">
        <div class="server-card">
          <span class="server-dot" :class="{ online: apiOnline }"></span>
          <div>
            <strong>{{ apiOnline ? 'ML Server Online' : 'Server Disconnected' }}</strong>
            <small>{{ apiOnline ? 'API connection active' : 'Waiting for connection' }}</small>
          </div>
        </div>
        <div class="sidebar-footer">PathSnap · ML Hackathon 2026</div>
      </div>
    </aside>

    <main class="main-content">
      <header class="topbar">
        <div>
          <span class="breadcrumb">WORKSPACE / <b>MAP MATCHING</b></span>
        </div>
        <div class="topbar-right">
          <span class="status-indicator" :class="{ connected: apiOnline }">
            <i></i>{{ apiOnline ? 'SYSTEM ONLINE' : 'CONNECTING' }}
          </span>
          <span class="version">v1.0</span>
        </div>
      </header>

      <div class="page-content">
        <section class="page-heading">
          <div>
            <span class="eyebrow">GPS TRAJECTORY ANALYSIS</span>
            <h2>Map Matching Dashboard</h2>
            <p>Transform noisy GPS observations into meaningful road trajectories.</p>
          </div>
          <div class="heading-chip">
            <span class="chip-dot"></span>
            Bayesian RF + Viterbi
          </div>
        </section>

        <section class="metric-grid">
          <article class="metric-card">
            <div class="metric-icon blue-icon">⌖</div>
            <div>
              <span>AVAILABLE TRIPS</span>
              <strong>{{ trips.length }}</strong>
              <small>Student test dataset</small>
            </div>
          </article>
          <article class="metric-card">
            <div class="metric-icon purple-icon">◈</div>
            <div>
              <span>ML MODEL</span>
              <strong class="model-value">Bayesian RF</strong>
              <small>Optimized classifier</small>
            </div>
          </article>
          <article class="metric-card">
            <div class="metric-icon green-icon">↗</div>
            <div>
              <span>DECODER</span>
              <strong class="model-value">{{ useHMM ? 'HMM + Viterbi' : 'Classifier only' }}</strong>
              <small>Sequence reconstruction</small>
            </div>
          </article>
          <article class="metric-card">
            <div class="metric-icon orange-icon">◷</div>
            <div>
              <span>INFERENCE TIME</span>
              <strong>{{ result ? `${result.inference_time_seconds}s` : '—' }}</strong>
              <small>Latest prediction</small>
            </div>
          </article>
        </section>

        <section class="workspace-grid">
          <div class="map-panel">
            <div class="panel-heading">
              <div>
                <h3>Route Visualization</h3>
                <p>GPS observations and reconstructed road segments</p>
              </div>
              <span class="map-badge">LIVE MAP</span>
            </div>

            <div class="map-wrapper">
              <div id="map">
                <div v-if="!result && !loading" class="map-placeholder">
                  <div class="placeholder-icon">⌖</div>
                  <strong>Your route will appear here</strong>
                  <span>Select a trip and run map matching to visualize its trajectory.</span>
                </div>
                <div v-if="loading" class="map-loading">
                  <div class="spinner"></div>
                  <strong>Processing GPS trajectory...</strong>
                  <span>Generating candidates and reconstructing route</span>
                </div>
              </div>
            </div>

            <div class="map-legend">
              <span><i class="legend-line gps-line"></i> Original GPS trajectory</span>
              <span><i class="legend-line route-line"></i> Predicted road route</span>
              <span><i class="legend-dot start-dot"></i> Start</span>
              <span><i class="legend-dot end-dot"></i> End</span>
            </div>
          </div>

          <aside class="control-column">
            <div class="control-panel">
              <div class="panel-heading compact-heading">
                <div>
                  <h3>Run Prediction</h3>
                  <p>Configure your map-matching run</p>
                </div>
                <span class="settings-symbol">⚙</span>
              </div>

              <label for="trip-select">SELECT GPS TRIP</label>
              <select id="trip-select" v-model="selectedTrip" :disabled="loading">
                <option value="" disabled>Select a trip</option>
                <option v-for="trip in trips" :key="trip" :value="trip">
                  {{ trip }}
                </option>
              </select>

              <div class="decoder-setting">
                <div>
                  <strong>HMM + Viterbi</strong>
                  <small>Enable sequence decoding</small>
                </div>
                <label class="switch">
                  <input type="checkbox" v-model="useHMM" :disabled="loading">
                  <span class="slider"></span>
                </label>
              </div>

              <div class="method-info">
                <span class="info-symbol">i</span>
                <p>{{ useHMM
                  ? 'Viterbi selects a likely connected road sequence using candidate probabilities.'
                  : 'The classifier independently selects the highest-probability road candidate.'
                }}</p>
              </div>

              <button class="run-button" @click="predict" :disabled="loading || !selectedTrip">
                <span v-if="loading" class="button-spinner"></span>
                <span v-else>▶</span>
                {{ loading ? 'Processing...' : 'Run Map Matching' }}
              </button>

              <button class="reset-button" @click="resetMap" :disabled="loading">
                Reset visualization
              </button>

              <p v-if="error" class="error-message">{{ error }}</p>
            </div>

            <div class="results-panel">
              <div class="panel-heading compact-heading">
                <div>
                  <h3>Prediction Summary</h3>
                  <p>Latest execution details</p>
                </div>
                <span class="result-check" v-if="result">✓</span>
              </div>

              <div v-if="result" class="result-content">
                <div class="result-row">
                  <span>Selected trip</span>
                  <strong>{{ result.trip_id }}</strong>
                </div>
                <div class="result-row">
                  <span>GPS observations</span>
                  <strong>{{ gpsCount.toLocaleString() }}</strong>
                </div>
                <div class="result-row">
                  <span>Predicted segments</span>
                  <strong>{{ segmentCount.toLocaleString() }}</strong>
                </div>
                <div class="result-row">
                  <span>ML classifier</span>
                  <strong>{{ result.model }}</strong>
                </div>
                <div class="result-row">
                  <span>Decoder</span>
                  <strong>{{ result.decoder }}</strong>
                </div>
                <div class="result-row">
                  <span>Inference time</span>
                  <strong>{{ result.inference_time_seconds }} sec</strong>
                </div>
                <div class="result-note">
                  Accuracy is not calculated during prediction because reference labels are not used.
                </div>
              </div>

              <div v-else class="empty-results">
                <span>▤</span>
                <p>No prediction yet.</p>
                <small>Run map matching to view results.</small>
              </div>
            </div>
          </aside>
        </section>

        <footer class="page-footer">
          <span>PathSnap · Machine Learning Based GPS Map Matching</span>
          <span>Developed for ML Hackathon 2026</span>
        </footer>
      </div>
    </main>
  </div>
</template>
