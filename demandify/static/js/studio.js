/**
 * Demand Studio - Interactive Traffic Scenario Workbench
 */

// Global Studio State
const Studio = {
    currentSource: null, // { type: 'dataset'|'run'|'scenario', id: '...' }
    networkData: null,   // GeoJSON FeatureCollection
    odPairs: [],         // Array of OD items
    edgeFlows: {},       // { [edgeId]: flow_vehs_h }
    edgeCongestion: {},  // { [edgeId]: { sim_speed, freeflow_speed, ratio } }
    selectedOrigin: null,
    selectedDestination: null,
    selectedOD: null,
    activeViewMode: 'flow', // 'flow' | 'congestion'
    multiplier: 1.0,
    preserveMinOne: true,
    lastTestStats: null,
    map: null,
    edgesLayer: null,
    selectedRouteLayer: null,
    destinationsGroup: null,
    originMarker: null,
    destinationMarker: null,
    spotlightMask: null,
    networkBoundaryRect: null,
    spotlightEnabled: true,
};

// Initialize Studio when DOM loads
document.addEventListener('DOMContentLoaded', () => {
    initMap();
    setupEventListeners();
    loadSourcesCatalog();
});

/* ==========================================================================
   Map Initialization & Layer Management
   ========================================================================== */
function initMap() {
    Studio.map = L.map('studio-map', {
        zoomControl: true,
    }).setView([50.0647, 19.9450], 13); // Default view

    // Standard OpenStreetMap tiles (100% free, no API key required)
    L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
        attribution: '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors',
        maxZoom: 19
    }).addTo(Studio.map);

    Studio.destinationsGroup = L.layerGroup().addTo(Studio.map);

    // Map-level click snap: makes road link picking effortless by snapping to the closest road
    // within 28 screen pixels even if the user does not click precisely on the thin line
    Studio.map.on('click', (e) => {
        const closest = findClosestEdge(e.latlng, 28);
        if (closest) {
            handleEdgeClick(closest.id, closest.nearestLatLng || e.latlng);
        }
    });
}

function getFlowColor(flow) {
    if (!flow || flow <= 0) return '#475569'; // Neutral slate for inactive network edges
    if (flow < 50) return '#fde047';         // Light golden yellow
    if (flow < 150) return '#fbbf24';        // Warm golden amber
    if (flow < 300) return '#f97316';        // Vibrant orange
    if (flow < 600) return '#ea580c';        // Demandify core orange
    return '#c2410c';                        // Deep dark orange / rust
}

/**
 * Finds the closest road edge to a clicked point in screen-pixel coordinates.
 * Returns { id, nearestLatLng, distance } or null if none within maxPixelDist.
 */
function findClosestEdge(latlng, maxPixelDist = 28) {
    if (!Studio.networkData || !Studio.networkData.features || !Studio.networkData.features.length) {
        return null;
    }
    const clickPt = Studio.map.latLngToContainerPoint(latlng);
    let best = null;
    let minDistance = maxPixelDist;

    for (let f = 0; f < Studio.networkData.features.length; f++) {
        const feature = Studio.networkData.features[f];
        if (!feature.geometry || feature.geometry.type !== 'LineString') continue;
        const coords = feature.geometry.coordinates;
        if (!coords || coords.length < 2) continue;

        for (let i = 0; i < coords.length - 1; i++) {
            // Leaflet coordinates are [lon, lat]
            const p1 = Studio.map.latLngToContainerPoint([coords[i][1], coords[i][0]]);
            const p2 = Studio.map.latLngToContainerPoint([coords[i + 1][1], coords[i + 1][0]]);

            // Fast bounding box check in pixels
            const minX = Math.min(p1.x, p2.x) - minDistance;
            const maxX = Math.max(p1.x, p2.x) + minDistance;
            const minY = Math.min(p1.y, p2.y) - minDistance;
            const maxY = Math.max(p1.y, p2.y) + minDistance;
            if (clickPt.x < minX || clickPt.x > maxX || clickPt.y < minY || clickPt.y > maxY) {
                continue;
            }

            const dx = p2.x - p1.x;
            const dy = p2.y - p1.y;
            const lenSq = dx * dx + dy * dy;

            let t = 0;
            if (lenSq > 0) {
                t = ((clickPt.x - p1.x) * dx + (clickPt.y - p1.y) * dy) / lenSq;
                t = Math.max(0, Math.min(1, t));
            }

            const projX = p1.x + t * dx;
            const projY = p1.y + t * dy;
            const dist = Math.hypot(clickPt.x - projX, clickPt.y - projY);

            if (dist < minDistance) {
                minDistance = dist;
                const nearestLatLng = Studio.map.containerPointToLatLng(L.point(projX, projY));
                best = {
                    id: feature.properties.id,
                    nearestLatLng: nearestLatLng,
                    distance: dist
                };
            }
        }
    }
    return best;
}

function getCongestionColor(ratio) {
    if (ratio === null || ratio === undefined) return '#94a3b8';
    if (ratio >= 0.85) return '#10b981'; // Green (free-flow)
    if (ratio >= 0.60) return '#84cc16'; // Lime
    if (ratio >= 0.40) return '#f59e0b'; // Amber (moderate)
    return '#ef4444';                    // Red (severe congestion)
}

function getEdgeStyle(feature) {
    const edgeId = feature.properties.id;
    if (Studio.activeViewMode === 'congestion') {
        const cong = Studio.edgeCongestion[edgeId];
        if (cong) {
            const ratio = cong.ratio;
            return {
                color: getCongestionColor(ratio),
                weight: 5.5,
                opacity: 0.95,
            };
        } else {
            return {
                color: '#94a3b8',
                weight: 1.5,
                opacity: 0.30,
            };
        }
    }

    const flow = Studio.edgeFlows[edgeId] || 0;
    const weight = flow > 0 ? Math.min(8.5, Math.max(3.2, Math.log10(flow + 1) * 2.5)) : 2.8;
    return {
        color: getFlowColor(flow),
        weight: weight,
        opacity: flow > 0 ? 0.90 : 0.70,
    };
}

function renderBoundarySpotlight() {
    if (Studio.spotlightMask) {
        Studio.map.removeLayer(Studio.spotlightMask);
        Studio.spotlightMask = null;
    }
    if (Studio.networkBoundaryRect) {
        Studio.map.removeLayer(Studio.networkBoundaryRect);
        Studio.networkBoundaryRect = null;
    }

    if (!Studio.networkData || !Studio.networkData.bounds || !Studio.spotlightEnabled) return;

    const bounds = Studio.networkData.bounds;
    const south = bounds[0][0];
    const west = bounds[0][1];
    const north = bounds[1][0];
    const east = bounds[1][1];

    // Polygon mask with hole over the network area (softly dims the outside map)
    const worldRing = [
        [-90, -180],
        [-90, 180],
        [90, 180],
        [90, -180]
    ];
    const hole = [
        [south, west],
        [south, east],
        [north, east],
        [north, west]
    ];

    Studio.spotlightMask = L.polygon([worldRing, hole], {
        stroke: false,
        fillColor: '#0f172a',
        fillOpacity: 0.28,
        interactive: false,
    }).addTo(Studio.map);

    // Subtle dashed orange boundary rectangle around the simulation area
    Studio.networkBoundaryRect = L.rectangle(bounds, {
        color: '#ea580c',
        weight: 2,
        dashArray: '8, 6',
        fill: false,
        interactive: false,
    }).addTo(Studio.map);
}

function renderNetworkEdges() {
    if (!Studio.networkData) return;

    if (Studio.edgesLayer) {
        Studio.map.removeLayer(Studio.edgesLayer);
    }

    Studio.edgesLayer = L.geoJSON(Studio.networkData, {
        style: getEdgeStyle,
        onEachFeature: (feature, layer) => {
            const props = feature.properties;
            const flow = Studio.edgeFlows[props.id] || 0;
            const cong = Studio.edgeCongestion[props.id];

            let tooltipContent = `<strong>Edge: ${props.id}</strong><br>` +
                `Type: ${props.type || 'road'}<br>` +
                `Speed Limit: ${props.speed_limit_kmh} km/h<br>` +
                `Flow: <strong>${Math.round(flow)} veh/h</strong> (${(flow/60).toFixed(2)} veh/min)`;

            if (cong) {
                tooltipContent += `<br>Sim Speed: <strong>${cong.sim_speed} km/h</strong> (Ratio: ${Math.round(cong.ratio * 100)}%)`;
            }

            layer.bindTooltip(tooltipContent, { sticky: true, className: 'studio-tooltip' });

            // Dynamic hover feedback: highlights and expands road stroke on hover
            layer.on('mouseover', () => {
                const currentWeight = (layer.options && layer.options.weight) || 2.8;
                layer.setStyle({
                    weight: currentWeight + 3.5,
                    opacity: 1.0,
                });
                if (!L.Browser.ie && !L.Browser.opera && !L.Browser.edge) {
                    layer.bringToFront();
                }
            });

            layer.on('mouseout', () => {
                if (Studio.edgesLayer) {
                    Studio.edgesLayer.resetStyle(layer);
                }
            });

            layer.on('click', (e) => {
                L.DomEvent.stopPropagation(e);
                handleEdgeClick(props.id, e.latlng);
            });
        }
    }).addTo(Studio.map);

    renderBoundarySpotlight();

    if (Studio.networkData.bounds) {
        Studio.map.fitBounds(Studio.networkData.bounds, { padding: [35, 35] });
    }
}

/* ==========================================================================
   Interactive OD Selection & Route Rendering
   ========================================================================== */
function handleEdgeClick(edgeId, latlng) {
    // If no origin is selected, treat this click as Origin
    if (!Studio.selectedOrigin) {
        selectOrigin(edgeId, latlng);
    } else if (Studio.selectedOrigin === edgeId) {
        // Clicking same edge clears selection
        clearSelection();
    } else {
        // Second edge clicked: Destination!
        selectDestination(edgeId, latlng);
    }
}

async function selectOrigin(edgeId, latlng) {
    Studio.selectedOrigin = edgeId;
    Studio.selectedDestination = null;
    Studio.selectedOD = null;

    // Place Green Origin Marker
    if (Studio.originMarker) Studio.map.removeLayer(Studio.originMarker);
    Studio.originMarker = L.circleMarker(latlng, {
        radius: 8,
        color: '#ffffff',
        fillColor: '#10b981',
        fillOpacity: 1,
        weight: 3,
        className: 'pin-pulse-green'
    }).addTo(Studio.map).bindPopup(`<strong>Origin:</strong> ${edgeId}`).openPopup();

    updateFloatingHelper(`Origin set to <code>${edgeId}</code>. Click an existing destination to view/edit, or click any other road to create a new OD.`);

    // Fetch and show existing destinations branching from this origin
    Studio.destinationsGroup.clearLayers();
    if (Studio.selectedRouteLayer) Studio.map.removeLayer(Studio.selectedRouteLayer);

    try {
        const res = await fetch('/api/studio/find-destinations', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                source_type: Studio.currentSource.type,
                source_id: Studio.currentSource.id,
                origin: edgeId,
                od_pairs: Studio.odPairs,
            })
        });
        const data = await res.json();

        // Highlight existing destination branches with golden amber dotted lines
        data.existing_destinations.forEach(item => {
            if (item.coordinates && item.coordinates.length > 0) {
                const latlngs = item.coordinates.map(c => [c[1], c[0]]);
                const poly = L.polyline(latlngs, {
                    color: '#f59e0b',
                    weight: 3.5,
                    dashArray: '6, 6',
                    opacity: 0.85,
                }).addTo(Studio.destinationsGroup);

                const lastCoord = latlngs[latlngs.length - 1];
                const marker = L.circleMarker(lastCoord, {
                    radius: 5.5,
                    color: '#ffffff',
                    fillColor: '#f59e0b',
                    fillOpacity: 1,
                    weight: 2
                }).addTo(Studio.destinationsGroup);

                marker.bindTooltip(`Dest: ${item.destination}<br>Flow: ${item.vehs_per_hour} veh/h`, { sticky: true });
                marker.on('click', () => selectODByEndpoints(edgeId, item.destination));
            }
        });
    } catch (err) {
        console.error('Error fetching destinations:', err);
    }
}

async function selectDestination(destEdgeId, latlng) {
    Studio.selectedDestination = destEdgeId;

    // Place Red Destination Marker
    if (Studio.destinationMarker) Studio.map.removeLayer(Studio.destinationMarker);
    Studio.destinationMarker = L.circleMarker(latlng, {
        radius: 8,
        color: '#ffffff',
        fillColor: '#ef4444',
        fillOpacity: 1,
        weight: 3
    }).addTo(Studio.map).bindPopup(`<strong>Destination:</strong> ${destEdgeId}`).openPopup();

    // Check if OD pair already exists
    let existing = Studio.odPairs.find(od => od.origin === Studio.selectedOrigin && od.destination === destEdgeId);

    if (existing) {
        Studio.selectedOD = existing;
        highlightODRoute(existing.origin, existing.destination);
        showODDetailCard(existing, false);
        scrollSidebarToOD(existing.id);
        updateFloatingHelper(`Selected existing OD: <code>${existing.origin} &rarr; ${existing.destination}</code> (${existing.vehs_per_hour} veh/h)`);
    } else {
        // Create new OD Pair!
        const newOD = {
            id: `od_custom_${Date.now()}`,
            origin: Studio.selectedOrigin,
            destination: destEdgeId,
            vehs_per_hour: 20, // default 20 veh/h = 0.33 veh/min
            vehs_per_min: 0.33,
            base_vehs_per_hour: 20,
            trips_in_window: 5,
        };
        Studio.odPairs.unshift(newOD); // place at top
        Studio.selectedOD = newOD;
        recalculateFlows();
        renderODList();
        highlightODRoute(newOD.origin, newOD.destination);
        showODDetailCard(newOD, true);
        updateFloatingHelper(`Created new OD pair: <code>${newOD.origin} &rarr; ${newOD.destination}</code>! Adjust flow in sidebar.`);
    }
}

async function highlightODRoute(origin, destination) {
    if (Studio.selectedRouteLayer) {
        Studio.map.removeLayer(Studio.selectedRouteLayer);
    }

    try {
        const res = await fetch('/api/studio/route', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                source_type: Studio.currentSource.type,
                source_id: Studio.currentSource.id,
                origin: origin,
                destination: destination,
            })
        });
        const data = await res.json();
        if (data.routable && data.coordinates && data.coordinates.length > 0) {
            const latlngs = data.coordinates.map(c => [c[1], c[0]]);
            Studio.selectedRouteLayer = L.polyline(latlngs, {
                color: '#facc15',
                weight: 6,
                opacity: 0.95,
                lineCap: 'round',
            }).addTo(Studio.map);

            // Fit bounds smoothly
            Studio.map.flyToBounds(Studio.selectedRouteLayer.getBounds(), { padding: [60, 60], maxZoom: 16 });
        }
    } catch (err) {
        console.error('Error fetching route:', err);
    }
}

function clearSelection() {
    Studio.selectedOrigin = null;
    Studio.selectedDestination = null;
    Studio.selectedOD = null;
    if (Studio.originMarker) Studio.map.removeLayer(Studio.originMarker);
    if (Studio.destinationMarker) Studio.map.removeLayer(Studio.destinationMarker);
    if (Studio.selectedRouteLayer) Studio.map.removeLayer(Studio.selectedRouteLayer);
    Studio.destinationsGroup.clearLayers();
    document.getElementById('od-detail-panel').classList.add('d-none');
    updateFloatingHelper('Click any road to select an Origin link.');
}

function updateFloatingHelper(text) {
    const el = document.getElementById('map-helper-text');
    if (el) el.innerHTML = text;
}

/* ==========================================================================
   Data Loading & Calculations
   ========================================================================== */
async function loadSourcesCatalog() {
    try {
        const res = await fetch('/api/studio/sources');
        const data = await res.json();
        populateSourceDropdown(data);
    } catch (err) {
        console.error('Error loading sources catalog:', err);
    }
}

function populateSourceDropdown(sources) {
    const sel = document.getElementById('studio-source-select');
    sel.innerHTML = '<option value="">-- Choose Network or Scenario --</option>';

    // Datasets
    if (sources.datasets && sources.datasets.length > 0) {
        const grp = document.createElement('optgroup');
        grp.label = 'Offline Datasets';
        sources.datasets.forEach(ds => {
            const opt = document.createElement('option');
            opt.value = `dataset:${ds.id}`;
            opt.textContent = `${ds.name} (${ds.quality_label || 'ready'})`;
            grp.appendChild(opt);
        });
        sel.appendChild(grp);
    }

    // Runs
    if (sources.runs && sources.runs.length > 0) {
        const grp = document.createElement('optgroup');
        grp.label = 'Calibration Runs';
        sources.runs.forEach(r => {
            const opt = document.createElement('option');
            opt.value = `run:${r.id}`;
            opt.textContent = `Run: ${r.name}`;
            grp.appendChild(opt);
        });
        sel.appendChild(grp);
    }

    // Custom Scenarios
    if (sources.scenarios && sources.scenarios.length > 0) {
        const grp = document.createElement('optgroup');
        grp.label = 'Saved Scenarios';
        sources.scenarios.forEach(s => {
            const opt = document.createElement('option');
            opt.value = `scenario:${s.id}`;
            opt.textContent = `Scenario: ${s.name}`;
            grp.appendChild(opt);
        });
        sel.appendChild(grp);
    }

    // If source parameter in URL, select it automatically
    const urlParams = new URLSearchParams(window.location.search);
    const sourceParam = urlParams.get('source');
    if (sourceParam) {
        sel.value = sourceParam;
        loadSelectedSource();
    } else if (sources.runs && sources.runs.length > 0) {
        sel.value = `run:${sources.runs[0].id}`;
        loadSelectedSource();
    } else if (sources.datasets && sources.datasets.length > 0) {
        sel.value = `dataset:${sources.datasets[0].id}`;
        loadSelectedSource();
    }
}

async function loadSelectedSource() {
    const sel = document.getElementById('studio-source-select');
    const val = sel.value;
    if (!val) return;

    const [type, ...rest] = val.split(':');
    const id = rest.join(':');
    Studio.currentSource = { type, id };

    showLoading(true, 'Loading road network and demand...');

    try {
        const res = await fetch(`/api/studio/load?source_type=${encodeURIComponent(type)}&source_id=${encodeURIComponent(id)}`);
        if (!res.ok) {
            const err = await res.json();
            throw new Error(err.detail || 'Failed to load source');
        }
        const data = await res.json();

        Studio.networkData = data.network;
        Studio.odPairs = data.od_pairs || [];
        Studio.multiplier = 1.0;
        document.getElementById('multiplier-slider').value = 1.0;
        document.getElementById('multiplier-val-display').textContent = '1.00×';

        clearSelection();
        recalculateFlows();
        renderNetworkEdges();
        renderODList();
        updateTopMetrics();

        showLoading(false);
    } catch (err) {
        showLoading(false);
        alert('Error loading source: ' + err.message);
    }
}

function recalculateFlows() {
    // Client-side quick recalculation of flows from active ODs
    // (For full precision, the server compute_edge_flows is used, but we keep an edge map)
    Studio.edgeFlows = {};
    if (Studio.networkData && Studio.networkData.features) {
        Studio.networkData.features.forEach(f => {
            Studio.edgeFlows[f.properties.id] = 0;
        });
    }

    let totalVehsH = 0;
    Studio.odPairs.forEach(od => {
        const vol = parseFloat(od.vehs_per_hour) || 0;
        if (vol > 0) {
            totalVehsH += vol;
        }
    });

    updateTopMetrics();
}

function updateTopMetrics() {
    let totalVehsH = 0;
    let activeCount = 0;

    Studio.odPairs.forEach(od => {
        const vol = parseFloat(od.vehs_per_hour) || 0;
        if (vol > 0) {
            totalVehsH += vol;
            activeCount++;
        }
    });

    const vehsMin = (totalVehsH / 60.0).toFixed(2);
    document.getElementById('stat-total-demand').textContent = `${Math.round(totalVehsH).toLocaleString()} veh/h`;
    document.getElementById('stat-total-demand-min').textContent = `(${vehsMin} veh/min)`;
    document.getElementById('stat-active-ods').textContent = `${activeCount} / ${Studio.odPairs.length}`;
    document.getElementById('stat-network-edges').textContent = Studio.networkData ? Studio.networkData.edge_count : 0;
}

/* ==========================================================================
   Demand Multipliers
   ========================================================================== */
function setMultiplier(val) {
    val = Math.max(0.05, Math.min(3.0, parseFloat(val)));
    Studio.multiplier = val;
    document.getElementById('multiplier-slider').value = val;
    document.getElementById('multiplier-val-display').textContent = `${val.toFixed(2)}×`;

    // Apply scaling
    let origTotal = 0;
    let scaledTotal = 0;

    Studio.odPairs.forEach(od => {
        const base = parseFloat(od.base_vehs_per_hour !== undefined ? od.base_vehs_per_hour : od.vehs_per_hour) || 0;
        origTotal += base;
        let scaled = Math.round(base * val);
        if (Studio.preserveMinOne && scaled === 0 && base > 0 && val > 0) {
            scaled = 1;
        }
        od.vehs_per_hour = scaled;
        od.vehs_per_min = parseFloat((scaled / 60.0).toFixed(2));
        scaledTotal += scaled;
    });

    const deltaPct = origTotal > 0 ? (((scaledTotal - origTotal) / origTotal) * 100).toFixed(1) : 0;
    const deltaSign = deltaPct >= 0 ? '+' : '';
    document.getElementById('multiplier-delta-preview').textContent = `Orig: ${Math.round(origTotal)} &rarr; Scaled: ${Math.round(scaledTotal)} (${deltaSign}${deltaPct}%)`;

    recalculateFlows();
    renderODList();
    renderNetworkEdges();
}

function resetMultiplier() {
    Studio.multiplier = 1.0;
    document.getElementById('multiplier-slider').value = 1.0;
    document.getElementById('multiplier-val-display').textContent = '1.00×';
    document.getElementById('multiplier-delta-preview').textContent = '';

    Studio.odPairs.forEach(od => {
        if (od.base_vehs_per_hour !== undefined) {
            od.vehs_per_hour = od.base_vehs_per_hour;
            od.vehs_per_min = parseFloat((od.base_vehs_per_hour / 60.0).toFixed(2));
        }
    });

    recalculateFlows();
    renderODList();
    renderNetworkEdges();
}

/* ==========================================================================
   Sidebar OD List Rendering & Editing
   ========================================================================== */
function renderODList(filterQuery = '') {
    const container = document.getElementById('od-list-container');
    container.innerHTML = '';

    const q = filterQuery.toLowerCase().trim();
    const filtered = Studio.odPairs.filter(od => {
        if (!q) return true;
        return od.id.toLowerCase().includes(q) ||
            od.origin.toLowerCase().includes(q) ||
            od.destination.toLowerCase().includes(q);
    });

    if (filtered.length === 0) {
        container.innerHTML = '<div class="text-center text-muted p-4"><i class="bi bi-inbox fs-3 d-block mb-2"></i>No OD pairs found.</div>';
        return;
    }

    filtered.forEach((od, idx) => {
        const card = document.createElement('div');
        card.className = `od-card ${Studio.selectedOD && Studio.selectedOD.id === od.id ? 'selected' : ''}`;
        card.id = `card-${od.id}`;

        card.innerHTML = `
            <div class="d-flex justify-content-between align-items-center mb-1">
                <div class="fw-bold small text-truncate" style="max-width: 220px;" title="${od.origin} &rarr; ${od.destination}">
                    <span class="text-muted me-1">#${idx+1}</span>
                    <span class="text-success"><i class="bi bi-geo-alt-fill"></i> ${od.origin}</span>
                    <span class="text-muted">&rarr;</span>
                    <span class="text-danger"><i class="bi bi-pin-map-fill"></i> ${od.destination}</span>
                </div>
                <div class="btn-group btn-group-sm">
                    <button class="btn btn-outline-secondary btn-sm p-1" title="Focus route on map" onclick="focusOD('${od.id}', event)">
                        <i class="bi bi-crosshair"></i>
                    </button>
                    <button class="btn btn-outline-danger btn-sm p-1" title="Delete OD pair" onclick="deleteOD('${od.id}', event)">
                        <i class="bi bi-trash"></i>
                    </button>
                </div>
            </div>
            <div class="d-flex justify-content-between align-items-center mt-2">
                <div class="d-flex align-items-center gap-1">
                    <button class="btn btn-sm btn-outline-secondary px-2 py-0" onclick="adjustODFlow('${od.id}', -5, event)">-5</button>
                    <button class="btn btn-sm btn-outline-secondary px-2 py-0" onclick="adjustODFlow('${od.id}', -1, event)">-1</button>
                    <input type="number" min="0" max="5000" class="form-control form-control-sm od-flow-input"
                        value="${od.vehs_per_hour}" onchange="setODFlowDirect('${od.id}', this.value)" onclick="event.stopPropagation();">
                    <button class="btn btn-sm btn-outline-secondary px-2 py-0" onclick="adjustODFlow('${od.id}', 1, event)">+1</button>
                    <button class="btn btn-sm btn-outline-secondary px-2 py-0" onclick="adjustODFlow('${od.id}', 5, event)">+5</button>
                </div>
                <div class="text-end">
                    <span class="badge bg-light text-primary border">${(od.vehs_per_hour / 60).toFixed(2)} veh/min</span>
                </div>
            </div>
        `;

        card.addEventListener('click', () => {
            selectOD(od);
        });

        container.appendChild(card);
    });
}

function selectOD(od) {
    Studio.selectedOD = od;
    Studio.selectedOrigin = od.origin;
    Studio.selectedDestination = od.destination;

    // Highlight card
    document.querySelectorAll('.od-card').forEach(c => c.classList.remove('selected'));
    const c = document.getElementById(`card-${od.id}`);
    if (c) c.classList.add('selected');

    highlightODRoute(od.origin, od.destination);
    showODDetailCard(od, false);
}

function selectODByEndpoints(origin, destination) {
    const found = Studio.odPairs.find(od => od.origin === origin && od.destination === destination);
    if (found) {
        selectOD(found);
        scrollSidebarToOD(found.id);
    }
}

function showODDetailCard(od, isNew = false) {
    const panel = document.getElementById('od-detail-panel');
    panel.classList.remove('d-none');
    document.getElementById('detail-od-title').innerHTML = isNew ?
        '<span class="badge bg-success me-1">NEW</span> Custom OD Pair' :
        `OD Pair: ${od.id}`;
    document.getElementById('detail-orig-id').textContent = od.origin;
    document.getElementById('detail-dest-id').textContent = od.destination;
    document.getElementById('detail-flow-h').value = od.vehs_per_hour;
    document.getElementById('detail-flow-min').textContent = (od.vehs_per_hour / 60).toFixed(2);
}

function focusOD(odId, e) {
    if (e) e.stopPropagation();
    const od = Studio.odPairs.find(x => x.id === odId);
    if (od) selectOD(od);
}

function adjustODFlow(odId, delta, e) {
    if (e) e.stopPropagation();
    const od = Studio.odPairs.find(x => x.id === odId);
    if (!od) return;
    const newVol = Math.max(0, parseInt(od.vehs_per_hour) + delta);
    od.vehs_per_hour = newVol;
    od.vehs_per_min = parseFloat((newVol / 60.0).toFixed(2));
    recalculateFlows();
    renderODList();
    renderNetworkEdges();
    if (Studio.selectedOD && Studio.selectedOD.id === odId) {
        showODDetailCard(od);
    }
}

function setODFlowDirect(odId, value) {
    const od = Studio.odPairs.find(x => x.id === odId);
    if (!od) return;
    const newVol = Math.max(0, parseInt(value) || 0);
    od.vehs_per_hour = newVol;
    od.vehs_per_min = parseFloat((newVol / 60.0).toFixed(2));
    recalculateFlows();
    renderODList();
    renderNetworkEdges();
    if (Studio.selectedOD && Studio.selectedOD.id === odId) {
        showODDetailCard(od);
    }
}

function deleteOD(odId, e) {
    if (e) e.stopPropagation();
    if (!confirm('Remove this OD pair from demand scenario?')) return;
    Studio.odPairs = Studio.odPairs.filter(x => x.id !== odId);
    if (Studio.selectedOD && Studio.selectedOD.id === odId) {
        clearSelection();
    }
    recalculateFlows();
    renderODList();
    renderNetworkEdges();
}

function scrollSidebarToOD(odId) {
    const el = document.getElementById(`card-${odId}`);
    if (el) {
        el.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
    }
}

/* ==========================================================================
   Background Test Simulation in SUMO
   ========================================================================== */
async function testSimulation() {
    if (!Studio.currentSource) return;
    const btn = document.getElementById('btn-test-sim');
    const origHtml = btn.innerHTML;
    btn.disabled = true;
    btn.innerHTML = '<span class="spinner-border spinner-border-sm me-1"></span> Simulating...';

    const windowMinutes = parseInt(document.getElementById('sim-window-select').value) || 15;

    try {
        const res = await fetch('/api/studio/test-simulation', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                source_type: Studio.currentSource.type,
                source_id: Studio.currentSource.id,
                od_pairs: Studio.odPairs,
                window_minutes: windowMinutes,
                mesosim: true,
            })
        });

        if (!res.ok) {
            const err = await res.json();
            throw new Error(err.detail || 'Simulation failed');
        }

        const data = await res.json();
        Studio.edgeCongestion = data.edge_congestion || {};
        Studio.lastTestStats = data.stats;

        // Switch view mode to Congestion Heatmap
        setViewMode('congestion');
        showSimulationResults(data.stats);

    } catch (err) {
        alert('Simulation test failed: ' + err.message);
    } finally {
        btn.disabled = false;
        btn.innerHTML = origHtml;
    }
}

function showSimulationResults(stats) {
    const card = document.getElementById('sim-results-card');
    card.classList.remove('d-none');
    document.getElementById('sim-res-inserted').textContent = stats.total_vehicles_inserted.toLocaleString();
    document.getElementById('sim-res-completion').textContent = `${stats.completion_rate_pct}%`;
    document.getElementById('sim-res-teleports').textContent = stats.teleports;
    document.getElementById('sim-res-speed').textContent = `${stats.mean_edge_speed_kmh} km/h`;

    const teleportsEl = document.getElementById('sim-res-teleports');
    if (stats.teleports > 0) {
        teleportsEl.className = 'text-warning fw-bold';
    } else {
        teleportsEl.className = 'text-success fw-bold';
    }
}

function setViewMode(mode) {
    Studio.activeViewMode = mode;
    document.querySelectorAll('.btn-view-mode').forEach(btn => {
        btn.classList.toggle('active', btn.dataset.mode === mode);
    });

    const legendFlow = document.getElementById('legend-flow');
    const legendCong = document.getElementById('legend-congestion');
    if (mode === 'congestion') {
        legendFlow.classList.add('d-none');
        legendCong.classList.remove('d-none');
    } else {
        legendFlow.classList.remove('d-none');
        legendCong.classList.add('d-none');
    }

    renderNetworkEdges();
}

/* ==========================================================================
   Save Scenario & OSM Fetching
   ========================================================================== */
async function saveScenario() {
    if (!Studio.currentSource) return;
    const name = prompt('Enter a name for this custom scenario:', `scenario_${Date.now()}`);
    if (!name || !name.trim()) return;

    showLoading(true, 'Saving scenario files...');

    const windowMinutes = parseInt(document.getElementById('sim-window-select').value) || 15;

    try {
        const res = await fetch('/api/studio/save-scenario', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                source_type: Studio.currentSource.type,
                source_id: Studio.currentSource.id,
                scenario_id: name.trim(),
                name: name.trim(),
                od_pairs: Studio.odPairs,
                window_minutes: windowMinutes,
                last_test_stats: Studio.lastTestStats,
            })
        });

        if (!res.ok) {
            const err = await res.json();
            throw new Error(err.detail || 'Save failed');
        }

        const data = await res.json();
        showLoading(false);
        alert(`Scenario saved successfully to: ${data.path}`);
        loadSourcesCatalog();
    } catch (err) {
        showLoading(false);
        alert('Error saving scenario: ' + err.message);
    }
}

async function fetchOSMNetwork() {
    const name = document.getElementById('osm-scenario-name').value.trim();
    const west = parseFloat(document.getElementById('osm-bbox-west').value);
    const south = parseFloat(document.getElementById('osm-bbox-south').value);
    const east = parseFloat(document.getElementById('osm-bbox-east').value);
    const north = parseFloat(document.getElementById('osm-bbox-north').value);

    if (!name) {
        alert('Please enter a scenario name.');
        return;
    }
    if (isNaN(west) || isNaN(south) || isNaN(east) || isNaN(north)) {
        alert('Please enter valid bounding box coordinates.');
        return;
    }

    const modal = bootstrap.Modal.getInstance(document.getElementById('osmModal'));
    if (modal) modal.hide();

    showLoading(true, 'Fetching OSM network via Overpass and converting to SUMO...');

    try {
        const res = await fetch('/api/studio/fetch-osm', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                bbox: [west, south, east, north],
                name: name,
            })
        });

        if (!res.ok) {
            const err = await res.json();
            throw new Error(err.detail || 'OSM fetch failed');
        }

        const data = await res.json();
        await loadSourcesCatalog();

        // Switch to the newly created scenario
        const sel = document.getElementById('studio-source-select');
        sel.value = `scenario:${data.scenario_id}`;
        loadSelectedSource();

    } catch (err) {
        showLoading(false);
        alert('Failed to fetch OSM network: ' + err.message);
    }
}

/* ==========================================================================
   UI Event Listeners & Helpers
   ========================================================================== */
function setupEventListeners() {
    // Source dropdown
    document.getElementById('studio-source-select').addEventListener('change', loadSelectedSource);

    // Multiplier slider
    const slider = document.getElementById('multiplier-slider');
    slider.addEventListener('input', (e) => setMultiplier(e.target.value));

    // Multiplier presets
    document.querySelectorAll('.preset-btn').forEach(btn => {
        btn.addEventListener('click', () => setMultiplier(btn.dataset.factor));
    });

    // Reset multiplier
    document.getElementById('btn-reset-multiplier').addEventListener('click', resetMultiplier);

    // Preserve min 1 checkbox
    document.getElementById('check-preserve-min').addEventListener('change', (e) => {
        Studio.preserveMinOne = e.target.checked;
        setMultiplier(Studio.multiplier);
    });

    // OD Search
    document.getElementById('od-search-input').addEventListener('input', (e) => {
        renderODList(e.target.value);
    });

    // View mode toggles
    document.querySelectorAll('.btn-view-mode').forEach(btn => {
        btn.addEventListener('click', () => setViewMode(btn.dataset.mode));
    });

    // Spotlight Area toggle
    const btnSpotlight = document.getElementById('btn-toggle-spotlight');
    if (btnSpotlight) {
        btnSpotlight.addEventListener('click', () => {
            Studio.spotlightEnabled = !Studio.spotlightEnabled;
            btnSpotlight.classList.toggle('active', Studio.spotlightEnabled);
            renderBoundarySpotlight();
        });
    }

    // Test Simulation button
    document.getElementById('btn-test-sim').addEventListener('click', testSimulation);

    // Save Scenario button
    document.getElementById('btn-save-scenario').addEventListener('click', saveScenario);

    // New OD action
    document.getElementById('btn-new-od').addEventListener('click', () => {
        clearSelection();
        updateFloatingHelper('<strong>Pick Mode:</strong> Click any road to select an Origin link.');
    });

    // Clear selection button in detail panel
    document.getElementById('btn-close-od-detail').addEventListener('click', clearSelection);

    // Flow change inside detail panel
    document.getElementById('detail-flow-h').addEventListener('change', (e) => {
        if (Studio.selectedOD) {
            setODFlowDirect(Studio.selectedOD.id, e.target.value);
        }
    });

    // Fetch OSM button
    document.getElementById('btn-submit-osm').addEventListener('click', fetchOSMNetwork);
}

function showLoading(show, message = 'Loading...') {
    const el = document.getElementById('studio-loading-overlay');
    if (!el) return;
    if (show) {
        document.getElementById('studio-loading-msg').textContent = message;
        el.classList.remove('d-none');
    } else {
        el.classList.add('d-none');
    }
}
