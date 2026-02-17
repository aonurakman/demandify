// demandify - Main JavaScript

let map;
let drawnItems;
let drawControl;
let drawControlAttached = false;
let currentBbox = null;
let currentRunId = null;
let progressInterval = null;
let progressRequestInFlight = false;
let dataMode = 'create';
let knownOfflineDatasets = [];
const OFFLINE_DATASET_NAME_PATTERN = /^[A-Za-z0-9_-]+$/;

// Initialize map
document.addEventListener('DOMContentLoaded', function () {
    loadKnownOfflineDatasetsFromPage();
    initMap();
    initEventListeners();
    setDataMode(document.getElementById('data_mode')?.value || 'create');
});

function initMap() {
    // Create map centered on Europe
    map = L.map('map').setView([48.8566, 2.3522], 12);

    // Add OpenStreetMap tiles
    L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
        attribution: '© OpenStreetMap contributors',
        maxZoom: 18
    }).addTo(map);

    // Initialize drawing controls
    drawnItems = new L.FeatureGroup();
    map.addLayer(drawnItems);

    drawControl = new L.Control.Draw({
        draw: {
            polyline: false,
            polygon: false,
            circle: {
                shapeOptions: {
                    color: '#2563eb',
                    weight: 2
                }
            },
            marker: false,
            circlemarker: false,
            rectangle: {
                shapeOptions: {
                    color: '#2563eb',
                    weight: 2
                }
            }
        },
        edit: {
            featureGroup: drawnItems,
            remove: true
        }
    });
    attachDrawControl();

    // Handle rectangle drawn
    map.on(L.Draw.Event.CREATED, function (event) {
        if (dataMode !== 'create') return;
        const layer = event.layer;

        // Clear previous rectangles
        drawnItems.clearLayers();
        drawnItems.addLayer(layer);

        if (event.layerType === 'circle') {
            const center = layer.getLatLng();
            const radiusMeters = layer.getRadius();
            const latKmPerDeg = 111.0;
            const lonKmPerDeg = 111.0 * Math.cos(center.lat * Math.PI / 180);

            const latDelta = (radiusMeters / 1000.0) / latKmPerDeg;
            const lonDelta = (radiusMeters / 1000.0) / lonKmPerDeg;

            currentBbox = {
                west: center.lng - lonDelta,
                south: center.lat - latDelta,
                east: center.lng + lonDelta,
                north: center.lat + latDelta
            };
        } else {
            // Get bounds (rectangle)
            const bounds = layer.getBounds();
            const sw = bounds.getSouthWest();
            const ne = bounds.getNorthEast();

            currentBbox = {
                west: sw.lng,
                south: sw.lat,
                east: ne.lng,
                north: ne.lat
            };
        }

        // Update form
        updateBboxForm();
    });

    map.on(L.Draw.Event.DELETED, function () {
        currentBbox = null;
        updateBboxForm();
    });
}

function updateBboxForm() {
    if (currentBbox) {
        document.getElementById('bbox_west').value = parseFloat(currentBbox.west).toFixed(4);
        document.getElementById('bbox_south').value = parseFloat(currentBbox.south).toFixed(4);
        document.getElementById('bbox_east').value = parseFloat(currentBbox.east).toFixed(4);
        document.getElementById('bbox_north').value = parseFloat(currentBbox.north).toFixed(4);

        // Calculate area
        const area = calculateBboxArea(currentBbox);
        document.getElementById('bbox-area').textContent =
            `Area: ~${area.toFixed(2)} km²`;

        // Enable run button if API key exists
        const runBtn = document.getElementById('run-btn');
        if (!runBtn.disabled) {
            // Already enabled due to API key
        }
    } else {
        document.getElementById('bbox_west').value = '';
        document.getElementById('bbox_south').value = '';
        document.getElementById('bbox_east').value = '';
        document.getElementById('bbox_north').value = '';
        document.getElementById('bbox-area').textContent = '';
    }
}

function loadKnownOfflineDatasetsFromPage() {
    try {
        const el = document.getElementById('known-offline-datasets-json');
        if (!el) {
            knownOfflineDatasets = [];
            return;
        }
        const parsed = JSON.parse(el.textContent || '[]');
        knownOfflineDatasets = Array.isArray(parsed) ? parsed : [];
    } catch (error) {
        console.error('Failed to parse offline dataset payload', error);
        knownOfflineDatasets = [];
    }
}

function calculateBboxArea(bbox) {
    const latKmPerDeg = 111.0;
    const avgLat = (bbox.south + bbox.north) / 2;
    const lonKmPerDeg = 111.0 * Math.cos(avgLat * Math.PI / 180);

    const width = (bbox.east - bbox.west) * lonKmPerDeg;
    const height = (bbox.north - bbox.south) * latKmPerDeg;

    return width * height;
}

function toggleSaveOfflineDatasetNameInput() {
    const checkbox = document.getElementById('save-offline-dataset-checkbox');
    const group = document.getElementById('save-offline-dataset-name-group');
    const input = document.getElementById('save-offline-dataset-name');
    if (!checkbox || !group || !input) return;

    if (checkbox.checked) {
        group.classList.remove('d-none');
    } else {
        group.classList.add('d-none');
        input.value = '';
    }
}

function isValidOfflineDatasetName(name) {
    return OFFLINE_DATASET_NAME_PATTERN.test(name);
}

function attachDrawControl() {
    if (drawControl && !drawControlAttached) {
        map.addControl(drawControl);
        drawControlAttached = true;
    }
}

function detachDrawControl() {
    if (drawControl && drawControlAttached) {
        map.removeControl(drawControl);
        drawControlAttached = false;
    }
}

function applyBboxToMap(bbox, fitMap) {
    const west = parseFloat(bbox.west);
    const east = parseFloat(bbox.east);
    const south = parseFloat(bbox.south);
    const north = parseFloat(bbox.north);
    if (isNaN(west) || isNaN(east) || isNaN(south) || isNaN(north)) return;
    if (west >= east || south >= north) return;

    const bounds = [[south, west], [north, east]];
    drawnItems.clearLayers();
    drawnItems.addLayer(L.rectangle(bounds, { color: '#2563eb', weight: 2 }));
    if (fitMap) {
        map.fitBounds(bounds);
    }
    currentBbox = { west, south, east, north };
    updateBboxForm();
}

function setDataMode(mode) {
    dataMode = (mode || 'create').toLowerCase() === 'import' ? 'import' : 'create';
    const importGroup = document.getElementById('import-dataset-group');
    const helpText = document.getElementById('bbox-help-text');
    const summaryEl = document.getElementById('offline-dataset-summary');
    const tileZoomInput = document.getElementById('traffic_tile_zoom');

    const bboxInputs = document.querySelectorAll('.bbox-input');
    if (dataMode === 'import') {
        if (importGroup) importGroup.classList.remove('d-none');
        detachDrawControl();
        if (tileZoomInput) tileZoomInput.disabled = true;
        bboxInputs.forEach(el => {
            el.readOnly = true;
            el.disabled = true;
        });
        if (helpText) helpText.textContent = 'Import mode: select a dataset to visualize its fixed calibration area.';
        const selectedId = document.getElementById('offline_dataset')?.value || '';
        if (selectedId) {
            applySelectedOfflineDataset(selectedId);
        } else {
            currentBbox = null;
            drawnItems.clearLayers();
            updateBboxForm();
            if (summaryEl) summaryEl.textContent = 'Select an offline dataset to import its bbox.';
        }
    } else {
        if (importGroup) importGroup.classList.add('d-none');
        attachDrawControl();
        if (tileZoomInput) tileZoomInput.disabled = false;
        bboxInputs.forEach(el => {
            el.readOnly = false;
            el.disabled = false;
        });
        if (helpText) helpText.textContent = 'Draw a rectangle (or circle) on the map to select your calibration area';
        if (summaryEl) summaryEl.textContent = '';
    }
}

function applySelectedOfflineDataset(datasetId) {
    const dataset = knownOfflineDatasets.find(ds => ds.id === datasetId);
    const summaryEl = document.getElementById('offline-dataset-summary');
    if (!dataset || !dataset.bbox) {
        if (summaryEl) summaryEl.textContent = 'Selected dataset is missing bbox metadata.';
        return;
    }
    applyBboxToMap(dataset.bbox, true);
    if (summaryEl) {
        const quality = dataset.quality_label
            ? `${String(dataset.quality_label).toUpperCase()}${dataset.quality_score !== null && dataset.quality_score !== undefined ? ` (${dataset.quality_score}/100)` : ''}`
            : 'N/A';
        summaryEl.textContent = `${dataset.name} [${dataset.source}] selected. Quality: ${quality}.`;
    }
}

function updateMapFromInputs() {
    if (dataMode !== 'create') return;
    const west = parseFloat(document.getElementById('bbox_west').value);
    const east = parseFloat(document.getElementById('bbox_east').value);
    const south = parseFloat(document.getElementById('bbox_south').value);
    const north = parseFloat(document.getElementById('bbox_north').value);

    // Validate
    if (isNaN(west) || isNaN(east) || isNaN(south) || isNaN(north)) return;
    if (west >= east || south >= north) {
        alert("Invalid bounds: West must be < East, South must be < North");
        return;
    }

    applyBboxToMap({ west, south, east, north }, true);

    // Enable run button logic check
    const runBtn = document.getElementById('run-btn');
    if (runBtn && runBtn.disabled && document.getElementById('api-key-input') === null) {
        runBtn.disabled = false;
    }
}

function initEventListeners() {
    // API key save
    const saveKeyBtn = document.getElementById('save-api-key-btn');
    if (saveKeyBtn) {
        saveKeyBtn.addEventListener('click', async function () {
            const keyInput = document.getElementById('api-key-input');
            const key = keyInput.value.trim();

            if (!key) {
                alert('Please enter an API key');
                return;
            }

            try {
                const formData = new FormData();
                formData.append('service', 'tomtom');
                formData.append('key', key);

                const response = await fetch('/api/config/api-key', {
                    method: 'POST',
                    body: formData
                });

                if (response.ok) {
                    alert('API key saved successfully!');
                    location.reload();
                } else {
                    alert('Failed to save API key');
                }
            } catch (error) {
                console.error('Error saving API key:', error);
                alert('Error saving API key');
            }
        });
    }

    // Bind bbox inputs for manual editing
    ['bbox_west', 'bbox_north', 'bbox_south', 'bbox_east'].forEach(id => {
        const el = document.getElementById(id);
        if (el) {
            el.addEventListener('change', updateMapFromInputs);
        }
    });

    // Duplicate Run ID Check
    const runIdInput = document.getElementById('run_id');
    const warningEl = document.getElementById('run-id-warning');
    let existingRuns = [];

    async function loadExistingRuns() {
        try {
            const resp = await fetch('/api/runs');
            if (resp.ok) {
                const data = await resp.json();
                existingRuns = data.runs;
            }
        } catch (e) {
            console.error('Failed to load runs', e);
        }
    }

    // Data mode and offline dataset selection
    const dataModeSelect = document.getElementById('data_mode');
    if (dataModeSelect) {
        dataModeSelect.addEventListener('change', function () {
            setDataMode(this.value);
        });
    }

    const offlineDatasetSelect = document.getElementById('offline_dataset');
    if (offlineDatasetSelect) {
        offlineDatasetSelect.addEventListener('change', function () {
            if (dataMode === 'import') {
                applySelectedOfflineDataset(this.value);
            }
        });
    }

    const saveOfflineDatasetCheckbox = document.getElementById('save-offline-dataset-checkbox');
    if (saveOfflineDatasetCheckbox) {
        saveOfflineDatasetCheckbox.addEventListener('change', toggleSaveOfflineDatasetNameInput);
    }

    // Initial load
    if (runIdInput) loadExistingRuns();

    if (runIdInput && warningEl) {
        runIdInput.addEventListener('input', function () {
            const val = this.value.trim();
            if (val && existingRuns.includes(val)) {
                warningEl.classList.remove('d-none');
            } else {
                warningEl.classList.add('d-none');
            }
        });

        // Also check on focus just in case list updated
        runIdInput.addEventListener('focus', loadExistingRuns);
    }

    // Window duration change handler
    const windowSelect = document.getElementById('window_minutes');
    if (windowSelect) {
        windowSelect.addEventListener('change', function () {
            const maxVal = parseInt(this.value);
            const binInput = document.getElementById('bin_minutes');
            if (binInput) {
                binInput.max = maxVal;
                if (parseInt(binInput.value) > maxVal) {
                    binInput.value = maxVal;
                    document.getElementById('bin-val').textContent = maxVal;
                }
            }
        });
    }

    // Change API key button
    const changeKeyBtn = document.getElementById('change-api-key-btn');
    if (changeKeyBtn) {
        changeKeyBtn.addEventListener('click', async function () {
            const newKey = prompt('Enter new TomTom API key:');
            if (!newKey || !newKey.trim()) {
                return;
            }

            try {
                const formData = new FormData();
                formData.append('service', 'tomtom');
                formData.append('key', newKey.trim());

                const response = await fetch('/api/config/api-key', {
                    method: 'POST',
                    body: formData
                });

                if (response.ok) {
                    alert('API key updated successfully!');
                    location.reload();
                } else {
                    alert('Failed to update API key');
                }
            } catch (error) {
                console.error('Error updating API key:', error);
                alert('Error updating API key');
            }
        });
    }

    // Run form submit
    const runForm = document.getElementById('run-form');
    let pendingRunId = null;

    runForm.addEventListener('submit', async function (e) {
        e.preventDefault();
        const hasApiKeyConfigured = document.getElementById('api-key-input') === null;

        if (dataMode === 'create' && !currentBbox) {
            alert('Please draw a bounding box on the map');
            return;
        }
        if (dataMode === 'create' && !hasApiKeyConfigured) {
            alert('TomTom API key is required in Create mode. Save your key or switch to Import mode.');
            return;
        }
        if (dataMode === 'import') {
            const selected = document.getElementById('offline_dataset')?.value || '';
            if (!selected) {
                alert('Please select an offline dataset to import.');
                return;
            }
        }

        // Show loading state
        const btn = document.getElementById('run-btn');
        const originalText = btn.innerHTML;
        btn.disabled = true;
        btn.innerHTML = '<span class="spinner-border spinner-border-sm" role="status" aria-hidden="true"></span> Checking...';

        const formData = new FormData(runForm);

        try {
            // Step 1: Check Feasibility
            const response = await fetch('/api/check_feasibility', {
                method: 'POST',
                body: formData
            });

            if (response.ok) {
                const data = await response.json();

                // Store run ID for confirmation
                pendingRunId = data.run_id;

                // Populate Modal
                document.getElementById('check-fetched-count').textContent = data.stats.fetched_segments;
                document.getElementById('check-matched-count').textContent = data.stats.matched_edges;

                const totalEl = document.getElementById('check-total-edges');
                if (totalEl) totalEl.textContent = data.stats.total_network_edges || '-';
                const qualityLabelEl = document.getElementById('check-quality-label');
                const qualityScoreEl = document.getElementById('check-quality-score');
                const qualitySummaryEl = document.getElementById('check-quality-summary');

                const warningEl = document.getElementById('check-warning');
                const criticalEl = document.getElementById('check-critical');

                warningEl.classList.add('d-none');
                criticalEl.classList.add('d-none');

                if (qualityLabelEl) {
                    qualityLabelEl.textContent = '-';
                    qualityLabelEl.className = 'badge bg-info rounded-pill';
                }
                if (qualityScoreEl) {
                    qualityScoreEl.textContent = '-';
                }
                if (qualitySummaryEl) {
                    qualitySummaryEl.textContent = '-';
                }

                const quality = data.quality || null;
                if (quality) {
                    const label = (quality.label || 'unknown').toUpperCase();
                    const score = Number.isFinite(quality.score) ? quality.score : '-';
                    const recommendation = quality.recommendation || '';
                    const warnings = Array.isArray(quality.warnings) ? quality.warnings : [];

                    if (qualityLabelEl) {
                        qualityLabelEl.textContent = label;
                        let badgeClass = 'bg-secondary';
                        if (quality.label === 'excellent') badgeClass = 'bg-success';
                        else if (quality.label === 'good') badgeClass = 'bg-primary';
                        else if (quality.label === 'fair') badgeClass = 'bg-info';
                        else if (quality.label === 'weak') badgeClass = 'bg-warning';
                        else if (quality.label === 'poor') badgeClass = 'bg-danger';
                        qualityLabelEl.className = `badge ${badgeClass} rounded-pill`;
                    }

                    if (qualityScoreEl) {
                        qualityScoreEl.textContent = score !== '-' ? `${score}/100` : '-';
                    }

                    if (qualitySummaryEl) {
                        let summary = quality.summary || '-';
                        if (warnings.length > 0) {
                            summary += ` Flags: ${warnings.join(', ')}`;
                        }
                        qualitySummaryEl.textContent = summary;
                    }

                    if (recommendation === 'do_not_proceed') {
                        criticalEl.classList.remove('d-none');
                    } else if (recommendation === 'high_risk' || recommendation === 'caution') {
                        warningEl.classList.remove('d-none');
                    }
                }

                const matched = data.stats.matched_edges;
                if (matched === 0) {
                    criticalEl.classList.remove('d-none');
                } else if (matched < 5) {
                    warningEl.classList.remove('d-none');
                }

                const saveSection = document.getElementById('save-offline-dataset-section');
                const saveCheckbox = document.getElementById('save-offline-dataset-checkbox');
                const saveNameInput = document.getElementById('save-offline-dataset-name');
                if (saveSection) {
                    saveSection.classList.toggle('d-none', dataMode !== 'create');
                }
                if (saveCheckbox) saveCheckbox.checked = false;
                if (saveNameInput) saveNameInput.value = '';
                toggleSaveOfflineDatasetNameInput();

                // Show Modal
                const modal = new bootstrap.Modal(document.getElementById('checkModal'));
                modal.show();

            } else {
                const error = await response.json();
                alert('Error checking feasibility: ' + (error.detail || error.message));
            }
        } catch (error) {
            console.error('Error:', error);
            alert('Error checking feasibility');
        } finally {
            btn.disabled = false;
            btn.innerHTML = originalText;
        }
    });

    // Confirm Run Button in Modal
    document.getElementById('confirm-run-btn').addEventListener('click', async function () {
        if (!pendingRunId) return;

        const saveCheckbox = document.getElementById('save-offline-dataset-checkbox');
        const saveNameInput = document.getElementById('save-offline-dataset-name');
        const shouldSaveOffline = !!(saveCheckbox && saveCheckbox.checked);
        const saveName = saveNameInput ? saveNameInput.value.trim() : '';

        if (shouldSaveOffline) {
            if (!saveName) {
                alert('Please provide a dataset name to save offline data.');
                return;
            }
            if (!isValidOfflineDatasetName(saveName)) {
                alert('Dataset name must use only letters, numbers, underscores, and hyphens.');
                return;
            }
            const existing = knownOfflineDatasets.some(ds => ds.name === saveName);
            if (existing) {
                alert(`Offline dataset '${saveName}' already exists. Choose a different name.`);
                return;
            }
        }

        // Hide modal
        const modalEl = document.getElementById('checkModal');
        const modal = bootstrap.Modal.getInstance(modalEl);
        modal.hide();

        // Show progress panel
        document.getElementById('info-panel').style.display = 'none';
        document.getElementById('progress-panel').style.display = 'block';
        document.getElementById('run-btn').disabled = true;

        const formData = new FormData(runForm);
        // Handle boolean checkboxes: set to true/false explicitly
        ['ga_assortative_mating', 'ga_deterministic_crowding'].forEach(function(name) {
            var cb = document.getElementById(name);
            if (cb) formData.set(name, cb.checked ? 'true' : 'false');
        });
        // Ensure we use the SAME run_id
        formData.set('run_id', pendingRunId);
        formData.set('save_offline_dataset', shouldSaveOffline ? 'true' : 'false');
        if (shouldSaveOffline) {
            formData.set('save_offline_dataset_name', saveName);
        }

        try {
            const response = await fetch('/api/run', {
                method: 'POST',
                body: formData
            });

            if (response.ok) {
                const data = await response.json();
                currentRunId = data.run_id;
                startProgressPolling();
            } else {
                const error = await response.json();
                alert('Error starting run: ' + error.detail);
                resetUI();
            }
        } catch (error) {
            console.error('Error starting run:', error);
            alert('Error starting run');
            resetUI();
        }
    });

    // Cancel button
    document.getElementById('cancel-btn').addEventListener('click', function () {
        if (confirm('Are you sure you want to cancel this run?')) {
            stopProgressPolling();
            resetUI();
        }
    });

    // Restart button
    document.getElementById('restart-btn').addEventListener('click', function () {
        resetUI();
    });

    // Toggle details
    document.getElementById('toggle-details').addEventListener('click', function () {
        const console = document.getElementById('log-console');
        if (console.style.height === '600px') {
            console.style.height = '300px';
            this.textContent = 'Show Details';
        } else {
            console.style.height = '600px';
            this.textContent = 'Hide Details';
        }
    });
}

function startProgressPolling() {
    stopProgressPolling();
    progressInterval = setInterval(async function () {
        if (!currentRunId) return;
        if (progressRequestInFlight) return;

        progressRequestInFlight = true;
        const runIdAtRequestStart = currentRunId;
        try {
            const response = await fetch(`/api/run/${runIdAtRequestStart}/progress?ts=${Date.now()}`, {
                cache: 'no-store'
            });
            if (response.ok) {
                const progress = await response.json();
                if (!currentRunId || currentRunId !== runIdAtRequestStart) return;
                updateProgress(progress);

                if (progress.status === 'completed') {
                    stopProgressPolling();
                    window.location.href = `/results?run_id=${currentRunId}`;
                } else if (progress.status === 'failed') {
                    stopProgressPolling();
                    alert('Pipeline failed. Check console output for details.');
                    resetUI();
                } else if (progress.status === 'aborted') {
                    stopProgressPolling();
                    alert('Pipeline was aborted. The UI has been reset.');
                    resetUI();
                }
            }
        } catch (error) {
            console.error('Error fetching progress:', error);
        } finally {
            progressRequestInFlight = false;
        }
    }, 1000);
}

function stopProgressPolling() {
    if (progressInterval) {
        clearInterval(progressInterval);
        progressInterval = null;
    }
    progressRequestInFlight = false;
}

function showRestartOption(status) {
    const restartBar = document.getElementById('restart-bar');
    const restartMsg = document.getElementById('restart-message');
    if (!restartBar || !restartMsg) return;

    if (status === 'completed') {
        restartMsg.innerHTML = '<i class="bi bi-check-circle text-success"></i> Calibration completed successfully.';
    } else if (status === 'aborted') {
        restartMsg.innerHTML = '<i class="bi bi-dash-circle text-warning"></i> Calibration was aborted.';
    } else {
        restartMsg.innerHTML = '<i class="bi bi-x-circle text-danger"></i> Calibration failed.';
    }
    restartBar.style.display = 'block';

    // Hide cancel button since run is done
    const cancelBtn = document.getElementById('cancel-btn');
    if (cancelBtn) cancelBtn.style.display = 'none';
}

function updateProgress(progress) {
    // Update stepper
    const steps = document.querySelectorAll('.step');
    const stageIndex = Math.min(progress.stage || 0, steps.length - 1);
    steps.forEach((step, index) => {
        step.classList.remove('active', 'completed');
        if (index < stageIndex) {
            step.classList.add('completed');
        } else if (index === stageIndex) {
            step.classList.add('active');
        }
    });

    // Update stage name
    document.getElementById('current-stage-name').textContent = progress.stage_name;

    // Update logs
    const logConsole = document.getElementById('log-console');
    if (progress.logs && progress.logs.length > 0) {
        logConsole.innerHTML = '';
        progress.logs.forEach(log => {
            const entry = document.createElement('div');
            entry.className = 'log-entry ' + (log.level || '');
            entry.textContent = log.message;
            logConsole.appendChild(entry);
        });
        logConsole.scrollTop = logConsole.scrollHeight;
    }

    // Check for terminal status included in progress response
    if (progress.status && progress.status !== 'running') {
        const runId = currentRunId;
        // Ensure this status still applies to the current run
        if (runId && runId === currentRunId) {
            stopProgressPolling();
            showRestartOption(progress.status);
        }
    }
}

function resetUI() {
    document.getElementById('info-panel').style.display = 'block';
    document.getElementById('progress-panel').style.display = 'none';
    document.getElementById('run-btn').disabled = false;
    currentRunId = null;

    // Reset restart bar
    const restartBar = document.getElementById('restart-bar');
    if (restartBar) restartBar.style.display = 'none';

    // Restore cancel button visibility
    const cancelBtn = document.getElementById('cancel-btn');
    if (cancelBtn) cancelBtn.style.display = '';

    // Clear log console
    const logConsole = document.getElementById('log-console');
    if (logConsole) logConsole.innerHTML = '';

    // Reset stepper
    document.querySelectorAll('.step').forEach(step => {
        step.classList.remove('active', 'completed');
    });
    document.getElementById('current-stage-name').textContent = 'Initializing...';
}
