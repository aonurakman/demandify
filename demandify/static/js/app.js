// demandify - Main JavaScript

let map;
let drawnItems;
let currentBbox = null;
let currentRunId = null;
let progressInterval = null;

// Initialize map
document.addEventListener('DOMContentLoaded', function () {
    initMap();
    initEventListeners();
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

    const drawControl = new L.Control.Draw({
        draw: {
            polyline: false,
            polygon: false,
            circle: false,
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
    map.addControl(drawControl);

    // Handle rectangle drawn
    map.on(L.Draw.Event.CREATED, function (event) {
        const layer = event.layer;

        // Clear previous rectangles
        drawnItems.clearLayers();
        drawnItems.addLayer(layer);

        // Get bounds
        const bounds = layer.getBounds();
        const sw = bounds.getSouthWest();
        const ne = bounds.getNorthEast();

        currentBbox = {
            west: sw.lng,
            south: sw.lat,
            east: ne.lng,
            north: ne.lat
        };

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

function calculateBboxArea(bbox) {
    const latKmPerDeg = 111.0;
    const avgLat = (bbox.south + bbox.north) / 2;
    const lonKmPerDeg = 111.0 * Math.cos(avgLat * Math.PI / 180);

    const width = (bbox.east - bbox.west) * lonKmPerDeg;
    const height = (bbox.north - bbox.south) * latKmPerDeg;

    return width * height;
}

function updateMapFromInputs() {
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

    // Create bounds: [[south, west], [north, east]]
    const bounds = [[south, west], [north, east]];

    // Clear existing layers
    drawnItems.clearLayers();

    // Add new rectangle
    const rect = L.rectangle(bounds, { color: '#2563eb', weight: 2 });
    drawnItems.addLayer(rect);

    // Pan map
    map.fitBounds(bounds);

    // Update state
    currentBbox = { west, south, east, north };

    // Update area display
    const area = calculateBboxArea(currentBbox);
    document.getElementById('bbox-area').textContent = `Area: ~${area.toFixed(2)} km²`;

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

        if (!currentBbox) {
            alert('Please draw a bounding box on the map');
            return;
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

                const warningEl = document.getElementById('check-warning');
                const criticalEl = document.getElementById('check-critical');

                warningEl.classList.add('d-none');
                criticalEl.classList.add('d-none');

                const matched = data.stats.matched_edges;
                if (matched === 0) {
                    criticalEl.classList.remove('d-none');
                } else if (matched < 5) {
                    warningEl.classList.remove('d-none');
                }

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
        // Hide modal
        const modalEl = document.getElementById('checkModal');
        const modal = bootstrap.Modal.getInstance(modalEl);
        modal.hide();

        // Start Actual Run
        if (!pendingRunId) return;

        // Show progress panel
        document.getElementById('info-panel').style.display = 'none';
        document.getElementById('progress-panel').style.display = 'block';
        document.getElementById('run-btn').disabled = true;

        const formData = new FormData(runForm);
        // Ensure we use the SAME run_id
        formData.set('run_id', pendingRunId);

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
    progressInterval = setInterval(async function () {
        if (!currentRunId) return;

        try {
            const response = await fetch(`/api/run/${currentRunId}/progress`);
            if (response.ok) {
                const progress = await response.json();
                updateProgress(progress);
            }
        } catch (error) {
            console.error('Error fetching progress:', error);
        }
    }, 1000);
}

function stopProgressPolling() {
    if (progressInterval) {
        clearInterval(progressInterval);
        progressInterval = null;
    }
}

function updateProgress(progress) {
    // Update stepper
    const steps = document.querySelectorAll('.step');
    steps.forEach((step, index) => {
        step.classList.remove('active', 'completed');
        if (index < progress.stage) {
            step.classList.add('completed');
        } else if (index === progress.stage) {
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
}

function resetUI() {
    document.getElementById('info-panel').style.display = 'block';
    document.getElementById('progress-panel').style.display = 'none';
    document.getElementById('run-btn').disabled = false;
    currentRunId = null;
}
