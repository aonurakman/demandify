// demandify - Offline dataset builder UI

let map;
let drawnItems;
let currentBbox = null;
let currentJobId = null;
let progressInterval = null;
let progressRequestInFlight = false;
let existingDatasets = [];
let knownDatasets = [];

document.addEventListener("DOMContentLoaded", function () {
    loadKnownDatasetsFromPage();
    initMap();
    initEventListeners();
    loadExistingDatasets();
});

function initMap() {
    map = L.map("map").setView([48.8566, 2.3522], 12);

    L.tileLayer("https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png", {
        attribution: "© OpenStreetMap contributors",
        maxZoom: 18
    }).addTo(map);

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
                    color: "#2563eb",
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

    map.on(L.Draw.Event.CREATED, function (event) {
        const layer = event.layer;
        drawnItems.clearLayers();
        drawnItems.addLayer(layer);

        const bounds = layer.getBounds();
        const sw = bounds.getSouthWest();
        const ne = bounds.getNorthEast();

        currentBbox = {
            west: sw.lng,
            south: sw.lat,
            east: ne.lng,
            north: ne.lat
        };
        updateBboxForm();
    });

    map.on(L.Draw.Event.DELETED, function () {
        currentBbox = null;
        updateBboxForm();
    });
}

function initEventListeners() {
    ["bbox_west", "bbox_north", "bbox_south", "bbox_east"].forEach(id => {
        const el = document.getElementById(id);
        if (el) {
            el.addEventListener("change", updateMapFromInputs);
        }
    });

    const datasetNameInput = document.getElementById("dataset_name");
    const warningEl = document.getElementById("dataset-name-warning");
    if (datasetNameInput && warningEl) {
        datasetNameInput.addEventListener("input", function () {
            const val = this.value.trim();
            if (val && existingDatasets.includes(val)) {
                warningEl.classList.remove("d-none");
            } else {
                warningEl.classList.add("d-none");
            }
        });
        datasetNameInput.addEventListener("focus", loadExistingDatasets);
    }

    const existingDatasetSelect = document.getElementById("existing_dataset_select");
    if (existingDatasetSelect) {
        existingDatasetSelect.addEventListener("change", function () {
            const datasetId = this.value;
            const infoEl = document.getElementById("existing-dataset-info");

            if (!datasetId) {
                if (infoEl) {
                    infoEl.textContent = knownDatasets.length > 0
                        ? "Pick one to auto-fill and draw its bounding box."
                        : "No existing datasets with metadata found in demandify_datasets or demandify/offline_datasets.";
                }
                return;
            }

            const selected = knownDatasets.find(ds => ds.id === datasetId);
            if (!selected || !selected.bbox) return;

            applyBbox(selected.bbox, true);

            if (infoEl) {
                const quality = selected.quality_label
                    ? `${String(selected.quality_label).toUpperCase()}${selected.quality_score !== null && selected.quality_score !== undefined ? ` ${selected.quality_score}/100` : ""}`
                    : "N/A";
                const createdAt = selected.created_at || "unknown";
                infoEl.textContent = `Imported bbox from ${selected.name} (${selected.source}, quality: ${quality}, created: ${createdAt}).`;
            }
        });
    }

    const form = document.getElementById("dataset-form");
    if (form) {
        form.addEventListener("submit", async function (event) {
            event.preventDefault();

            if (!currentBbox) {
                alert("Please draw a bounding box on the map.");
                return;
            }

            const datasetName = document.getElementById("dataset_name").value.trim();
            if (existingDatasets.includes(datasetName)) {
                alert("This dataset name already exists. Choose a different name.");
                return;
            }

            const buildBtn = document.getElementById("build-btn");
            const originalHtml = buildBtn.innerHTML;
            buildBtn.disabled = true;
            buildBtn.innerHTML = '<span class="spinner-border spinner-border-sm"></span> Starting...';

            const formData = new FormData(form);
            try {
                const response = await fetch("/api/datasets/build", {
                    method: "POST",
                    body: formData
                });

                if (!response.ok) {
                    const error = await response.json();
                    throw new Error(error.detail || "Failed to start dataset build");
                }

                const data = await response.json();
                currentJobId = data.job_id;
                showProgressPanel();
                startProgressPolling();
            } catch (error) {
                alert(error.message || "Failed to start dataset build");
                buildBtn.disabled = false;
                buildBtn.innerHTML = originalHtml;
            }
        });
    }

    const newBuildBtn = document.getElementById("new-build-btn");
    if (newBuildBtn) {
        newBuildBtn.addEventListener("click", function () {
            resetUI();
            loadExistingDatasets();
        });
    }
}

function loadKnownDatasetsFromPage() {
    try {
        const el = document.getElementById("known-datasets-json");
        if (!el) {
            knownDatasets = [];
            return;
        }
        const parsed = JSON.parse(el.textContent || "[]");
        knownDatasets = Array.isArray(parsed) ? parsed : [];
    } catch (error) {
        console.error("Failed to parse known datasets payload", error);
        knownDatasets = [];
    }
}

async function loadExistingDatasets() {
    try {
        const response = await fetch("/api/datasets");
        if (response.ok) {
            const data = await response.json();
            existingDatasets = data.datasets || [];
        }
    } catch (error) {
        console.error("Failed to load datasets", error);
    }
}

function showProgressPanel() {
    document.getElementById("info-panel").style.display = "none";
    document.getElementById("progress-panel").style.display = "block";
}

function startProgressPolling() {
    stopProgressPolling();
    progressInterval = setInterval(async function () {
        if (!currentJobId) return;
        if (progressRequestInFlight) return;

        progressRequestInFlight = true;
        const jobIdAtRequestStart = currentJobId;
        try {
            const response = await fetch(`/api/datasets/${jobIdAtRequestStart}/progress?ts=${Date.now()}`, {
                cache: "no-store"
            });
            if (!response.ok) return;

            const progress = await response.json();
            if (!currentJobId || currentJobId !== jobIdAtRequestStart) return;
            updateProgress(progress);

            if (progress.status === "completed" || progress.status === "failed") {
                stopProgressPolling();
                onTerminalStatus(progress);
            }
        } catch (error) {
            console.error("Error fetching dataset progress", error);
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

function updateProgress(progress) {
    const steps = document.querySelectorAll(".step");
    const stage = Math.min(progress.stage || 0, steps.length - 1);

    steps.forEach((step, index) => {
        step.classList.remove("active", "completed");
        if (index < stage) {
            step.classList.add("completed");
        } else if (index === stage) {
            step.classList.add("active");
        }
    });

    const stageName = document.getElementById("current-stage-name");
    stageName.textContent = progress.stage_name || "Running";

    const badge = document.getElementById("job-status-badge");
    badge.textContent = progress.status || "running";
    badge.className = "badge " + statusBadgeClass(progress.status);

    const logConsole = document.getElementById("log-console");
    if (progress.logs && progress.logs.length > 0) {
        logConsole.innerHTML = "";
        progress.logs.forEach(log => {
            const entry = document.createElement("div");
            entry.className = "log-entry " + (log.level || "info");
            entry.textContent = log.message;
            logConsole.appendChild(entry);
        });
        logConsole.scrollTop = logConsole.scrollHeight;
    }
}

function statusBadgeClass(status) {
    if (status === "completed") return "bg-success";
    if (status === "failed") return "bg-danger";
    return "bg-primary";
}

function onTerminalStatus(progress) {
    const messageBox = document.getElementById("completed-message");
    const messageText = document.getElementById("completed-text");
    const newBuildBtn = document.getElementById("new-build-btn");

    if (progress.status === "completed") {
        const outputDir = progress.output_dir || "demandify_datasets/<name>";
        messageText.textContent = `Dataset build completed. Files saved in ${outputDir}.`;
        messageBox.classList.remove("d-none", "alert-danger");
        messageBox.classList.add("alert-success");
    } else {
        messageText.textContent = "Dataset build failed. Check logs for details.";
        messageBox.classList.remove("d-none", "alert-success");
        messageBox.classList.add("alert-danger");
    }
    newBuildBtn.classList.remove("d-none");
}

function updateBboxForm() {
    const areaLabel = document.getElementById("bbox-area");

    if (!currentBbox) {
        document.getElementById("bbox_west").value = "";
        document.getElementById("bbox_south").value = "";
        document.getElementById("bbox_east").value = "";
        document.getElementById("bbox_north").value = "";
        areaLabel.textContent = "";
        return;
    }

    document.getElementById("bbox_west").value = currentBbox.west.toFixed(4);
    document.getElementById("bbox_south").value = currentBbox.south.toFixed(4);
    document.getElementById("bbox_east").value = currentBbox.east.toFixed(4);
    document.getElementById("bbox_north").value = currentBbox.north.toFixed(4);
    areaLabel.textContent = `Area: ~${calculateBboxArea(currentBbox).toFixed(2)} km²`;
}

function applyBbox(bbox, fitMap) {
    const west = parseFloat(bbox.west);
    const east = parseFloat(bbox.east);
    const south = parseFloat(bbox.south);
    const north = parseFloat(bbox.north);

    if (isNaN(west) || isNaN(east) || isNaN(south) || isNaN(north)) return;
    if (west >= east || south >= north) return;

    const bounds = [[south, west], [north, east]];
    drawnItems.clearLayers();
    const rect = L.rectangle(bounds, { color: "#2563eb", weight: 2 });
    drawnItems.addLayer(rect);
    if (fitMap) {
        map.fitBounds(bounds);
    }

    currentBbox = { west, south, east, north };
    updateBboxForm();
}

function updateMapFromInputs() {
    const west = parseFloat(document.getElementById("bbox_west").value);
    const east = parseFloat(document.getElementById("bbox_east").value);
    const south = parseFloat(document.getElementById("bbox_south").value);
    const north = parseFloat(document.getElementById("bbox_north").value);

    if (isNaN(west) || isNaN(east) || isNaN(south) || isNaN(north)) return;
    if (west >= east || south >= north) {
        alert("Invalid bounds: West < East and South < North are required.");
        return;
    }

    applyBbox({ west, south, east, north }, true);
}

function calculateBboxArea(bbox) {
    const latKmPerDeg = 111.0;
    const avgLat = (bbox.south + bbox.north) / 2;
    const lonKmPerDeg = 111.0 * Math.cos(avgLat * Math.PI / 180);
    const width = (bbox.east - bbox.west) * lonKmPerDeg;
    const height = (bbox.north - bbox.south) * latKmPerDeg;
    return width * height;
}

function resetUI() {
    stopProgressPolling();
    currentJobId = null;

    document.getElementById("progress-panel").style.display = "none";
    document.getElementById("info-panel").style.display = "block";

    const buildBtn = document.getElementById("build-btn");
    buildBtn.disabled = false;
    buildBtn.innerHTML = '<i class="bi bi-play-circle"></i> Build Offline Dataset';

    document.getElementById("log-console").innerHTML = "";
    document.getElementById("current-stage-name").textContent = "Initializing";

    const messageBox = document.getElementById("completed-message");
    messageBox.classList.add("d-none");
    messageBox.classList.remove("alert-danger");
    messageBox.classList.add("alert-success");

    document.getElementById("new-build-btn").classList.add("d-none");
}
