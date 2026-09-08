function refreshImages() {
  const stamp = Date.now();
  document.querySelectorAll('.live-image').forEach(image => {
    image.src = `/data/${image.dataset.file}?t=${stamp}`;
  });
}

async function recalibrate() {
  const button = document.getElementById('recalibrate');
  button.disabled = true;
  button.textContent = 'Realigning…';
  try {
    const response = await fetch('/actions/recalibrate', {method: 'POST'});
    if (!response.ok) throw new Error('Request failed');
    button.textContent = 'Requested';
  } catch (error) {
    button.textContent = 'Failed';
  } finally {
    setTimeout(() => {
      button.disabled = false;
      button.textContent = 'Recalibrate';
    }, 1200);
  }
}

const VIEWS = ['top', 'left'];
let selection = null;

function viewImage(name) {
  return document.querySelector(`.view[data-view="${name}"] img`);
}

function viewTile(name) {
  return document.querySelector(`.view[data-view="${name}"]`).closest('.tile');
}

function imageFrame(img) {
  const rect = img.getBoundingClientRect();
  if (!img.naturalWidth || !img.naturalHeight) return null;
  const scale = Math.min(
    rect.width / img.naturalWidth, rect.height / img.naturalHeight);
  const width = img.naturalWidth * scale;
  const height = img.naturalHeight * scale;
  return {
    scale,
    left: (rect.width - width) / 2,
    top: (rect.height - height) / 2,
    width,
    height,
    rect,
  };
}

function imagePoint(img, event) {
  const frame = imageFrame(img);
  if (!frame) return null;
  const x = (event.clientX - frame.rect.left - frame.left) / frame.scale;
  const y = (event.clientY - frame.rect.top - frame.top) / frame.scale;
  if (x < 0 || y < 0 || x >= img.naturalWidth || y >= img.naturalHeight) {
    return null;
  }
  return [x, y];
}

function placeMarkers() {
  VIEWS.forEach(name => {
    const marker = document.getElementById(`${name}-marker`);
    const point = selection?.[name];
    const frame = imageFrame(viewImage(name));
    if (!point || !frame) {
      marker.hidden = true;
      return;
    }
    marker.hidden = false;
    marker.style.left = `${frame.left + point[0] * frame.scale}px`;
    marker.style.top = `${frame.top + point[1] * frame.scale}px`;
  });
}

function activeView() {
  return selection ? VIEWS.find(name => !selection[name]) : null;
}

function renderSelection() {
  document.body.classList.toggle('selecting', selection !== null);
  const active = activeView();
  VIEWS.forEach(name => viewTile(name).classList.toggle('active', name === active));
  const button = document.getElementById('select-targets');
  button.textContent = selection ? 'Cancel selection' : 'Select targets';
  placeMarkers();
}

function beginSelection() {
  selection = {top: null, left: null};
  renderSelection();
}

function endSelection() {
  selection = null;
  renderSelection();
}

async function submitSelection(points) {
  const button = document.getElementById('select-targets');
  button.disabled = true;
  button.textContent = 'Saving targets…';
  try {
    const response = await fetch('/actions/select-targets', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify(points),
    });
    if (!response.ok) throw new Error('Request failed');
  } catch (error) {
    document.getElementById('overall-text').textContent = 'Selection failed';
  } finally {
    button.disabled = false;
    endSelection();
  }
}

function pickPoint(name, event) {
  if (activeView() !== name) return;
  const point = imagePoint(viewImage(name), event);
  if (!point) return;
  selection[name] = point;
  renderSelection();
  if (VIEWS.every(view => selection[view])) submitSelection(selection);
}

function toggleSelection() {
  if (selection) endSelection(); else beginSelection();
}

let outlierRequestTimer;

function requestOutlierTrim() {
  const slider = document.getElementById('outlier-trim');
  const value = Number(slider.value);
  document.getElementById('outlier-value').textContent = `${value}%`;
  clearTimeout(outlierRequestTimer);
  outlierRequestTimer = setTimeout(async () => {
    try {
      await fetch('/actions/outlier-trim', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({percentage: value}),
      });
    } catch (error) {
      document.getElementById('outlier-value').textContent = 'failed';
    }
  }, 80);
}

function updateCamera(name, data, fresh) {
  const metrics = document.getElementById(`${name}-metrics`);
  if (selection) {
    metrics.textContent = activeView() === name
      ? 'Click the object' : selection[name] ? 'Picked' : 'Waiting';
    return;
  }
  if (!data) {
    metrics.textContent = 'Waiting';
    return;
  }
  const fps = data.fps ? `${data.fps.toFixed(1)} fps` : '— fps';
  const area = data.mask_area
    ? `${data.mask_area.toLocaleString()} px` : 'no mask';
  const match = data.match_score ? ` · match ${data.match_score.toFixed(2)}` : '';
  metrics.textContent = fresh ? `${fps} · ${area}${match}` : 'Stale';
}

async function refreshStatus() {
  try {
    const response = await fetch('/status.json', {cache: 'no-store'});
    const status = await response.json();
    const fresh = status.running
      && Date.now() / 1000 - status.updated_at < 3;
    document.getElementById('overall-dot').className
      = `dot ${fresh ? 'live' : ''}`;
    document.getElementById('overall-text').textContent = fresh
      ? status.phase : 'Tracker offline';
    updateCamera('top', status.cameras?.top, fresh);
    updateCamera('left', status.cameras?.left, fresh);
    const slider = document.getElementById('outlier-trim');
    if (document.activeElement !== slider) {
      const percentage = status.outlier_trim_percent ?? 0;
      slider.value = percentage;
      document.getElementById('outlier-value').textContent = `${percentage}%`;
    }
  } catch (error) {
    document.getElementById('overall-text').textContent = 'Tracker offline';
    document.getElementById('overall-dot').className = 'dot';
  }
}

refreshImages();
refreshStatus();
document.getElementById('recalibrate').addEventListener('click', recalibrate);
document.getElementById('select-targets').addEventListener('click', toggleSelection);
VIEWS.forEach(name => viewImage(name).addEventListener('click', event => pickPoint(name, event)));
document.addEventListener('keydown', event => {
  if (event.key === 'Escape' && selection) endSelection();
});
window.addEventListener('resize', placeMarkers);
document.getElementById('outlier-trim').addEventListener('input', requestOutlierTrim);
setInterval(refreshImages, 200);
setInterval(refreshStatus, 800);
