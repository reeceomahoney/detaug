function refreshImages() {
  const stamp = Date.now();
  document.querySelectorAll('.live-image').forEach(image => {
    image.src = `/data/${image.dataset.file}?t=${stamp}`;
  });
}

function refreshTargets() {
  const stamp = Date.now();
  document.querySelectorAll('.target-image').forEach(image => {
    image.src = `/targets/${image.dataset.file}?t=${stamp}`;
  });
}

async function recalibrate() {
  const button = document.getElementById('recalibrate');
  button.disabled = true;
  button.textContent = 'Realigning…';
  try {
    const response = await fetch('/actions/recalibrate', {method: 'POST'});
    if (!response.ok) throw new Error('Request failed');
    refreshTargets();
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
refreshTargets();
refreshStatus();
document.getElementById('recalibrate').addEventListener('click', recalibrate);
document.getElementById('outlier-trim').addEventListener('input', requestOutlierTrim);
setInterval(refreshImages, 200);
setInterval(refreshStatus, 800);
