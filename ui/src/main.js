/* main.js – UI logic for PathSnap demo */

/* -------------------------------------------------
   UPDATE THIS line with the actual IP of Laptop A
   (the machine running `uvicorn`).  Change the port
   if you started the API on a non‑default port.
   ------------------------------------------------- */
const API_BASE = "http://<LAPTOP_A_IP>:8000"; // e.g. "http://192.168.1.12:8000"
/* ------------------------------------------------- */

const tripListDiv = document.getElementById('trip-list');
const resultSection = document.getElementById('results');
const resultPre = document.getElementById('result-json');
const mapIframe = document.getElementById('map-frame');

async function fetchJSON(url) {
  const resp = await fetch(url);
  if (!resp.ok) {
    const txt = await resp.text();
    throw new Error(`❗ ${resp.status} ${resp.statusText}: ${txt}`);
  }
  return resp.json();
}

async function loadTrips() {
  try {
    const data = await fetchJSON(`${API_BASE}/trips`);
    const trips = data.trips;
    if (!trips || trips.length === 0) {
      tripListDiv.innerHTML = '<p>No trips found.</p>';
      return;
    }
    tripListDiv.innerHTML = '';
    trips.forEach(id => {
      const btn = document.createElement('button');
      btn.textContent = id;
      btn.onclick = () => loadTrip(id);
      tripListDiv.appendChild(btn);
    });
  } catch (e) {
    tripListDiv.innerHTML = `<p class="error">Error loading trips: ${e.message}</p>`;
  }
}

async function loadTrip(tripId) {
  resultSection.hidden = true;
  resultPre.textContent = 'Loading…';
  mapIframe.src = '';
  try {
    const pred = await fetchJSON(`${API_BASE}/predict/${tripId}`);
    resultPre.textContent = JSON.stringify(pred, null, 2);
    mapIframe.src = `${API_BASE}/map/${tripId}`;
    resultSection.hidden = false;
  } catch (e) {
    resultPre.textContent = `❌ Failed: ${e.message}`;
    resultSection.hidden = false;
  }
}

loadTrips();
