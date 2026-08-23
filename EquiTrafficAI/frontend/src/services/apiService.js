// Production API Service connecting React Leaflet GIS Frontend directly to FastAPI Backend

const BACKEND_URL = "http://127.0.0.1:8000";

// Fallback sample sensors for instant render if backend is starting
export const MOCK_SENSORS = [
  { sensor_id: "773869", lat: 34.15497, lng: -118.31829, name: "US-101 N" },
  { sensor_id: "767541", lat: 34.11621, lng: -118.23799, name: "I-5 S" },
  { sensor_id: "767542", lat: 34.11631, lng: -118.23805, name: "I-5 N" },
  { sensor_id: "717447", lat: 34.07246, lng: -118.27091, name: "US-101 S" },
  { sensor_id: "717446", lat: 34.07142, lng: -118.26973, name: "US-101 N" }
];

export const fetchSensors = async (city = "la") => {
  try {
    const res = await fetch(`${BACKEND_URL}/api/state?city=${city}`);
    if (res.ok) {
      const data = await res.json();
      return {
        dataset: data.city || "metr-la",
        center: city === "sd" ? { lat: 32.7157, lng: -117.1611 } : { lat: 34.0522, lng: -118.2437 },
        sensors: data.sensors || MOCK_SENSORS,
        edges: data.edges || []
      };
    }
  } catch (err) {
    console.warn("Backend connect warning, using cached GIS state:", err);
  }
  return {
    dataset: "metr-la",
    center: { lat: 34.0522, lng: -118.2437 },
    sensors: MOCK_SENSORS,
    edges: []
  };
};

export const fetchTrafficState = async (timestampIndex = 0, city = "la") => {
  try {
    const res = await fetch(`${BACKEND_URL}/api/predict/congestion_15min?city=${city}`);
    if (res.ok) {
      const data = await res.json();
      return {
        timestampIndex,
        readings: data.congested_nodes || []
      };
    }
  } catch (err) {
    console.warn("Traffic forecast fetch warning:", err);
  }
  return { timestampIndex, readings: [] };
};

export const planSmartRoute = async (originId, destId, city = "la") => {
  try {
    const res = await fetch(`${BACKEND_URL}/api/route/plan`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ origin_id: originId, destination_id: destId, target_time: "08:30 AM", city })
    });
    if (res.ok) {
      return await res.json();
    }
  } catch (err) {
    console.error("Route planning fetch error:", err);
  }
  return null;
};
