// Mock data simulating the METR-LA loop detector coordinates and sample data

// 10 sample sensors (METR-LA has 207, we'll mock 10 for performance in this demo)
export const MOCK_SENSORS = [
  { sensor_id: "773869", lat: 34.15497, lng: -118.31829, name: "US-101 N" },
  { sensor_id: "767541", lat: 34.11621, lng: -118.23799, name: "I-5 S" },
  { sensor_id: "767542", lat: 34.11631, lng: -118.23805, name: "I-5 N" },
  { sensor_id: "717447", lat: 34.07246, lng: -118.27091, name: "US-101 S" },
  { sensor_id: "717446", lat: 34.07142, lng: -118.26973, name: "US-101 N" },
  { sensor_id: "717445", lat: 34.06941, lng: -118.26765, name: "US-101 S" },
  { sensor_id: "773062", lat: 34.12061, lng: -118.35824, name: "US-101 N" },
  { sensor_id: "767620", lat: 34.13521, lng: -118.22557, name: "I-5 S" },
  { sensor_id: "737529", lat: 34.13601, lng: -118.22538, name: "I-5 N" },
  { sensor_id: "717816", lat: 34.05374, lng: -118.24357, name: "US-101 S" }
];

// Generate synthetic speed data for these sensors across a timeline
export const generateMockTrafficData = (timestampIndex) => {
  // Use index to simulate wave of traffic (simple oscillation)
  return MOCK_SENSORS.map(sensor => {
    // Base speed around 60, varies by sensor and time
    const variance = Math.sin(timestampIndex * 0.5 + parseInt(sensor.sensor_id)) * 30;
    let speed = 60 + variance;
    
    // clamp speed
    if (speed < 5) speed = 5;
    if (speed > 85) speed = 85;

    let status = 'fast';
    if (speed < 30) status = 'slow';
    else if (speed < 55) status = 'medium';

    return {
      sensor_id: sensor.sensor_id,
      speed: Math.round(speed * 10) / 10,
      status
    };
  });
};

export const fetchSensors = async () => {
  // Simulate network delay
  return new Promise(resolve => {
    setTimeout(() => {
      resolve({
        dataset: "metr-la",
        center: { lat: 34.0522, lng: -118.2437 }, // Los Angeles
        sensors: MOCK_SENSORS
      });
    }, 500);
  });
};

export const fetchTrafficState = async (timestampIndex) => {
  return new Promise(resolve => {
    setTimeout(() => {
      resolve({
        timestampIndex,
        readings: generateMockTrafficData(timestampIndex)
      });
    }, 300);
  });
};
