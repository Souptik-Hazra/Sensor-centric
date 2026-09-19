import { create } from 'zustand';
import { fetchSensors } from '../services/apiService';

const useTrafficStore = create((set, get) => ({
  // State
  sensors: [],
  mapCenter: { lat: 34.0522, lng: -118.2437 },
  trafficData: [],
  currentTimestampIndex: 0,
  isLoading: false,
  error: null,

  // Actions
  initializeData: async ()=>{
    set({ isLoading: true, error: null });
    try {
      const topology = await fetchSensors();
      
      set({
        sensors: topology.sensors,
        mapCenter: topology.center,
        // The monitoring table needs the complete telemetry snapshot. The
        // 15-minute endpoint only returns warning nodes, not every sensor.
        trafficData: topology.sensors,
        isLoading: false
      });
    } catch (err) {
      set({ error: err.message, isLoading: false });
    }
  },

  setTimestampIndex: async (index, _city = "la") => {
    // Only update if it changed
    if(index === get().currentTimestampIndex) 
      return;
    
    set({ currentTimestampIndex: index });
    
    // The monitoring view currently has no timestamp playback control. Keep
    // this action harmless for existing callers until a full telemetry
    // timestamp endpoint is introduced.
  }
}));

export default useTrafficStore;