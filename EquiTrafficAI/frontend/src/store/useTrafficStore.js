import { create } from 'zustand';
import { fetchSensors, fetchTrafficState } from '../services/apiService';

const useTrafficStore = create((set, get) => ({
  // State
  sensors: [],
  mapCenter: { lat: 34.0522, lng: -118.2437 },
  trafficData: [],
  currentTimestampIndex: 0,
  isLoading: false,
  error: null,

  // Actions
  initializeData: async () => {
    set({ isLoading: true, error: null });
    try {
      const topology = await fetchSensors();
      const initialTraffic = await fetchTrafficState(0);
      
      set({
        sensors: topology.sensors,
        mapCenter: topology.center,
        trafficData: initialTraffic.readings,
        isLoading: false
      });
    } catch (err) {
      set({ error: err.message, isLoading: false });
    }
  },

  setTimestampIndex: async (index, city = "la") => {
    // Only update if it changed
    if (index === get().currentTimestampIndex) return;
    
    set({ currentTimestampIndex: index });
    
    try {
      const newData = await fetchTrafficState(index, city);
      set({ trafficData: newData.readings });
    } catch (err) {
      console.error("Failed to fetch traffic state", err);
    }
  }
}));

export default useTrafficStore;
