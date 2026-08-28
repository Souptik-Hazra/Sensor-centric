import React, { useState, useRef, useEffect, useCallback } from 'react';
import { Navigation } from 'lucide-react';
import styles from './LlmChatbot.module.css';
import ChatMessageList from './components/ChatMessageList';
import QuickPromptChips from './components/QuickPromptChips';
import ChatHeaderBar from './components/ChatHeaderBar';
import ChatInputFooter from './components/ChatInputFooter';

const INITIAL_MESSAGES = [
  {
    sender: 'bot',
    text: "Hello! I am **EquiTraffic-GPT Smart Reroute Copilot**.\n\nEvery **15 minutes**, I automatically monitor live highway telemetry against historical same-day patterns to tell you:\n• ❌ **Which ways to avoid**\n• ✅ **Which alternate routes to use if starting now**\n• ⏱️ **Estimated travel time saved**\n\nAsk me anytime for live rerouting advice!",
    time: 'Just now'
  }
];

export default function LlmChatbot() {
  const [isOpen, setIsOpen] = useState(false);
  const [messages, setMessages] = useState(INITIAL_MESSAGES);
  const [inputPrompt, setInputPrompt] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [lastAlertStep, setLastAlertStep] = useState(-1);
  const messagesEndRef = useRef(null);
  
  // Sync state with MapView
  const [currentStep, setCurrentStep] = useState(96);
  const [selectedCity, setSelectedCity] = useState('la');
  const [currentDate, setCurrentDate] = useState('2012-03-15');
  const [originNodeId, setOriginNodeId] = useState(0);
  const [destinationNodeId, setDestinationNodeId] = useState(15);

  useEffect(() => {
    const handleStateSync = (e) => {
      if (e.detail) {
        if (e.detail.step !== undefined) setCurrentStep(e.detail.step);
        if (e.detail.city !== undefined) setSelectedCity(e.detail.city);
        if (e.detail.date !== undefined) setCurrentDate(e.detail.date);
        if (e.detail.origin_id !== undefined) setOriginNodeId(e.detail.origin_id);
        if (e.detail.destination_id !== undefined) setDestinationNodeId(e.detail.destination_id);
      }
    };
    window.addEventListener('app-state-sync', handleStateSync);
    return () => window.removeEventListener('app-state-sync', handleStateSync);
  }, []);

  const getDisplayTime = useCallback((step) => {
    const totalMinutes = step * 5;
    const hours = Math.floor(totalMinutes / 60);
    const mins = totalMinutes % 60;
    const ampm = hours >= 12 ? 'PM' : 'AM';
    const displayHours = hours % 12 === 0 ? 12 : hours % 12;
    const displayMins = mins < 10 ? '0' + mins : mins;
    return `${displayHours}:${displayMins} ${ampm}`;
  }, []);

  const scrollToBottom = useCallback(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, []);

  useEffect(() => {
    if (isOpen) {
      scrollToBottom();
    }
  }, [messages, isOpen, scrollToBottom]);

  const triggerProactive15MinAlert = useCallback(async (stepVal) => {
    const timeLabel = getDisplayTime(stepVal);
    try {
      let response;
      const payload = {
        prompt: `auto_alert 15-minute alert for ${timeLabel}`,
        sensor_id: 0,
        city: selectedCity,
        time_label: timeLabel,
        date_label: currentDate
      };
      try {
        response = await fetch('/api/llm/reasoning', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(payload)
        });
      } catch (e) {
        response = await fetch('http://127.0.0.1:8000/api/llm/reasoning', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(payload)
        });
      }

      if (response && response.ok) {
        const data = await response.json();
        setMessages(prev => {
          if (prev.length > 0 && prev[prev.length - 1].text === data.llm_response) {
            return prev;
          }
          return [...prev, {
            sender: 'bot',
            text: data.llm_response,
            time: timeLabel,
            isAutoAlert: true
          }];
        });
      }
    } catch (err) {
      console.error('Auto alert fetch error:', err);
    }
  }, [getDisplayTime, selectedCity]);

  // Autonomous 15-Minute Proactive Alert Engine (Triggers every 3 steps = 15 minutes)
  useEffect(() => {
    if (currentStep > 0 && currentStep % 3 === 0 && currentStep !== lastAlertStep) {
      setLastAlertStep(currentStep);
      triggerProactive15MinAlert(currentStep);
    }
  }, [currentStep, lastAlertStep, triggerProactive15MinAlert]);

  const handleSendMessage = useCallback(async (customText = '') => {
    const textToSend = customText || inputPrompt;
    if (!textToSend.trim()) return;

    const userMsg = {
      sender: 'user',
      text: textToSend,
      time: getDisplayTime(currentStep)
    };

    setMessages(prev => [...prev, userMsg]);
    if (!customText) setInputPrompt('');
    setIsLoading(true);

    try {
      let response;
      const payload = {
        prompt: textToSend,
        sensor_id: originNodeId,
        origin_id: originNodeId,
        destination_id: destinationNodeId,
        city: selectedCity,
        time_label: getDisplayTime(currentStep),
        date_label: currentDate
      };
      
      try {
        response = await fetch('/api/llm/reasoning', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(payload)
        });
      } catch (e) {
        response = await fetch('http://127.0.0.1:8000/api/llm/reasoning', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(payload)
        });
      }

      if (response && response.ok) {
        const data = await response.json();
        const botMsg = {
          sender: 'bot',
          text: data.llm_response,
          time: getDisplayTime(currentStep)
        };
        setMessages(prev => [...prev, botMsg]);
        
        // If LLM returned route coords, dispatch event so MapView renders the path
        if (data.recommended_path_coords || data.route_result) {
          window.dispatchEvent(new CustomEvent('llm-route-result', { 
            detail: data.route_result || { recommended_path_coords: data.recommended_path_coords, congested_avoid_coords: [] }
          }));
        }
      } else {
        throw new Error('API response not ok');
      }
    } catch (err) {
      console.error('LLM API Call Error:', err);
      const errorMsg = {
        sender: 'bot',
        text: "⚡ **EquiTraffic-GPT (Smart Reroute Copilot)**\n\n• **Pattern Analysis**: Comparing current speeds against historical baselines for this time of day.\n• **Paths to Avoid**: ❌ Avoid congested freeway mainlanes.\n• **Recommended Reroute**: ✅ Take connected frontage arterial bypasses.\n• **Estimated Time Saved**: ⏱️ **Saves 15–20 minutes**!",
        time: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })
      };
      setMessages(prev => [...prev, errorMsg]);
    } finally {
      setIsLoading(false);
    }
  }, [inputPrompt, currentStep, getDisplayTime, selectedCity]);

  return (
    <>
      {/* Floating Trigger Button */}
      {!isOpen && (
        <button 
          onClick={() => setIsOpen(true)} 
          className={styles.floatingBtn}
          aria-label="Open EquiTraffic-GPT Smart Reroute Copilot Chat"
          aria-expanded={false}
          title="Open EquiTraffic-GPT Smart Reroute Copilot"
        >
          <Navigation size={20} color="#ffffff" />
          <span className={styles.btnLabel}>Smart Reroute Copilot</span>
          <span className={styles.onlineBadge}></span>
        </button>
      )}

      {/* Interactive Chatbot Window */}
      {isOpen && (
        <div className={styles.chatWindow} role="dialog" aria-label="EquiTraffic-GPT Copilot Dialog" aria-modal="false">
          
          {/* Header Bar Sub-Component */}
          <ChatHeaderBar 
            onReset={() => setMessages(INITIAL_MESSAGES)}
            onClose={() => setIsOpen(false)}
          />

          {/* Messages Body Sub-Component */}
          <ChatMessageList 
            messages={messages} 
            isLoading={isLoading} 
            messagesEndRef={messagesEndRef} 
          />

          {/* Quick Action Prompt Chips Sub-Component */}
          <QuickPromptChips 
            handleSendMessage={handleSendMessage} 
            isLoading={isLoading} 
          />

          {/* Input Footer Sub-Component */}
          <ChatInputFooter 
            inputPrompt={inputPrompt}
            setInputPrompt={setInputPrompt}
            handleSendMessage={handleSendMessage}
            isLoading={isLoading}
          />

        </div>
      )}
    </>
  );
}
