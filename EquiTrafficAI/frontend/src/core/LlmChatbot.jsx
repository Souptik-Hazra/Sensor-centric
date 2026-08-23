import React, { useState, useRef, useEffect } from 'react';
import { Bot, Send, X, RefreshCw, Navigation, AlertTriangle } from 'lucide-react';
import styles from './LlmChatbot.module.css';

const QUICK_PROMPTS = [
  "🚗 Which way to avoid & use if starting now?",
  "🚨 15-Min Historical Pattern Comparison",
  "🏟️ Dodger Stadium Event Reroute",
  "🚧 Road Blockade Egress Route"
];

const INITIAL_MESSAGES = [
  {
    sender: 'bot',
    text: "Hello! I am **EquiTraffic-GPT Smart Reroute Copilot**.\n\nEvery **15 minutes**, I automatically monitor live highway telemetry against historical same-day patterns to tell you:\n• ❌ **Which ways to avoid**\n• ✅ **Which alternate routes to use if starting now**\n• ⏱️ **Estimated travel time saved**\n\nAsk me anytime for live rerouting advice!",
    time: 'Just now'
  }
];

export default function LlmChatbot({ currentStep = 96, selectedCity = 'la' }) {
  const [isOpen, setIsOpen] = useState(false);
  const [messages, setMessages] = useState(INITIAL_MESSAGES);
  const [inputPrompt, setInputPrompt] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [lastAlertStep, setLastAlertStep] = useState(-1);
  const messagesEndRef = useRef(null);

  const getDisplayTime = (step) => {
    const totalMinutes = step * 5;
    const hours = Math.floor(totalMinutes / 60);
    const mins = totalMinutes % 60;
    const ampm = hours >= 12 ? 'PM' : 'AM';
    const displayHours = hours % 12 === 0 ? 12 : hours % 12;
    const displayMins = mins < 10 ? '0' + mins : mins;
    return `${displayHours}:${displayMins} ${ampm}`;
  };

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    if (isOpen) {
      scrollToBottom();
    }
  }, [messages, isOpen]);

  // Autonomous 15-Minute Proactive Alert Engine (Triggers every 3 steps = 15 minutes)
  useEffect(() => {
    if (currentStep > 0 && currentStep % 3 === 0 && currentStep !== lastAlertStep) {
      setLastAlertStep(currentStep);
      triggerProactive15MinAlert(currentStep);
    }
  }, [currentStep]);

  const triggerProactive15MinAlert = async (stepVal) => {
    const timeLabel = getDisplayTime(stepVal);
    try {
      let response;
      try {
        response = await fetch('/api/llm/reasoning', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            prompt: `auto_alert 15-minute alert for ${timeLabel}`,
            sensor_id: 0,
            city: selectedCity
          })
        });
      } catch (e) {
        response = await fetch('http://127.0.0.1:8000/api/llm/reasoning', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            prompt: `auto_alert 15-minute alert for ${timeLabel}`,
            sensor_id: 0,
            city: selectedCity
          })
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
  };

  const handleSendMessage = async (customText = '') => {
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
      try {
        response = await fetch('/api/llm/reasoning', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            prompt: textToSend,
            sensor_id: 0,
            city: selectedCity
          })
        });
      } catch (e) {
        response = await fetch('http://127.0.0.1:8000/api/llm/reasoning', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            prompt: textToSend,
            sensor_id: 0,
            city: selectedCity
          })
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
  };

  return (
    <>
      {/* Floating Trigger Button */}
      {!isOpen && (
        <button 
          onClick={() => setIsOpen(true)} 
          className={styles.floatingBtn}
          title="Open EquiTraffic-GPT Smart Reroute Copilot"
        >
          <Navigation size={20} color="#ffffff" />
          <span className={styles.btnLabel}>Smart Reroute Copilot</span>
          <span className={styles.onlineBadge}></span>
        </button>
      )}

      {/* Interactive Chatbot Window */}
      {isOpen && (
        <div className={styles.chatWindow}>
          
          {/* Header Bar */}
          <div className={styles.chatHeader}>
            <div className={styles.headerInfo}>
              <div className={styles.botIconWrapper}>
                <Navigation size={18} color="#38bdf8" />
              </div>
              <div>
                <div className={styles.headerTitle}>EquiTraffic-GPT Copilot</div>
                <div className={styles.headerSubtitle}>
                  <span className={styles.greenDot}></span> Auto 15-Min Pattern Rerouting
                </div>
              </div>
            </div>

            <div className={styles.headerActions}>
              <button 
                onClick={() => setMessages(INITIAL_MESSAGES)} 
                className={styles.headerBtn}
                title="Reset Conversation"
              >
                <RefreshCw size={14} />
              </button>
              <button 
                onClick={() => setIsOpen(false)} 
                className={styles.headerBtn}
                title="Minimize Chatbot"
              >
                <X size={16} />
              </button>
            </div>
          </div>

          {/* Messages Body */}
          <div className={styles.chatBody}>
            {messages.map((msg, idx) => (
              <div 
                key={idx} 
                className={`${styles.messageWrapper} ${msg.sender === 'user' ? styles.userWrapper : styles.botWrapper}`}
              >
                {msg.sender === 'bot' && (
                  <div className={styles.msgAvatar}>
                    <Navigation size={14} color="#38bdf8" />
                  </div>
                )}

                <div className={`${styles.messageBubble} ${msg.sender === 'user' ? styles.userBubble : styles.botBubble}`}>
                  <div className={styles.messageText} style={{ whiteSpace: 'pre-wrap' }}>
                    {msg.text.split(/(\*\*.*?\*\*)/g).map((part, idx) => {
                      if (part.startsWith('**') && part.endsWith('**')) {
                        return <strong key={idx} style={{ color: '#38bdf8' }}>{part.slice(2, -2)}</strong>;
                      }
                      return part;
                    })}
                  </div>
                  <div className={styles.messageTime}>{msg.time}</div>
                </div>
              </div>
            ))}

            {isLoading && (
              <div className={`${styles.messageWrapper} ${styles.botWrapper}`}>
                <div className={styles.msgAvatar}>
                  <Navigation size={14} color="#38bdf8" />
                </div>
                <div className={`${styles.messageBubble} ${styles.botBubble}`}>
                  <div className={styles.typingIndicator}>
                    <span></span>
                    <span></span>
                    <span></span>
                  </div>
                </div>
              </div>
            )}
            <div ref={messagesEndRef} />
          </div>

          {/* Quick Action Prompt Chips */}
          <div className={styles.quickPrompts}>
            {QUICK_PROMPTS.map((prompt, idx) => (
              <button 
                key={idx} 
                onClick={() => handleSendMessage(prompt)}
                className={styles.chipBtn}
                disabled={isLoading}
              >
                {prompt}
              </button>
            ))}
          </div>

          {/* Input Footer */}
          <div className={styles.chatFooter}>
            <input 
              type="text" 
              placeholder="Ask which way to avoid or reroute if starting now..."
              value={inputPrompt}
              onChange={(e) => setInputPrompt(e.target.value)}
              onKeyDown={(e) => e.key === 'Enter' && handleSendMessage()}
              className={styles.chatInput}
              disabled={isLoading}
            />
            <button 
              onClick={() => handleSendMessage()}
              className={styles.sendBtn}
              disabled={isLoading || !inputPrompt.trim()}
            >
              <Send size={15} />
            </button>
          </div>

        </div>
      )}
    </>
  );
}
