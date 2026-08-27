import React from 'react';
import { Navigation, RefreshCw, X } from 'lucide-react';
import styles from '../LlmChatbot.module.css';

const ChatHeaderBar = ({ onReset, onClose }) => {
  return (
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
          onClick={onReset} 
          className={styles.headerBtn}
          aria-label="Reset Conversation History"
          title="Reset Conversation"
        >
          <RefreshCw size={14} />
        </button>
        <button 
          onClick={onClose} 
          className={styles.headerBtn}
          aria-label="Minimize Chatbot Window"
          title="Minimize Chatbot"
        >
          <X size={16} />
        </button>
      </div>
    </div>
  );
};

export default React.memo(ChatHeaderBar);
