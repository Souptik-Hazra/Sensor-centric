import React from 'react';
import styles from '../LlmChatbot.module.css';

const QUICK_PROMPTS = [
  "🚗 Which way to avoid & use if starting now?",
  "🚨 15-Min Historical Pattern Comparison",
  "🏟️ Dodger Stadium Event Reroute",
  "🚧 Road Blockade Egress Route"
];

const QuickPromptChips = ({ handleSendMessage, isLoading }) => {
  return (
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
  );
};

export default QuickPromptChips;
