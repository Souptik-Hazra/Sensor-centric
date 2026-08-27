import React from 'react';
import { Send } from 'lucide-react';
import styles from '../LlmChatbot.module.css';

const ChatInputFooter = ({
  inputPrompt,
  setInputPrompt,
  handleSendMessage,
  isLoading
}) => {
  return (
    <div className={styles.chatFooter}>
      <input 
        type="text" 
        placeholder="Ask which way to avoid or reroute if starting now..."
        value={inputPrompt}
        onChange={(e) => setInputPrompt(e.target.value)}
        onKeyDown={(e) => e.key === 'Enter' && handleSendMessage()}
        className={styles.chatInput}
        disabled={isLoading}
        aria-label="Ask EquiTraffic-GPT a traffic question"
      />
      <button 
        onClick={() => handleSendMessage()}
        className={styles.sendBtn}
        disabled={isLoading || !inputPrompt.trim()}
        aria-label="Send Message to EquiTraffic-GPT"
      >
        <Send size={15} />
      </button>
    </div>
  );
};

export default React.memo(ChatInputFooter);
