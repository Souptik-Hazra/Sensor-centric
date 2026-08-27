import React from 'react';
import { Bot, AlertTriangle } from 'lucide-react';
import styles from '../LlmChatbot.module.css';

const ChatMessageList = ({ messages, isLoading, messagesEndRef }) => {
  return (
    <div className={styles.messagesList}>
      {messages.map((msg, index) => (
        <div 
          key={index} 
          className={`${styles.messageWrapper} ${msg.sender === 'user' ? styles.userMsgWrapper : styles.botMsgWrapper}`}
        >
          <div className={`${styles.messageBubble} ${msg.sender === 'user' ? styles.userBubble : styles.botBubble} ${msg.isAutoAlert ? styles.autoAlertBubble : ''}`}>
            {msg.sender === 'bot' && (
              <div className={styles.botAvatarHeader}>
                {msg.isAutoAlert ? <AlertTriangle size={14} color="#ef4444" /> : <Bot size={14} color="#38bdf8" />}
                <span className={styles.botName}>
                  {msg.isAutoAlert ? '🚨 15-Min Proactive Bottleneck Alert' : 'EquiTraffic-GPT Copilot'}
                </span>
                <span className={styles.msgTime}>{msg.time}</span>
              </div>
            )}
            <div className={`${styles.msgContent} ui-whitespace-preline`}>
              {msg.text}
            </div>
            {msg.sender === 'user' && (
              <div className={styles.userTime}>{msg.time}</div>
            )}
          </div>
        </div>
      ))}

      {isLoading && (
        <div className={`${styles.messageWrapper} ${styles.botMsgWrapper}`}>
          <div className={`${styles.messageBubble} ${styles.botBubble}`}>
            <div className={styles.botAvatarHeader}>
              <Bot size={14} color="#38bdf8" />
              <span className={styles.botName}>EquiTraffic-GPT Copilot</span>
            </div>
            <div className={styles.loadingDots}>
              <span>.</span><span>.</span><span>.</span>
            </div>
          </div>
        </div>
      )}
      <div ref={messagesEndRef} />
    </div>
  );
};

export default ChatMessageList;
