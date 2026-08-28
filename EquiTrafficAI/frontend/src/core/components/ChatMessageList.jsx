import React from 'react';
import { Bot, AlertTriangle } from 'lucide-react';
import styles from '../LlmChatbot.module.css';

const parseMarkdown = (text) => {
  if (!text) return '';
  let html = text;
  // Headers
  html = html.replace(/^### (.*$)/gim, '<h3 style="color: #00f2fe; margin-top: 10px; margin-bottom: 5px;">$1</h3>');
  html = html.replace(/^## (.*$)/gim, '<h2 style="color: #00f2fe; margin-top: 10px; margin-bottom: 5px;">$1</h2>');
  html = html.replace(/^# (.*$)/gim, '<h1 style="color: #00f2fe; margin-top: 10px; margin-bottom: 5px;">$1</h1>');
  // Bold
  html = html.replace(/\*\*(.*?)\*\*/g, '<strong style="color: #fff;">$1</strong>');
  // Italic
  html = html.replace(/\*(.*?)\*/g, '<em>$1</em>');
  // Bullets
  html = html.replace(/^[\*-]\s+(.*$)/gim, '• $1');
  return html;
};

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
            <div 
              className={`${styles.msgContent} ui-whitespace-preline`}
              dangerouslySetInnerHTML={{ __html: parseMarkdown(msg.text) }}
            />
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
