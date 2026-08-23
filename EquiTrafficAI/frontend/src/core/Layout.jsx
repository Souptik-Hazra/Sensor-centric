import React from 'react';
import { Outlet, useLocation } from 'react-router-dom';
import Sidebar from './Sidebar';
import LlmChatbot from './LlmChatbot';
import styles from './Layout.module.css';

const Layout = () => {
  const location = useLocation();
  const isMapPage = location.pathname === '/map';
  
  const getPageTitle = () => {
    switch (location.pathname) {
      case '/': return 'Traffic Monitoring';
      case '/map': return 'Web GIS';
      case '/analytics': return 'Data Analytics';
      case '/settings': return 'System Settings';
      default: return 'Traffic System';
    }
  };

  return (
    <div className={styles.layout}>
      <Sidebar />
      <div className={styles.mainContent}>
        {!isMapPage && (
          <header className={styles.topbar}>
            <h1 className={styles.topbarTitle}>{getPageTitle()}</h1>
          </header>
        )}
        <main className={`${styles.contentArea} ${isMapPage ? styles.mapContentArea : ''}`}>
          <Outlet />
        </main>
      </div>

      {/* High-Impact Interactive EquiTraffic-GPT Chatbot Widget */}
      <LlmChatbot />
    </div>
  );
};

export default Layout;
