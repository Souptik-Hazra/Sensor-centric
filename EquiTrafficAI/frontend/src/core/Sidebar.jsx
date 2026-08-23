import React, { useState } from 'react';
import { NavLink } from 'react-router-dom';
import { Map, Activity, BarChart2, Settings, TrafficCone, ChevronLeft, ChevronRight } from 'lucide-react';
import styles from './Sidebar.module.css';

const Sidebar = () => {
  const [isCollapsed, setIsCollapsed] = useState(false);

  const navItems = [
    { path: '/', icon: <Activity size={20} />, label: 'Monitoring' },
    { path: '/map', icon: <Map size={20} />, label: 'Web GIS' },
    { path: '/analytics', icon: <BarChart2 size={20} />, label: 'Analytics' },
    { path: '/settings', icon: <Settings size={20} />, label: 'Settings' },
  ];

  return (
    <aside className={`${styles.sidebar} ${isCollapsed ? styles.collapsed : ''}`}>
      <div className={styles.brand}>
        <TrafficCone className={styles.brandIcon} size={26} />
        {!isCollapsed && <span className={styles.brandText}>TrafficOS</span>}
      </div>

      <nav className={styles.nav}>
        {navItems.map((item) => (
          <NavLink
            key={item.path}
            to={item.path}
            title={isCollapsed ? item.label : ''}
            className={({ isActive }) =>
              isActive ? `${styles.navItem} ${styles.navItemActive}` : styles.navItem
            }
          >
            {item.icon}
            {!isCollapsed && <span>{item.label}</span>}
          </NavLink>
        ))}
      </nav>

      {/* Collapse Toggle Button */}
      <button 
        onClick={() => setIsCollapsed(!isCollapsed)}
        className={styles.toggleBtn}
        title={isCollapsed ? "Expand Sidebar" : "Collapse Sidebar"}
      >
        {isCollapsed ? <ChevronRight size={18} /> : <ChevronLeft size={18} />}
        {!isCollapsed && <span style={{ fontSize: '11px', fontWeight: 'bold' }}>Minimize Sidebar</span>}
      </button>
    </aside>
  );
};

export default Sidebar;
