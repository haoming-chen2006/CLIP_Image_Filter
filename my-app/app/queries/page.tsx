"use client";
import React, { useState } from 'react'
import Navbar from "../Components/navbar";
import SearchBar from "../Components/search_bar";
import styles from '../styles.module.css';

interface QueryLog {
  query: string;
  status: 'success' | 'error';
  message: string;
}

const QueryPage = () => {
  const [logs, setLogs] = useState<QueryLog[]>([]);

  const handleSearch = async (q: string) => {
    try {
      const res = await fetch(`/api/search?q=${encodeURIComponent(q)}`);
      const data = await res.json();
      if (res.ok && data.image) {
        setLogs(prev => [...prev, { query: q, status: 'success', message: data.image }]);
      } else {
        setLogs(prev => [...prev, { query: q, status: 'error', message: data.error || 'Unknown error' }]);
      }
    } catch (err) {
      setLogs(prev => [...prev, { query: q, status: 'error', message: 'Request failed' }]);
    }
  };

  return (
    <div className={styles.container}>
      <Navbar />
      <div className={styles.card}>
        <h1 className={styles.header}>Query History</h1>
        <SearchBar onSearch={handleSearch} />
        <ul>
          {logs.map((log, idx) => (
            <li key={idx} className={styles.userItem}>
              <div className={styles.userTitle}>{log.query}</div>
              <div className={styles.userMeta}>
                <span className={`${styles.status} ${log.status === 'success' ? styles.completed : styles.pending}`}>{log.status}</span>
                <span> {log.message}</span>
              </div>
            </li>
          ))}
        </ul>
      </div>
    </div>
  )
}

export default QueryPage;
