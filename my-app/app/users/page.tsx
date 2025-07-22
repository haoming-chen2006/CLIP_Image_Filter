"use client";
import React, { useState, useEffect } from 'react'
import Navbar from "../Components/navbar";
import styles from '../styles.module.css';

interface User {
    userId: number;
    id: number;
    title: string;
    completed: boolean;
}

const UsersPage = () => {
  const [userlist, setUsers] = useState<User[]>([]);

  useEffect(() => {
    changeUsers();
  }, [])

  const changeUsers = async() => {
    const res = await fetch("https://jsonplaceholder.typicode.com/users/1/todos");
    const users: User[] = await res.json();
    setUsers(users);
  }
  
  const handleClick = () => {
    changeUsers();
  }

  return (
    <div className={styles.container}>
      <Navbar />
      <div className={styles.card}>
        <h1 className={styles.header}>Users Page</h1>
        <p>Welcome to the users page! Manage and view user information below.</p>
        
        <div className={styles.grid}>
          <div className={styles.card}>
            <h3>User Management</h3>
            <p>Manage user accounts and permissions</p>
          </div>
          <div className={styles.card}>
            <h3>User Analytics</h3>
            <p>View user engagement and statistics</p>
          </div>
          <div className={styles.card}>
            <h3>User Settings</h3>
            <p>Configure user preferences and settings</p>
          </div>
        </div>
      </div>

      {/* User Tasks Section */}
      <section>
        <h2 className={styles.header}>User Task List</h2>
        <button onClick={handleClick} className={styles.button}>
          Refresh User List
        </button>
        
        <div>
          {userlist.map(user => (
            <div key={user.id} className={styles.userItem}>
              <div className={styles.userTitle}>{user.title}</div>
              <div className={styles.userMeta}>
                <span>ID: {user.id} | User ID: {user.userId}</span>
                <span 
                  className={`${styles.status} ${user.completed ? styles.completed : styles.pending}`}
                >
                  {user.completed ? 'Completed' : 'Pending'}
                </span>
              </div>
            </div>
          ))}
        </div>
      </section>
    </div>
  )
}

export default UsersPage
