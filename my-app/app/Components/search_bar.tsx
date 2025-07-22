"use client";

import React, { useState, useEffect } from 'react'
import styles from '../styles.module.css';

const SearchBar = () => {
    const [info, setInfo] = useState(
        {
            "subject": '',
            "color": '',
            "other": '',
        }
    )
    const [display, setdisplay] = useState(false);

    // Define form fields configuration
    const formFields = [
        { key: 'subject', label: 'Subject', placeholder: 'Enter subject' },
        { key: 'color', label: 'Color', placeholder: 'Enter color' },
        { key: 'other', label: 'Other', placeholder: 'Enter other info' }
    ];

    // Real-time monitoring of info changes
    useEffect(() => {
        console.log('Info updated in real-time:', info);
    }, [info]);

    // Handle input changes
    const handleInputChange = (field: string, value: string) => {
        setInfo(prev => ({
            ...prev,
            [field]: value
        }));
    }

    // Handle form submission
    const handleSubmit = (e: React.FormEvent) => {
        e.preventDefault();
        setdisplay(true);
        // You can add your submission logic here
    }

    const displayCard = () => {
        return (
            <div>
                <h3>Submitted Information</h3>
                {formFields.map(field => (
                    <div key={field.key}>
                        <strong>{field.label}:</strong> {info[field.key as keyof typeof info] || 'Not provided'}
                    </div>
                ))}
            </div>
        )
    }
    const handleReset = () => {
        // Reset using the same structure as initial state
        const resetInfo = {
            subject: '',
            color: '',
            other: ''
        };
        
        setInfo(resetInfo);
        setdisplay(false);
    }

    const reset = () => {
        return (
            <button onClick={handleReset}>click for reset</button>
        );
    }
  return (
    <div className={styles.card}>
      <h2>Search Information</h2>
      <form onSubmit={handleSubmit}>
        {formFields.map(field => (
          <input
            key={field.key}
            type="text"
            placeholder={field.placeholder}
            value={info[field.key as keyof typeof info]}
            onChange={(e) => handleInputChange(field.key, e.target.value)}
            className={styles.input}
          />
        ))}
        <button type="submit" className={styles.button}>Submit</button>
      </form> 
      {display && (
        <div className={styles.card}>
          {displayCard()}
          {reset()}
        </div>
      )}
    </div>
  )
}

export default SearchBar
