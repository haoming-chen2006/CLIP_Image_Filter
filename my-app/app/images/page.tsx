"use client";
import React from 'react';
import Image from 'next/image';
import Navbar from "../Components/navbar";
import styles from '../styles.module.css';
import SearchBar from '../Components/search_bar';

const ImagesPage = () => {
  // Local images array
  const localImages = [
    '00000319.JPG', '00000320.JPG', '00000322.JPG', '00000323.JPG', '00000324.JPG',
    '00000325.JPG', '00000327.JPG', '00000338.JPG', '00000343.JPG', '00000356.JPG',
    '00000359.JPG', '00000368.JPG', '00000370.JPG', '00000374.JPG', '00000375.JPG',
    '00000382.JPG', '00000383.JPG', '00000386.JPG', '00000387.JPG', '00000388.JPG',
    'IMG_4417.JPG', 'IMG_4453.JPG', 'IMG_4455.JPG', 'IMG_4463.JPG', 'IMG_4464.CR3'
  ].filter(img => img.endsWith('.JPG')); // Only show JPG files

  return (
    <div className={styles.container}>
      <Navbar />
      <SearchBar />
      
      {/* Image Gallery Section */}
      <section>
        <h1 className={styles.header}>Local Image Gallery</h1>
        <div className={styles.grid}>
          {localImages.map((imageName, index) => (
            <div key={index} className={styles.imageCard}>
              <Image
                src={`/images/${imageName}`}
                alt={`Gallery image ${imageName}`}
                width={250}
                height={200}
                className={styles.image}
              />
              <div className={styles.imageTitle}>
                {imageName}
              </div>
            </div>
          ))}
        </div>
      </section>
    </div>
  )
}

export default ImagesPage


