"use client";
import React from 'react';
import Image from 'next/image';
import Navbar from "../Components/navbar";
import styles from '../styles.module.css';
import SearchBar from '../Components/search_bar';

const ImagesPage = () => {
  // Load image paths from file (equivalent to Python code)
  const loadImagePaths = async (filePath: string): Promise<string[]> => {
    try {
      const response = await fetch(filePath);
      const text = await response.text();
      const imagePaths = text
        .split('\n')
        .map(line => line.trim())
        .filter(line => line.length > 0);
      
      console.log(imagePaths);
      return imagePaths;
    } catch (error) {
      console.error('Error loading image paths:', error);
      return [];
    }
  };
  const [imagePaths, setImagePaths] = React.useState<string[]>([]);
  const [isLoading, setIsLoading] = React.useState<boolean>(true);
  const [searchedImage, setSearchedImage] = React.useState<string>('00000319.JPG');
  const [searchQuery, setSearchQuery] = React.useState<string>('');
  
  // Function to simulate search (you can replace this with actual CLIP inference later)
  const handleSearch = async (query: string) => {
    setSearchQuery(query);
    try {
      const res = await fetch(`/api/search?q=${encodeURIComponent(query)}`);
      const data = await res.json();
      if (data.image) {
        setSearchedImage(data.image);
      }
    } catch (err) {
      console.error('Search request failed', err);
    }
  };
  
  const fallbackImages = [
    '00000319.JPG', '00000320.JPG', '00000322.JPG', '00000323.JPG', '00000324.JPG',
    '00000325.JPG', '00000327.JPG', '00000338.JPG', '00000343.JPG', '00000356.JPG',
    '00000359.JPG', '00000368.JPG', '00000370.JPG', '00000374.JPG', '00000375.JPG',
    '00000382.JPG', '00000383.JPG', '00000386.JPG', '00000387.JPG', '00000388.JPG',
    'IMG_4417.JPG', 'IMG_4453.JPG', 'IMG_4455.JPG', 'IMG_4463.JPG'
  ].filter(img => img.endsWith('.JPG'));

  React.useEffect(() => {
    // initialize backend vector store
    fetch('/api/init').catch(err => console.error('Init failed', err));
    loadImagePaths('/image_paths.csv')
      .then(paths => {
        if (paths.length > 0) {
          const filenames = paths
            .map(p => p.split('/').pop() || '')
            .filter(name => name.endsWith('.JPG'));
          if (filenames.length > 0) {
            setImagePaths(filenames);
            return;
          }
        }
        setImagePaths(fallbackImages);
      })
      .catch((error) => {
        console.error('Failed to load CSV, using fallback images:', error);
        setImagePaths(fallbackImages);
      })
      .finally(() => {
        setIsLoading(false);
      });
  }, []);

  // Always use imagePaths (which starts with fallback and may be updated with CSV data)
  const displayImages = imagePaths;

  return (
    <div className={styles.container}>
      <Navbar />
      <SearchBar onSearch={handleSearch} />
      
      {/* Search Result Section */}
      <section>
        <h2 className={styles.header}>Best Match</h2>
        <div className={styles.searchResultContainer}>
          <div className={styles.searchResultCard}>
            <Image
              src={`/images/${searchedImage}`}
              alt={`Search result: ${searchedImage}`}
              width={400}
              height={300}
              className={styles.searchResultImage}
              onError={(e) => {
                console.log(`Failed to load search result image: ${searchedImage}`);
              }}
            />
            <div className={styles.searchResultInfo}>
              <h3>Best match for: "{searchQuery || 'Initial image'}"</h3>
              <p className={styles.searchResultTitle}>{searchedImage}</p>
              <p className={styles.confidenceScore}>Confidence: 95%</p>
            </div>
          </div>
        </div>
      </section>
      
      {/* Image Gallery Section */}
      <section>
        <h1 className={styles.header}>Local Image Gallery</h1>
        {isLoading && <p>Loading images...</p>}
        <p>Displaying {displayImages.length} images</p>
        <div className={styles.grid}>
          {displayImages.map((imageName: string, index: number) => (
            <div key={index} className={styles.imageCard}>
              <Image
                src={`/images/${imageName}`}
                alt={`Gallery image ${imageName}`}
                width={250}
                height={200}
                className={styles.image}
                onError={(e) => {
                  console.log(`Failed to load image: ${imageName}`);
                }}
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


