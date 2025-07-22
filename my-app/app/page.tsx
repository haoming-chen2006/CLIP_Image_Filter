import Link from "next/link";
import SearchBar from "./Components/search_bar";
import Navbar from "./Components/navbar";
import styles from './styles.module.css';

export default function Home() {
  return (
    <div className={styles.container}>
      <Navbar />
      <div className={styles.card}>
        <h1 className={styles.header}>Welcome to Our App</h1>
        <p>Explore our features using the search bar below or navigate through the menu.</p>
      </div>
    </div>
  )
}
