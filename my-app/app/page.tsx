import SearchBar from "./Components/search_bar";
import Navbar from "./Components/navbar";
import styles from './styles.module.css';

export default function Home() {
  return (
    <div className={styles.container}>
      <Navbar />
      <div className={styles.card}>
        <h1 className={styles.header}>Welcome to Our App</h1>
        <p>Use the search form below to describe the kind of image you would like to find.</p>
        <ul>
          <li>Keep descriptions short and non-offensive.</li>
          <li>No illegal or harmful content.</li>
          <li>Every query is logged on the Queries page.</li>
        </ul>
      </div>
      <SearchBar />
    </div>
  )
}
