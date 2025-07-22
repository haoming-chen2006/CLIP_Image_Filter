"use client";
import styles from "./bar.module.css";
import { useState } from "react";
import Link from "next/link";

import React from 'react'




const Navbar = () => {
    const links = [
        {name: "home", url:"/"},
        {name: "images", url: "/images"},
        {name: "queries", url: "/queries"},
        {name: "configs", url: "/configs"},
    ]
  return (
    <div className = {styles.card}>
      {links.map(link => (
        <div key={link.name}>
            <Link href={link.url}>{link.name}</Link>
        </div>
      ))}
    </div>
  )
}

export default Navbar;
