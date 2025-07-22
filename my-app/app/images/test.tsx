import React from 'react'

const test = () => {
    const fs = require("fs");
    const path = require("path");
    const imagesDir = path.join(__dirname, "public", "images");
    console.log(imagesDir)
  return (
    <div>
      Hi
    </div>
  )
}

export default test
