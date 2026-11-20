import { copyFile, mkdir } from "node:fs/promises";

const assets = [
  "store.js", "store.css", "coastal-terrace.jpg",
  "city-expo.jpg", "city-fifa.jpg", "city-icc.jpg",
  "photo-expo.jpg", "photo-fifa.jpg", "photo-icc.jpg",
];

await mkdir("dist/assets", { recursive: true });
await Promise.all([
  copyFile("website/index.html", "dist/index.html"),
  ...assets.map((file) => copyFile("website/" + file, "dist/assets/" + file)),
]);
