import { copyFile, mkdir, readFile, writeFile } from "node:fs/promises";

const assets = [
  "store.js", "store.css", "coastal-terrace.jpg",
  "city-expo.jpg", "city-fifa.jpg", "city-icc.jpg",
  "photo-expo.jpg", "photo-fifa.jpg", "photo-icc.jpg",
];

const bundle = process.argv.includes("--vercel");
const stage = "runs/af-stays-vercel";
const destination = bundle ? stage + "/public" : "dist";
await mkdir(destination + "/assets", { recursive: true });
await Promise.all([
  copyFile("website/index.html", destination + "/index.html"),
  ...assets.map((file) => copyFile("website/" + file, destination + "/assets/" + file)),
]);
if (bundle) {
  const config = JSON.parse(await readFile("vercel.json", "utf8"));
  delete config.buildCommand;
  config.outputDirectory = "public";
  await mkdir(stage + "/api/storefront", { recursive: true });
  await Promise.all([
    copyFile("api/storefront/route.mjs", stage + "/api/storefront/route.mjs"),
    writeFile(stage + "/vercel.json", JSON.stringify(config, null, 2) + "\n"),
  ]);
}
