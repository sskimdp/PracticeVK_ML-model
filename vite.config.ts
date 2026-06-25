import { defineConfig } from 'vite';

// base: './' — относительные пути, чтобы сборка работала на любом
// статическом хостинге (GitHub Pages в подпапке, Netlify, Vercel).
export default defineConfig({
  base: './',
  worker: {
    format: 'es',
  },
  build: {
    target: 'es2022',
    sourcemap: false,
  },
});
