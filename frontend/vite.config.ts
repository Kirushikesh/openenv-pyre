import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import path from 'path'

// https://vite.dev/config/
export default defineConfig({
  plugins: [react()],
  build: {
    outDir: path.resolve(__dirname, '../server/static'),
    emptyOutDir: true,
  },
  server: {
    proxy: {
      '/state': 'http://127.0.0.1:8000',
      '/reset': 'http://127.0.0.1:8000',
      '/step': 'http://127.0.0.1:8000',
      '/scene': 'http://127.0.0.1:8000',
    }
  }
})
