import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import tailwindcss from '@tailwindcss/vite'

// https://vitejs.dev/config/
export default defineConfig({
  plugins: [react(), tailwindcss()],
  server: {
    proxy: {
      '/predict': 'http://localhost:8000',
      '/samples': 'http://localhost:8000',
      '/predict_sample': 'http://localhost:8000'
    }
  }
})
