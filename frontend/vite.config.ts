import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

const allowedHosts = ['localhost', '127.0.0.1', '0.0.0.0', '.ngrok-free.app']

export default defineConfig({
  plugins: [react()],
  server: {
    host: true,
    strictPort: true,
    port: 5173,
    hmr: {
      clientPort: 443
    },
    allowedHosts, // 👈 This tells Vite what hosts are allowed to connect
  }
})