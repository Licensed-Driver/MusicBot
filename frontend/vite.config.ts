import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

const allowedHosts = ['localhost', '127.0.0.1', '0.0.0.0', '.ngrok-free.app', '374c-70-74-152-126.ngrok-free.app']

export default defineConfig({
  plugins: [react()],
  server: {
    proxy:{
      // Creates a proxy for any vite fetches that start with the specified proxy string
      // Takes any address that starts it's direct with api and replaces anything before the "api" part to just be the local api redicrection
      "/api": {
        target: "https://41e6-70-74-152-126.ngrok-free.app",
        changeOrigin: true,
        rewrite: (path) => path.replace(/^\/api/, "")
      },
      "/front": {
        target: "https://ec0d-70-74-152-126.ngrok-free.app",
        changeOrigin: true,
        rewrite: (path) => path.replace(/^\/front/, "")
      },
    },
    host: true,
    strictPort: true,
    port: 5173,
    hmr: {
      clientPort: 443
    },
    allowedHosts, // 👈 This tells Vite what hosts are allowed to connect
  }
})