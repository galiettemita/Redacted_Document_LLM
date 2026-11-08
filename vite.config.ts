/// <reference types="vitest/config" />
import react from '@vitejs/plugin-react'
import { defineConfig } from 'vite'

export default defineConfig({
  plugins: [react()],
  resolve: {
    alias: [
      { find: /^pdfjs-dist$/, replacement: 'pdfjs-dist/legacy/build/pdf.mjs' },
      {
        find: /^pdfjs-dist\/web\/pdf_viewer\.mjs$/,
        replacement: 'pdfjs-dist/legacy/web/pdf_viewer.mjs',
      },
    ],
  },
  test: {
    environment: 'jsdom',
    fileParallelism: false,
    globals: true,
    setupFiles: ['./src/test/setup.ts'],
    css: true,
  },
})
