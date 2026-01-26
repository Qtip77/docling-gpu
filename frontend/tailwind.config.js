/** @type {import('tailwindcss').Config} */
export default {
  content: ['./index.html', './src/**/*.{js,ts,jsx,tsx}'],
  theme: {
    extend: {
      fontFamily: {
        sans: ['JetBrains Mono', 'monospace'],
      },
      colors: {
        midnight: '#0a0e14',
        charcoal: '#1a1f2e',
        slate: '#2d3548',
        steel: '#4a5568',
        silver: '#a0aec0',
        pearl: '#e2e8f0',
        azure: '#00d4ff',
        electric: '#7c3aed',
        coral: '#ff6b6b',
        mint: '#10b981',
      },
      typography: {
        invert: {
          css: {
            '--tw-prose-body': '#e2e8f0',
            '--tw-prose-headings': '#e2e8f0',
            '--tw-prose-links': '#00d4ff',
            '--tw-prose-bold': '#e2e8f0',
            '--tw-prose-code': '#00d4ff',
            '--tw-prose-pre-bg': '#0a0e14',
            '--tw-prose-pre-code': '#e2e8f0',
          },
        },
      },
    },
  },
  plugins: [require('@tailwindcss/typography')],
}
