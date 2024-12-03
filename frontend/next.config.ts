/** @type {import('next').NextConfig} */
const nextConfig = {
  reactStrictMode: true, // Existing configuration
  experimental: {
    serverActions: {
      bodySizeLimit: '50mb', // Allow request body up to 50 MB
    },
  },
  // Any other existing configurations can remain unchanged
};

module.exports = nextConfig;
