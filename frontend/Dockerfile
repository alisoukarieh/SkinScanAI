# Basse image for Node.js
FROM node:14

# Create app directory
WORKDIR /src/app

# Copy package.json and package-lock.json
COPY package*.json ./

# Install dependencies
RUN npm install

# Copy all files
COPY . .

# Expose port 3000
EXPOSE 3000

# Start the app
CMD ["npm", "run", "start"]