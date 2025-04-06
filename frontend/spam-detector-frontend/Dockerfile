# Step 1: Build the React app
FROM node:16-slim as build

# Set the working directory inside the container
WORKDIR /app

# Copy the package.json and package-lock.json (if available)
COPY frontend/spam-detector-frontend/package*.json ./

# Install dependencies
RUN npm install

# Copy the rest of the app's source code
COPY frontend/spam-detector-frontend/ .

# Build the React app
RUN npm run build

# Step 2: Serve the app with nginx
FROM nginx:alpine

# Copy the built React app from the previous build step
COPY --from=build /app/build /usr/share/nginx/html

# Expose port 80 (default for nginx)
EXPOSE 80

# Start nginx
CMD ["nginx", "-g", "daemon off;"]
