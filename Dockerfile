# Use the official Python image
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Copy the application files
COPY . .

# Expose the port
EXPOSE 8080

# Command to run the application
CMD ["python", "app.py"]
