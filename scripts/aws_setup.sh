#!/bin/bash
# AWS EC2 Setup Script for IISc Chatbot
# Run this script on a fresh Ubuntu/Amazon Linux instance

set -e

echo "=== IISc Chatbot AWS Deployment Setup ==="

# Update system
echo "Updating system packages..."
sudo apt-get update && sudo apt-get upgrade -y

# Install Docker
echo "Installing Docker..."
sudo apt-get install -y \
    ca-certificates \
    curl \
    gnupg \
    lsb-release

sudo mkdir -p /etc/apt/keyrings
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --dearmor -o /etc/apt/keyrings/docker.gpg

echo \
  "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/ubuntu \
  $(lsb_release -cs) stable" | sudo tee /etc/apt/sources.list.d/docker.list > /dev/null

sudo apt-get update
sudo apt-get install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin

# Add current user to docker group
sudo usermod -aG docker $USER

# Install Docker Compose standalone (if needed)
echo "Installing Docker Compose..."
sudo curl -L "https://github.com/docker/compose/releases/latest/download/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
sudo chmod +x /usr/local/bin/docker-compose

# Install git
echo "Installing Git..."
sudo apt-get install -y git

# Create application directory
echo "Creating application directory..."
sudo mkdir -p /opt/iisc-chatbot
sudo chown $USER:$USER /opt/iisc-chatbot

# Install nginx (optional - for reverse proxy)
echo "Installing Nginx..."
sudo apt-get install -y nginx

# Configure firewall
echo "Configuring firewall..."
sudo ufw allow 22/tcp
sudo ufw allow 80/tcp
sudo ufw allow 443/tcp
sudo ufw allow 8000/tcp
sudo ufw allow 8501/tcp
sudo ufw --force enable

echo "=== Setup complete! ==="
echo "Next steps:"
echo "1. Log out and log back in for docker group to take effect"
echo "2. Clone your repository to /opt/iisc-chatbot"
echo "3. Run deploy.sh to start the application"
