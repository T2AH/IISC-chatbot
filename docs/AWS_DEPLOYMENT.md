# IISc Chatbot - AWS Deployment Guide

## Prerequisites

- AWS Account
- Domain name (optional, for HTTPS)
- Gemini API Key

## Deployment Options

### Option 1: AWS EC2 (Recommended for getting started)

#### 1. Launch EC2 Instance

1. **Choose AMI**: Ubuntu Server 22.04 LTS or Amazon Linux 2023
2. **Instance Type**: 
   - Minimum: t3.medium (2 vCPU, 4 GB RAM)
   - Recommended: t3.large (2 vCPU, 8 GB RAM) for better performance
3. **Storage**: 30 GB minimum (for application + index data)
4. **Security Group**: Open ports:
   - 22 (SSH)
   - 80 (HTTP)
   - 443 (HTTPS)
   - 8000 (API - optional, can be closed if using nginx)
   - 8501 (UI - optional, can be closed if using nginx)

#### 2. Connect to Instance

```bash
ssh -i your-key.pem ubuntu@your-instance-ip
```

#### 3. Run Setup Script

```bash
# Clone repository
cd /opt
sudo git clone https://github.com/YOUR_USERNAME/IISC-chatbot.git iisc-chatbot
cd iisc-chatbot

# Make scripts executable
chmod +x scripts/*.sh

# Run setup
./scripts/aws_setup.sh

# Log out and back in for docker group to take effect
exit
# ssh back in
```

#### 4. Configure Environment

```bash
cd /opt/iisc-chatbot

# Create .env file
cp ENV_PLACEHOLDER .env
nano .env

# Add your configuration:
# GEMINI_API_KEY=your_actual_api_key_here
# INDEX_DIR=data/index/fastembed_bge_small_iisc_2025nov07
```

#### 5. Deploy Application

```bash
# Make deploy script executable
chmod +x scripts/deploy.sh

# Deploy
./scripts/deploy.sh
```

#### 6. Access Application

- API: `http://your-instance-ip:8000`
- UI: `http://your-instance-ip:8501`
- Health: `http://your-instance-ip:8000/health`

#### 7. Setup Auto-Start (Optional)

```bash
# Install systemd service
sudo cp scripts/iisc-chatbot.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable iisc-chatbot
sudo systemctl start iisc-chatbot

# Check status
sudo systemctl status iisc-chatbot
```

#### 8. Setup Nginx Reverse Proxy (Optional, for production)

```bash
# Install certbot for SSL
sudo apt-get install -y certbot python3-certbot-nginx

# Copy nginx config
sudo cp scripts/nginx.conf /etc/nginx/sites-available/iisc-chatbot

# Edit config with your domain
sudo nano /etc/nginx/sites-available/iisc-chatbot

# Enable site
sudo ln -s /etc/nginx/sites-available/iisc-chatbot /etc/nginx/sites-enabled/
sudo nginx -t
sudo systemctl restart nginx

# Get SSL certificate
sudo certbot --nginx -d your-domain.com
```

### Option 2: AWS ECS (Elastic Container Service)

For scalable production deployment:

1. Push Docker image to ECR
2. Create ECS Task Definition
3. Create ECS Service with ALB
4. Configure Auto Scaling

(Detailed ECS setup guide coming soon)

### Option 3: AWS Lambda + API Gateway

For serverless deployment (requires code modifications for cold start optimization).

## Monitoring

### View Logs

```bash
# Docker logs
docker-compose -f docker-compose.prod.yml logs -f

# Application logs
tail -f logs/api.log
tail -f logs/ui.log
```

### Health Checks

```bash
# API health
curl http://localhost:8000/health

# Check running containers
docker ps
```

## Updating the Application

```bash
cd /opt/iisc-chatbot
git pull origin main
./scripts/deploy.sh
```

## Troubleshooting

### Container won't start

```bash
# Check logs
docker-compose -f docker-compose.prod.yml logs

# Check resources
docker stats
free -h
df -h
```

### API not responding

```bash
# Check if port is listening
sudo netstat -tulpn | grep 8000

# Check container health
docker inspect --format='{{.State.Health.Status}}' iisc-chatbot
```

### Out of Memory

- Upgrade instance type
- Add swap space:
```bash
sudo fallocate -l 4G /swapfile
sudo chmod 600 /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile
```

## Security Best Practices

1. **Use IAM Roles** instead of hardcoding credentials
2. **Enable VPC** with private subnets
3. **Use AWS Secrets Manager** for GEMINI_API_KEY
4. **Enable CloudWatch** logging
5. **Setup AWS WAF** for DDoS protection
6. **Use HTTPS only** with valid SSL certificates
7. **Regular security updates**:
   ```bash
   sudo apt-get update && sudo apt-get upgrade -y
   ```

## Cost Optimization

1. Use **Reserved Instances** for 1-3 year commitment (up to 75% savings)
2. Setup **Auto Scaling** based on CPU/memory metrics
3. Use **Spot Instances** for non-critical workloads
4. Enable **CloudWatch Alarms** for unusual activity
5. Regular **cost monitoring** via AWS Cost Explorer

## Backup Strategy

```bash
# Backup index data (if modified)
aws s3 sync /opt/iisc-chatbot/data s3://your-backup-bucket/data/

# Backup configuration
aws s3 cp /opt/iisc-chatbot/.env s3://your-backup-bucket/config/.env
```

## Support

For issues or questions:
- Check logs: `docker-compose logs`
- GitHub Issues: [Create an issue](https://github.com/YOUR_USERNAME/IISC-chatbot/issues)
- Documentation: See README.md
