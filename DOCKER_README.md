# Docker Deployment Guide - EC50 Curve Fitting API Server

This guide explains how to build and deploy the EC50 curve fitting API server (`fit-server` binary) using Docker.

## 🐳 Quick Start

### Using Docker Compose (Recommended)

```bash
# Build and start the service
docker-compose up --build

# Run in background
docker-compose up -d --build

# Stop the service
docker-compose down
```

The server will be available at **http://localhost:3001** (externally mapped from internal port 3000)

### Using Docker directly

```bash
# Build the image
docker build -t fitrs-fit-server .

# Run the container (map external 3001 to internal 3000)
docker run -p 3001:3000 fitrs-fit-server

# Run with volume mounting for data
docker run -p 3001:3000 -v $(pwd)/data:/app/data:ro fitrs-fit-server
```

## 📦 Docker Configuration

### Dockerfile Features

- **Multi-stage build**: Optimized for production with separate build and runtime stages
- **Cargo Chef**: Efficient dependency caching for faster rebuilds
- **Security**: Runs as non-root user (`appuser`)
- **Health checks**: Built-in health monitoring
- **Minimal runtime**: Based on Debian slim for smaller image size

### Image Details

- **Base Image**: `debian:bookworm-slim`
- **Build Tool**: `cargo-chef` for optimized Rust builds
- **Port**: 3001
- **User**: `appuser` (non-root)
- **Health Check**: HTTP GET to `/` every 30 seconds

## 🔧 Configuration Options

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `RUST_LOG` | `info` | Log level (error, warn, info, debug, trace) |
| `TMPDIR` | `/tmp/fitrs` | Temporary directory for plot generation |

### Docker Compose Environment

```yaml
environment:
  - RUST_LOG=debug
  - TMPDIR=/tmp/fitrs
```

### Volume Mounts

```yaml
volumes:
  # Mount local data directory (read-only)
  - ./data:/app/data:ro
  
  # Mount custom config (optional)
  - ./config:/app/config:ro
```

## 🚀 Deployment Scenarios

### 1. Development

```bash
# Simple development setup
docker-compose up --build

# With file watching (requires separate tool)
docker-compose -f docker-compose.yml -f docker-compose.dev.yml up
```

### 2. Production with Nginx

```bash
# Start with nginx reverse proxy
docker-compose --profile production up -d --build
```

This setup includes:
- HTMX server on internal port 3001
- Nginx reverse proxy on ports 80/443
- Rate limiting and security headers
- SSL termination (configure certificates)

### 3. Kubernetes Deployment

Create `k8s-deployment.yaml`:

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: fitrs-htmx-server
spec:
  replicas: 3
  selector:
    matchLabels:
      app: fitrs-htmx-server
  template:
    metadata:
      labels:
        app: fitrs-htmx-server
    spec:
      containers:
      - name: htmx-server
        image: fitrs-htmx-server:latest
        ports:
        - containerPort: 3001
        env:
        - name: RUST_LOG
          value: "info"
        livenessProbe:
          httpGet:
            path: /
            port: 3001
          initialDelaySeconds: 10
          periodSeconds: 30
        readinessProbe:
          httpGet:
            path: /
            port: 3001
          initialDelaySeconds: 5
          periodSeconds: 10
---
apiVersion: v1
kind: Service
metadata:
  name: fitrs-htmx-service
spec:
  selector:
    app: fitrs-htmx-server
  ports:
  - port: 80
    targetPort: 3001
  type: LoadBalancer
```

## 🔒 Security Considerations

### Container Security

- **Non-root user**: Application runs as `appuser`
- **Minimal dependencies**: Only essential runtime packages
- **Read-only filesystem**: Templates mounted read-only
- **Limited capabilities**: No privileged access required

### Network Security

- **Nginx proxy**: Rate limiting, security headers
- **HTTPS**: SSL/TLS termination at proxy level
- **Internal networking**: Container-to-container communication

### Recommended Security Headers

```nginx
add_header X-Frame-Options DENY;
add_header X-Content-Type-Options nosniff;
add_header X-XSS-Protection "1; mode=block";
add_header Referrer-Policy strict-origin-when-cross-origin;
```

## 📊 Monitoring and Logs

### Container Logs

```bash
# View logs
docker-compose logs -f htmx-server

# View specific container logs
docker logs <container-id>

# Follow logs with timestamps
docker-compose logs -f -t
```

### Health Monitoring

```bash
# Check health status
docker-compose ps

# Manual health check
curl -f http://localhost:3001/ || echo "Health check failed"

# Detailed container inspection
docker inspect <container-id>
```

### Metrics Collection

For production monitoring, consider:

- **Prometheus**: Metrics collection
- **Grafana**: Visualization
- **Jaeger**: Distributed tracing
- **ELK Stack**: Log aggregation

## 🔧 Troubleshooting

### Common Issues

#### 1. Port Already in Use

```bash
# Check what's using port 3001
lsof -i:3001

# Kill existing processes
docker-compose down
```

#### 2. Build Failures

```bash
# Clean build (no cache)
docker-compose build --no-cache

# Remove old images
docker system prune -a
```

#### 3. Template Not Found

```bash
# Check if templates are copied correctly
docker exec -it <container-id> ls -la /app/templates/

# Rebuild with templates
docker-compose build --no-cache
```

#### 4. Permission Issues

```bash
# Check file permissions
docker exec -it <container-id> ls -la /app/

# Fix ownership (if needed)
docker exec -it <container-id> chown -R appuser:appuser /app/
```

### Debug Mode

```bash
# Run with debug logging
RUST_LOG=debug docker-compose up

# Interactive shell access
docker exec -it <container-id> /bin/bash
```

## 📋 Build Optimization

### Multi-stage Benefits

1. **Dependency caching**: Cargo Chef caches dependencies separately
2. **Smaller images**: Only runtime dependencies in final image
3. **Security**: No build tools in production image
4. **Faster deploys**: Efficient layer caching

### Build Performance

```bash
# Build with specific target
docker build --target builder -t fitrs-builder .

# Use BuildKit for faster builds
DOCKER_BUILDKIT=1 docker build -t fitrs-htmx-server .

# Parallel builds
docker-compose build --parallel
```

## 🌐 Production Checklist

- [ ] Configure SSL certificates
- [ ] Set up monitoring and alerting  
- [ ] Configure log rotation
- [ ] Set resource limits
- [ ] Enable backup strategies
- [ ] Configure firewall rules
- [ ] Set up load balancing
- [ ] Test disaster recovery

## 📈 Scaling

### Horizontal Scaling

```yaml
# docker-compose.yml
services:
  htmx-server:
    deploy:
      replicas: 3
    # ... other config
```

### Load Balancing

- Use nginx upstream for multiple containers
- Consider external load balancers (HAProxy, AWS ALB)
- Implement session affinity if needed

### Resource Limits

```yaml
# docker-compose.yml
services:
  htmx-server:
    deploy:
      resources:
        limits:
          cpus: '1.0'
          memory: 512M
        reservations:
          cpus: '0.5'
          memory: 256M
```

## 🔗 Integration Examples

### CI/CD Pipeline

```yaml
# .github/workflows/docker.yml
name: Docker Build and Deploy
on:
  push:
    branches: [main]

jobs:
  build:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v3
    
    - name: Build Docker image
      run: docker build -t fitrs-htmx-server .
      
    - name: Run tests
      run: docker run --rm fitrs-htmx-server cargo test
      
    - name: Push to registry
      run: |
        docker tag fitrs-htmx-server registry.example.com/fitrs-htmx-server
        docker push registry.example.com/fitrs-htmx-server
```

### Health Check Script

```bash
#!/bin/bash
# health-check.sh
response=$(curl -s -o /dev/null -w "%{http_code}" http://localhost:3001/)
if [ $response = "200" ]; then
    echo "Health check passed"
    exit 0
else
    echo "Health check failed with status $response"
    exit 1
fi
```