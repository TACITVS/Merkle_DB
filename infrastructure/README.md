# MerkleDB Infrastructure Guide

This directory contains production infrastructure configurations for deploying and monitoring MerkleDB.

## Directory Structure

```
infrastructure/
├── backup/              # Automated backup scripts
│   ├── backup_raft.sh   # Raft data backup
│   ├── restore_raft.sh  # Raft data restore
│   └── backup_cron.sh   # Cron automation
├── deployment/          # Deployment automation
│   ├── deploy.sh        # Rolling deployment script
│   ├── docker-compose.yml # Docker stack
│   └── systemd/         # SystemD service files
├── monitoring/          # Monitoring & alerting
│   ├── prometheus.yml   # Prometheus config
│   ├── alerts.yml       # Alert rules
│   └── alertmanager.yml # Alertmanager config
└── nginx/               # Load balancer
    └── merkledb.conf    # Nginx configuration
```

## Quick Start

### 1. Monitoring Setup

#### Prometheus + Grafana (Docker)

```bash
cd infrastructure/deployment
docker-compose up -d prometheus grafana alertmanager
```

Access:
- Prometheus: http://localhost:9090
- Grafana: http://localhost:3000 (admin/admin)
- Alertmanager: http://localhost:9093

#### Manual Installation

```bash
# Install Prometheus
sudo apt-get install prometheus
sudo cp monitoring/prometheus.yml /etc/prometheus/
sudo cp monitoring/alerts.yml /etc/prometheus/
sudo systemctl restart prometheus

# Install Grafana
sudo apt-get install grafana
sudo systemctl enable grafana-server
sudo systemctl start grafana-server
```

### 2. Automated Backups

#### Setup Cron Job

```bash
# Make scripts executable
chmod +x infrastructure/backup/*.sh

# Add to crontab (daily backups at 2 AM)
crontab -e
```

Add line:
```
0 2 * * * /opt/merkledb/infrastructure/backup/backup_cron.sh
```

#### Manual Backup

```bash
cd /opt/merkledb
./infrastructure/backup/backup_raft.sh
```

#### Restore from Backup

```bash
# List available backups
ls -lh /var/backups/merkledb/

# Restore specific backup
./infrastructure/backup/restore_raft.sh /var/backups/merkledb/merkledb_raft_20260109_143000.tar.gz
```

### 3. Load Balancer Setup

#### Nginx Installation

```bash
sudo apt-get install nginx

# Copy configuration
sudo cp infrastructure/nginx/merkledb.conf /etc/nginx/sites-available/
sudo ln -s /etc/nginx/sites-available/merkledb.conf /etc/nginx/sites-enabled/

# Update server names and SSL paths in merkledb.conf
sudo nano /etc/nginx/sites-available/merkledb.conf

# Test configuration
sudo nginx -t

# Reload nginx
sudo systemctl reload nginx
```

#### SSL Certificates (Let's Encrypt)

```bash
sudo apt-get install certbot python3-certbot-nginx
sudo certbot --nginx -d merkledb.example.com
```

### 4. Production Deployment

#### Build Release

```bash
# Set production environment
export MIX_ENV=prod

# Get dependencies
mix deps.get --only prod

# Compile
mix compile

# Build release
mix release
```

#### Deploy to Servers

```bash
# Configure deployment
export NODES="node1 node2 node3"
export DEPLOY_USER="deploy"
export RELEASE_VERSION="1.0.0"

# Run deployment
./infrastructure/deployment/deploy.sh
```

#### SystemD Service

```bash
# Copy service file
sudo cp infrastructure/deployment/systemd/merkledb.service /etc/systemd/system/

# Reload systemd
sudo systemctl daemon-reload

# Enable service
sudo systemctl enable merkledb

# Start service
sudo systemctl start merkledb

# Check status
sudo systemctl status merkledb
```

### 5. Docker Deployment

#### Build Docker Image

```bash
# Create Dockerfile (example)
cat > Dockerfile << 'EOF'
FROM elixir:1.15-alpine AS builder
WORKDIR /app
COPY mix.exs mix.lock ./
COPY config config
RUN mix local.hex --force && \
    mix local.rebar --force && \
    mix deps.get --only prod
COPY lib lib
COPY native native
RUN mix release

FROM alpine:3.18
RUN apk add --no-cache openssl ncurses-libs libstdc++
WORKDIR /opt/merkledb
COPY --from=builder /app/_build/prod/rel/merkle_db ./
CMD ["/opt/merkledb/bin/merkle_db", "start"]
EOF

# Build image
docker build -t merkledb:latest .
```

#### Start Cluster

```bash
cd infrastructure/deployment

# Set environment
export VERSION=latest
export RELEASE_COOKIE=$(openssl rand -base64 32)
export GRAFANA_PASSWORD=secure_password

# Start cluster
docker-compose up -d

# Check status
docker-compose ps

# View logs
docker-compose logs -f merkledb-node1
```

## Monitoring & Alerts

### Available Metrics

MerkleDB exposes Prometheus metrics at `GET /metrics`:

- `merkledb_queries_total` - Total queries executed
- `merkledb_query_duration_seconds` - Query latency
- `merkledb_cache_hit_rate` - Cache effectiveness
- `merkledb_vector_count` - Database size
- `merkledb_memory_usage_bytes` - Memory consumption
- `merkledb_raft_leader` - Raft leadership status
- `merkledb_health_status` - Overall health

### Alert Rules

See `monitoring/alerts.yml` for configured alerts:

- **Critical**: Service down, no Raft leader, health check failing
- **Warning**: High latency, low cache hit rate, resource pressure
- **Info**: Large vector count (scaling indicators)

### Grafana Dashboards

Import dashboards from:
- `monitoring/grafana/dashboards/merkledb-overview.json`
- `monitoring/grafana/dashboards/merkledb-performance.json`

## Backup & Recovery

### Backup Strategy

- **Frequency**: Daily at 2 AM (configurable)
- **Retention**: 30 days (configurable)
- **Location**: `/var/backups/merkledb/`
- **Format**: Compressed tar.gz

### Cloud Backups

Uncomment cloud upload in `backup/backup_cron.sh`:

```bash
# AWS S3
aws s3 cp "${BACKUP_DIR}/merkledb_raft_*.tar.gz" s3://your-bucket/merkledb/

# Google Cloud Storage
gcloud storage cp "${BACKUP_DIR}/merkledb_raft_*.tar.gz" gs://your-bucket/merkledb/
```

## Security Checklist

- [ ] Enable API authentication (`MERKLEDB_REQUIRE_AUTH=true`)
- [ ] Configure SSL/TLS certificates
- [ ] Set strong Raft cookies (`RELEASE_COOKIE`)
- [ ] Restrict `/metrics` endpoint to monitoring network
- [ ] Enable rate limiting in Nginx
- [ ] Configure firewall (allow only 80, 443, 4001)
- [ ] Set up log rotation
- [ ] Enable audit logging
- [ ] Regular security updates

## Performance Tuning

### Erlang VM Settings

Edit `rel/vm.args.eex`:
```
+P 1048576          # Max processes
+Q 65536            # Max ports
+K true             # Kernel polling
+A 32               # Async threads
+stbt ts            # Scheduler bind type
+swt very_low       # Scheduler wakeup threshold
```

### Nginx Tuning

```nginx
worker_processes auto;
worker_rlimit_nofile 65536;
events {
    worker_connections 4096;
    use epoll;
}
```

## Troubleshooting

### Check Health

```bash
# Liveness
curl http://localhost:4001/health/live

# Readiness
curl http://localhost:4001/health/ready

# Detailed
curl http://localhost:4001/health/detailed | jq
```

### View Logs

```bash
# SystemD
sudo journalctl -u merkledb -f

# Docker
docker-compose logs -f merkledb-node1

# File-based
tail -f /var/log/merkledb/erlang.log
```

### Raft Cluster Status

```bash
# Check leader
curl http://localhost:4001/health/detailed | jq '.raft.leader'

# Check members
curl http://localhost:4001/raft/members
```

## Scaling Guidelines

### Horizontal Scaling

Add nodes to `docker-compose.yml` or deploy new servers:

```yaml
merkledb-node4:
  image: merkledb:latest
  environment:
    - RELEASE_NODE=merkledb@merkledb-node4
  # ... configuration
```

### Vertical Scaling

Increase resources based on metrics:
- **Memory**: 16GB recommended for 1M+ vectors
- **CPU**: 8+ cores for high query load
- **Disk**: SSD for Raft data directory

## Support & Documentation

- Main docs: `docs/DEPLOYMENT.md`
- API reference: `docs/API.md`
- Configuration: `docs/CONFIGURATION.md`
- GitHub: https://github.com/TACITVS/Merkle_DB
