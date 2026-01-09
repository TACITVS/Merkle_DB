#!/bin/bash
# MerkleDB Automated Backup with Cron
# Add to crontab: 0 2 * * * /path/to/backup_cron.sh

set -euo pipefail

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_FILE="/var/log/merkledb/backup.log"
MERKLEDB_DIR="${MERKLEDB_DIR:-/opt/merkledb}"
BACKUP_DIR="${BACKUP_DIR:-/var/backups/merkledb}"

# Ensure log directory exists
mkdir -p "$(dirname "$LOG_FILE")"

# Log function
log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "$LOG_FILE"
}

log "=== Starting automated MerkleDB backup ==="

# Change to MerkleDB directory
cd "$MERKLEDB_DIR"

# Run backup
if "$SCRIPT_DIR/backup_raft.sh" >> "$LOG_FILE" 2>&1; then
    log "Backup completed successfully"

    # Optional: Upload to cloud storage (uncomment and configure)
    # aws s3 cp "${BACKUP_DIR}/merkledb_raft_$(date +%Y%m%d)*.tar.gz" s3://your-bucket/merkledb/
    # gcloud storage cp "${BACKUP_DIR}/merkledb_raft_$(date +%Y%m%d)*.tar.gz" gs://your-bucket/merkledb/

    exit 0
else
    log "ERROR: Backup failed"

    # Optional: Send alert (uncomment and configure)
    # curl -X POST https://your-alerting-service/alert \
    #   -d "MerkleDB backup failed on $(hostname)"

    exit 1
fi
