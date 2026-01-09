#!/bin/bash
# MerkleDB Raft Data Backup Script
# Backs up Raft consensus data directory with compression

set -euo pipefail

# Configuration
RAFT_DATA_DIR="${RAFT_DATA_DIR:-data/raft}"
BACKUP_DIR="${BACKUP_DIR:-/var/backups/merkledb}"
RETENTION_DAYS="${RETENTION_DAYS:-30}"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
BACKUP_NAME="merkledb_raft_${TIMESTAMP}.tar.gz"

# Create backup directory if it doesn't exist
mkdir -p "$BACKUP_DIR"

echo "[$(date)] Starting Raft data backup..."

# Check if Raft data directory exists
if [ ! -d "$RAFT_DATA_DIR" ]; then
    echo "ERROR: Raft data directory not found: $RAFT_DATA_DIR"
    exit 1
fi

# Create compressed backup
tar -czf "${BACKUP_DIR}/${BACKUP_NAME}" -C "$(dirname "$RAFT_DATA_DIR")" "$(basename "$RAFT_DATA_DIR")"

# Verify backup was created
if [ -f "${BACKUP_DIR}/${BACKUP_NAME}" ]; then
    BACKUP_SIZE=$(du -h "${BACKUP_DIR}/${BACKUP_NAME}" | cut -f1)
    echo "[$(date)] Backup created: ${BACKUP_NAME} (${BACKUP_SIZE})"
else
    echo "ERROR: Backup file not created"
    exit 1
fi

# Clean up old backups (older than RETENTION_DAYS)
echo "[$(date)] Cleaning up backups older than ${RETENTION_DAYS} days..."
find "$BACKUP_DIR" -name "merkledb_raft_*.tar.gz" -type f -mtime +${RETENTION_DAYS} -delete

# List current backups
echo "[$(date)] Current backups:"
ls -lh "$BACKUP_DIR"/merkledb_raft_*.tar.gz 2>/dev/null || echo "No backups found"

echo "[$(date)] Backup complete!"
