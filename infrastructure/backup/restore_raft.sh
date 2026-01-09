#!/bin/bash
# MerkleDB Raft Data Restore Script
# Restores Raft consensus data from backup

set -euo pipefail

# Configuration
RAFT_DATA_DIR="${RAFT_DATA_DIR:-data/raft}"
BACKUP_DIR="${BACKUP_DIR:-/var/backups/merkledb}"

# Check for backup file argument
if [ $# -eq 0 ]; then
    echo "Usage: $0 <backup_file>"
    echo ""
    echo "Available backups:"
    ls -lh "$BACKUP_DIR"/merkledb_raft_*.tar.gz 2>/dev/null || echo "No backups found"
    exit 1
fi

BACKUP_FILE="$1"

# Validate backup file exists
if [ ! -f "$BACKUP_FILE" ]; then
    echo "ERROR: Backup file not found: $BACKUP_FILE"
    exit 1
fi

echo "[$(date)] Starting Raft data restore from: $BACKUP_FILE"

# Safety check - ask for confirmation
read -p "WARNING: This will replace current Raft data. Continue? (yes/no): " confirm
if [ "$confirm" != "yes" ]; then
    echo "Restore cancelled"
    exit 0
fi

# Backup current data before restore
if [ -d "$RAFT_DATA_DIR" ]; then
    SAFETY_BACKUP="${RAFT_DATA_DIR}_before_restore_$(date +%Y%m%d_%H%M%S)"
    echo "[$(date)] Creating safety backup: $SAFETY_BACKUP"
    cp -r "$RAFT_DATA_DIR" "$SAFETY_BACKUP"
fi

# Remove old data
echo "[$(date)] Removing current Raft data..."
rm -rf "$RAFT_DATA_DIR"

# Extract backup
echo "[$(date)] Extracting backup..."
mkdir -p "$(dirname "$RAFT_DATA_DIR")"
tar -xzf "$BACKUP_FILE" -C "$(dirname "$RAFT_DATA_DIR")"

# Verify restore
if [ -d "$RAFT_DATA_DIR" ]; then
    echo "[$(date)] Restore complete!"
    echo "[$(date)] Restored data size: $(du -sh "$RAFT_DATA_DIR" | cut -f1)"
else
    echo "ERROR: Restore failed - data directory not found"
    exit 1
fi

echo ""
echo "NOTE: You must restart MerkleDB for changes to take effect"
echo "Run: systemctl restart merkledb"
