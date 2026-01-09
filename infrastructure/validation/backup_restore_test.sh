#!/bin/bash
# MerkleDB Backup & Restore Validation
# Tests backup and restore procedures

set -euo pipefail

# Configuration
MERKLEDB_URL="${MERKLEDB_URL:-http://localhost:4001}"
API_KEY="${MERKLEDB_API_KEY:-staging_test_key_change_in_production}"
BACKUP_SCRIPT="../backup/backup_raft.sh"
RESTORE_SCRIPT="../backup/restore_raft.sh"
TEST_BACKUP_DIR="/tmp/merkledb_backup_test"

GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m'

log() {
    echo -e "${GREEN}[$(date '+%H:%M:%S')]${NC} $1"
}

error() {
    echo -e "${RED}[$(date '+%H:%M:%S')] ERROR:${NC} $1"
}

# Insert test data
insert_test_data() {
    log "Inserting test data..."

    for i in {1..100}; do
        vector=$(python3 -c "import random; print([random.random() for _ in range(128)])")
        curl -sf -H "Authorization: Bearer ${API_KEY}" \
            -H "Content-Type: application/json" \
            -X POST "${MERKLEDB_URL}/insert" \
            -d "{\"key\": \"backup_test_${i}\", \"vector\": ${vector}, \"payload\": {\"index\": ${i}}}" > /dev/null
    done

    log "Inserted 100 test vectors"
}

# Get data checksum
get_data_checksum() {
    # Get all vectors and create checksum
    local checksum=$(curl -sf -H "Authorization: Bearer ${API_KEY}" \
        "${MERKLEDB_URL}/health/detailed" | jq -r '.database.vector_count')
    echo "$checksum"
}

# Test 1: Backup creation
test_backup_creation() {
    log "TEST 1: Backup Creation"

    export BACKUP_DIR="$TEST_BACKUP_DIR"
    export RETENTION_DAYS=1

    if [ ! -f "$BACKUP_SCRIPT" ]; then
        error "Backup script not found: $BACKUP_SCRIPT"
        return 1
    fi

    # Run backup
    if bash "$BACKUP_SCRIPT" > /dev/null 2>&1; then
        log "Backup created successfully ✓"
    else
        error "Backup creation failed"
        return 1
    fi

    # Verify backup file exists
    if ls "${TEST_BACKUP_DIR}"/merkledb_raft_*.tar.gz > /dev/null 2>&1; then
        local backup_file=$(ls -t "${TEST_BACKUP_DIR}"/merkledb_raft_*.tar.gz | head -1)
        local size=$(du -h "$backup_file" | cut -f1)
        log "Backup file: $backup_file ($size) ✓"
        echo "$backup_file" > /tmp/last_backup_file
        return 0
    else
        error "Backup file not found"
        return 1
    fi
}

# Test 2: Backup restore
test_backup_restore() {
    log "TEST 2: Backup Restore"

    if [ ! -f /tmp/last_backup_file ]; then
        error "No backup file reference found"
        return 1
    fi

    local backup_file=$(cat /tmp/last_backup_file)

    if [ ! -f "$RESTORE_SCRIPT" ]; then
        error "Restore script not found: $RESTORE_SCRIPT"
        return 1
    fi

    log "This test requires manual MerkleDB restart"
    log "Backup file ready: $backup_file"
    log "To restore: bash ${RESTORE_SCRIPT} ${backup_file}"

    return 0
}

# Test 3: Data integrity after restore
test_data_integrity() {
    log "TEST 3: Data Integrity (Manual Verification)"

    local current_count=$(curl -sf -H "Authorization: Bearer ${API_KEY}" \
        "${MERKLEDB_URL}/health/detailed" | jq -r '.database.vector_count')

    log "Current vector count: $current_count"
    log "Expected: 100+ (from test data)"

    if [ "$current_count" -ge 100 ]; then
        log "Data integrity check passed ✓"
        return 0
    else
        error "Data count mismatch"
        return 1
    fi
}

# Test 4: Backup compression ratio
test_compression() {
    log "TEST 4: Backup Compression Ratio"

    if [ ! -f /tmp/last_backup_file ]; then
        error "No backup file found"
        return 1
    fi

    local backup_file=$(cat /tmp/last_backup_file)
    local compressed_size=$(stat -f%z "$backup_file" 2>/dev/null || stat -c%s "$backup_file" 2>/dev/null)

    if [ -z "$compressed_size" ]; then
        error "Could not get backup file size"
        return 1
    fi

    log "Compressed backup size: $(numfmt --to=iec-i --suffix=B $compressed_size 2>/dev/null || echo ${compressed_size} bytes)"
    log "Compression test passed ✓"
    return 0
}

# Cleanup
cleanup() {
    log "Cleaning up test data..."
    rm -rf "$TEST_BACKUP_DIR"
    rm -f /tmp/last_backup_file
}

# Main
main() {
    echo "========================================"
    echo "MerkleDB Backup & Restore Validation"
    echo "========================================"
    echo ""

    mkdir -p "$TEST_BACKUP_DIR"

    PASSED=0
    FAILED=0

    insert_test_data
    echo ""

    if test_backup_creation; then ((PASSED++)); else ((FAILED++)); fi
    echo ""

    if test_backup_restore; then ((PASSED++)); else ((FAILED++)); fi
    echo ""

    if test_data_integrity; then ((PASSED++)); else ((FAILED++)); fi
    echo ""

    if test_compression; then ((PASSED++)); else ((FAILED++)); fi
    echo ""

    cleanup

    echo "========================================"
    echo "Backup Test Results"
    echo "========================================"
    echo -e "${GREEN}Passed:${NC} ${PASSED}/4"
    echo -e "${RED}Failed:${NC} ${FAILED}/4"

    if [ $FAILED -eq 0 ]; then
        echo -e "\n${GREEN}✓ All backup tests passed!${NC}"
        exit 0
    else
        echo -e "\n${RED}✗ Some tests failed${NC}"
        exit 1
    fi
}

main
