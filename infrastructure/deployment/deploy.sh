#!/bin/bash
# MerkleDB Production Deployment Script
# Deploys MerkleDB to production servers with zero-downtime rolling update

set -euo pipefail

# Configuration
NODES="${NODES:-node1 node2 node3}"
DEPLOY_USER="${DEPLOY_USER:-deploy}"
MERKLEDB_DIR="${MERKLEDB_DIR:-/opt/merkledb}"
RELEASE_VERSION="${1:-latest}"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

log() {
    echo -e "${GREEN}[$(date '+%Y-%m-%d %H:%M:%S')]${NC} $1"
}

error() {
    echo -e "${RED}[$(date '+%Y-%m-%d %H:%M:%S')] ERROR:${NC} $1"
}

warn() {
    echo -e "${YELLOW}[$(date '+%Y-%m-%d %H:%M:%S')] WARNING:${NC} $1"
}

# Pre-deployment checks
preflight_checks() {
    log "Running pre-flight checks..."

    # Check if release file exists
    if [ ! -f "_build/prod/merkle_db-${RELEASE_VERSION}.tar.gz" ]; then
        error "Release file not found: _build/prod/merkle_db-${RELEASE_VERSION}.tar.gz"
        exit 1
    fi

    # Check SSH connectivity to all nodes
    for node in $NODES; do
        if ! ssh -o ConnectTimeout=5 ${DEPLOY_USER}@${node} "echo 'SSH OK'" &>/dev/null; then
            error "Cannot connect to ${node}"
            exit 1
        fi
    done

    log "Pre-flight checks passed!"
}

# Deploy to a single node
deploy_node() {
    local node=$1
    log "Deploying to ${node}..."

    # Create deployment directory
    ssh ${DEPLOY_USER}@${node} "mkdir -p ${MERKLEDB_DIR}/releases/${RELEASE_VERSION}"

    # Upload release
    scp "_build/prod/merkle_db-${RELEASE_VERSION}.tar.gz" \
        ${DEPLOY_USER}@${node}:${MERKLEDB_DIR}/releases/${RELEASE_VERSION}/

    # Extract release
    ssh ${DEPLOY_USER}@${node} "cd ${MERKLEDB_DIR}/releases/${RELEASE_VERSION} && tar -xzf merkle_db-${RELEASE_VERSION}.tar.gz"

    # Create/update symlink
    ssh ${DEPLOY_USER}@${node} "ln -sfn ${MERKLEDB_DIR}/releases/${RELEASE_VERSION} ${MERKLEDB_DIR}/current"

    log "Deployed to ${node}"
}

# Graceful restart of a node
restart_node() {
    local node=$1
    log "Restarting ${node}..."

    # Stop the node gracefully
    ssh ${DEPLOY_USER}@${node} "systemctl stop merkledb || ${MERKLEDB_DIR}/current/bin/merkle_db stop" || true

    # Wait for node to stop
    sleep 5

    # Start the node
    ssh ${DEPLOY_USER}@${node} "systemctl start merkledb || ${MERKLEDB_DIR}/current/bin/merkle_db daemon"

    # Wait for node to be healthy
    local retries=0
    while [ $retries -lt 30 ]; do
        if ssh ${DEPLOY_USER}@${node} "curl -sf http://localhost:4001/health/ready" &>/dev/null; then
            log "${node} is healthy"
            return 0
        fi
        sleep 2
        ((retries++))
    done

    error "${node} failed to become healthy"
    return 1
}

# Rolling deployment
rolling_deploy() {
    log "Starting rolling deployment..."

    for node in $NODES; do
        log "Processing ${node}..."

        # Deploy new version
        deploy_node "$node"

        # Restart node
        if ! restart_node "$node"; then
            error "Failed to restart ${node}"

            # Rollback decision
            read -p "Rollback deployment? (yes/no): " rollback
            if [ "$rollback" = "yes" ]; then
                warn "Rolling back ${node}..."
                ssh ${DEPLOY_USER}@${node} "systemctl restart merkledb"
            fi
            exit 1
        fi

        # Brief pause between nodes
        if [ "$node" != "${NODES##* }" ]; then
            log "Waiting 30 seconds before next node..."
            sleep 30
        fi
    done

    log "Rolling deployment complete!"
}

# Post-deployment verification
verify_deployment() {
    log "Verifying deployment..."

    for node in $NODES; do
        # Check health
        if ! ssh ${DEPLOY_USER}@${node} "curl -sf http://localhost:4001/health/ready" &>/dev/null; then
            error "${node} health check failed"
            return 1
        fi

        # Check version
        version=$(ssh ${DEPLOY_USER}@${node} "curl -s http://localhost:4001/health/detailed | jq -r .version")
        log "${node}: version ${version}"
    done

    log "Deployment verified successfully!"
}

# Main deployment flow
main() {
    log "=== MerkleDB Production Deployment ==="
    log "Version: ${RELEASE_VERSION}"
    log "Nodes: ${NODES}"

    preflight_checks
    rolling_deploy
    verify_deployment

    log "=== Deployment Complete! ==="
}

# Run main deployment
main
