#!/bin/bash
# MerkleDB Raft Failover Testing
# Tests Raft cluster resilience and leader election

set -euo pipefail

# Configuration
NODES="${NODES:-node1:4001 node2:4002 node3:4003}"
API_KEY="${MERKLEDB_API_KEY:-staging_test_key_change_in_production}"
TEST_DURATION=300  # 5 minutes

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

warn() {
    echo -e "${YELLOW}[$(date '+%H:%M:%S')] WARNING:${NC} $1"
}

# Get current Raft leader
get_leader() {
    for node in $NODES; do
        host=$(echo "$node" | cut -d: -f1)
        port=$(echo "$node" | cut -d: -f2)

        response=$(curl -sf "http://${host}:${port}/health/detailed" 2>/dev/null || echo "{}")
        is_leader=$(echo "$response" | jq -r '.raft.is_leader // false' 2>/dev/null)

        if [ "$is_leader" = "true" ]; then
            echo "$node"
            return 0
        fi
    done
    return 1
}

# Check if node is healthy
is_healthy() {
    local node=$1
    local host=$(echo "$node" | cut -d: -f1)
    local port=$(echo "$node" | cut -d: -f2)

    curl -sf "http://${host}:${port}/health/ready" > /dev/null 2>&1
}

# Stop a node (Docker)
stop_node() {
    local node_name=$1
    log "Stopping ${node_name}..."
    docker stop "${node_name}" > /dev/null 2>&1 || true
}

# Start a node (Docker)
start_node() {
    local node_name=$1
    log "Starting ${node_name}..."
    docker start "${node_name}" > /dev/null 2>&1 || true
    sleep 5  # Give time to start
}

# Continuous write test
continuous_writes() {
    local node=$1
    local host=$(echo "$node" | cut -d: -f1)
    local port=$(echo "$node" | cut -d: -f2)
    local count=0

    while true; do
        vector=$(python3 -c "import random; print([random.random() for _ in range(128)])")
        key="failover_test_${count}"

        response=$(curl -sf -H "Authorization: Bearer ${API_KEY}" \
            -H "Content-Type: application/json" \
            -X POST "http://${host}:${port}/insert" \
            -d "{\"key\": \"${key}\", \"vector\": ${vector}}" 2>/dev/null || echo '{"status":"error"}')

        if echo "$response" | jq -e '.status == "ok"' > /dev/null 2>&1; then
            ((count++))
        fi

        sleep 1
    done
}

# Test 1: Leader election
test_leader_election() {
    log "TEST 1: Leader Election"

    # Get current leader
    leader=$(get_leader)
    if [ -z "$leader" ]; then
        error "No leader found in cluster"
        return 1
    fi
    log "Current leader: $leader"

    # Stop leader
    leader_container=$(echo "$leader" | cut -d: -f1)
    stop_node "merkledb-${leader_container}"

    # Wait for new leader election
    log "Waiting for new leader election..."
    sleep 10

    new_leader=$(get_leader)
    if [ -z "$new_leader" ]; then
        error "No new leader elected after stopping old leader"
        start_node "merkledb-${leader_container}"  # Restore
        return 1
    fi

    if [ "$new_leader" = "$leader" ]; then
        error "Same leader elected (should be different)"
        start_node "merkledb-${leader_container}"
        return 1
    fi

    log "New leader elected: $new_leader ✓"

    # Restore old leader
    start_node "merkledb-${leader_container}"
    sleep 10

    log "Leader election test passed ✓"
    return 0
}

# Test 2: Write availability during failover
test_write_availability() {
    log "TEST 2: Write Availability During Failover"

    leader=$(get_leader)
    follower=$(echo "$NODES" | tr ' ' '\n' | grep -v "$leader" | head -1)

    log "Starting continuous writes to follower: $follower"

    # Start background writes
    continuous_writes "$follower" > /tmp/failover_writes.log 2>&1 &
    write_pid=$!

    sleep 5

    # Kill leader
    leader_container=$(echo "$leader" | cut -d: -f1)
    log "Stopping leader: $leader"
    stop_node "merkledb-${leader_container}"

    # Continue writes for 30 seconds
    sleep 30

    # Stop writes
    kill $write_pid 2>/dev/null || true

    # Count successful writes
    successful_writes=$(grep -c '"status":"ok"' /tmp/failover_writes.log || echo 0)
    log "Successful writes during failover: $successful_writes"

    # Restore leader
    start_node "merkledb-${leader_container}"

    if [ "$successful_writes" -gt 0 ]; then
        log "Write availability test passed ✓"
        return 0
    else
        error "No successful writes during failover"
        return 1
    fi
}

# Test 3: Split brain prevention
test_split_brain() {
    log "TEST 3: Split Brain Prevention"

    # Stop 2 nodes (minority)
    nodes_arr=($NODES)
    node1="${nodes_arr[0]}"
    node2="${nodes_arr[1]}"

    container1=$(echo "$node1" | cut -d: -f1)
    container2=$(echo "$node2" | cut -d: -f1)

    log "Stopping 2 nodes to create minority partition..."
    stop_node "merkledb-${container1}"
    stop_node "merkledb-${container2}"

    sleep 10

    # Try to get leader (should fail - no quorum)
    leader=$(get_leader || echo "")

    if [ -z "$leader" ]; then
        log "No leader elected without quorum ✓"
        result=0
    else
        error "Leader elected without quorum (split brain possible)"
        result=1
    fi

    # Restore nodes
    start_node "merkledb-${container1}"
    start_node "merkledb-${container2}"
    sleep 15

    return $result
}

# Test 4: Data consistency after failover
test_data_consistency() {
    log "TEST 4: Data Consistency After Failover"

    # Insert known data
    test_key="consistency_test_$(date +%s)"
    test_vector=$(python3 -c "import random; print([random.random() for _ in range(128)])")

    leader=$(get_leader)
    host=$(echo "$leader" | cut -d: -f1)
    port=$(echo "$leader" | cut -d: -f2)

    log "Inserting test vector on leader..."
    curl -sf -H "Authorization: Bearer ${API_KEY}" \
        -H "Content-Type: application/json" \
        -X POST "http://${host}:${port}/insert" \
        -d "{\"key\": \"${test_key}\", \"vector\": ${test_vector}}" > /dev/null

    sleep 2

    # Force failover
    leader_container=$(echo "$leader" | cut -d: -f1)
    stop_node "merkledb-${leader_container}"
    sleep 10

    # Read from new leader
    new_leader=$(get_leader)
    new_host=$(echo "$new_leader" | cut -d: -f1)
    new_port=$(echo "$new_leader" | cut -d: -f2)

    log "Checking data on new leader..."
    response=$(curl -sf -H "Authorization: Bearer ${API_KEY}" \
        "http://${new_host}:${new_port}/get/${test_key}" || echo '{"status":"error"}')

    # Restore
    start_node "merkledb-${leader_container}"

    if echo "$response" | jq -e '.vector' > /dev/null 2>&1; then
        log "Data consistency test passed ✓"
        return 0
    else
        error "Data not found after failover"
        return 1
    fi
}

# Main test execution
main() {
    echo "========================================"
    echo "MerkleDB Raft Failover Tests"
    echo "========================================"
    echo "Nodes: $NODES"
    echo ""

    PASSED=0
    FAILED=0

    if test_leader_election; then ((PASSED++)); else ((FAILED++)); fi
    echo ""

    if test_write_availability; then ((PASSED++)); else ((FAILED++)); fi
    echo ""

    if test_split_brain; then ((PASSED++)); else ((FAILED++)); fi
    echo ""

    if test_data_consistency; then ((PASSED++)); else ((FAILED++)); fi
    echo ""

    echo "========================================"
    echo "Failover Test Results"
    echo "========================================"
    echo -e "${GREEN}Passed:${NC} ${PASSED}/4"
    echo -e "${RED}Failed:${NC} ${FAILED}/4"

    if [ $FAILED -eq 0 ]; then
        echo -e "\n${GREEN}✓ All failover tests passed!${NC}"
        exit 0
    else
        echo -e "\n${RED}✗ Some failover tests failed${NC}"
        exit 1
    fi
}

main
