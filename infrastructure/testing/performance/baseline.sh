#!/bin/bash
# MerkleDB Performance Baseline
# Establishes performance metrics baseline for regression detection

set -euo pipefail

# Configuration
MERKLEDB_URL="${MERKLEDB_URL:-http://localhost:4001}"
API_KEY="${MERKLEDB_API_KEY:-staging_test_key_change_in_production}"
OUTPUT_FILE="${OUTPUT_FILE:-baseline_$(date +%Y%m%d_%H%M%S).json}"

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

# Generate random vector
random_vector() {
    python3 -c "import random; print([random.random() for _ in range(128)])"
}

# Measure operation latency
measure_latency() {
    local url=$1
    local payload=$2
    local iterations=${3:-100}

    local total_time=0
    local success_count=0

    for ((i=1; i<=iterations; i++)); do
        start=$(date +%s%N)
        response=$(curl -sf -H "Authorization: Bearer ${API_KEY}" \
            -H "Content-Type: application/json" \
            -X POST "$url" \
            -d "$payload" 2>/dev/null || echo '{"status":"error"}')
        end=$(date +%s%N)

        if echo "$response" | jq -e '.status != "error"' > /dev/null 2>&1; then
            duration=$(( (end - start) / 1000000 ))  # Convert to milliseconds
            total_time=$((total_time + duration))
            ((success_count++))
        fi
    done

    if [ $success_count -gt 0 ]; then
        echo $((total_time / success_count))
    else
        echo "0"
    fi
}

# Test 1: Insert Performance
test_insert_performance() {
    log "TEST 1: Insert Performance (100 operations)"

    local vector=$(random_vector)
    local payload="{\"key\": \"baseline_\$RANDOM\", \"vector\": ${vector}}"

    local avg_latency=$(measure_latency "${MERKLEDB_URL}/insert" "$payload" 100)

    log "Average insert latency: ${avg_latency}ms"
    echo "$avg_latency"
}

# Test 2: Search Performance
test_search_performance() {
    log "TEST 2: Search Performance (100 operations)"

    # First, insert some data
    for i in {1..1000}; do
        vector=$(random_vector)
        curl -sf -H "Authorization: Bearer ${API_KEY}" \
            -H "Content-Type: application/json" \
            -X POST "${MERKLEDB_URL}/insert" \
            -d "{\"key\": \"search_baseline_${i}\", \"vector\": ${vector}}" > /dev/null 2>&1 || true
    done

    sleep 2

    local query=$(random_vector)
    local payload="{\"vector\": ${query}, \"k\": 10}"

    local avg_latency=$(measure_latency "${MERKLEDB_URL}/search" "$payload" 100)

    log "Average search latency: ${avg_latency}ms"
    echo "$avg_latency"
}

# Test 3: Batch Insert Performance
test_batch_performance() {
    log "TEST 3: Batch Insert Performance (10 batches of 100)"

    local vectors="["
    for i in {1..100}; do
        vectors="${vectors}{\"key\":\"batch_baseline_${i}\",\"vector\":$(random_vector)}"
        if [ $i -lt 100 ]; then
            vectors="${vectors},"
        fi
    done
    vectors="${vectors}]"

    local payload="{\"vectors\": ${vectors}}"
    local avg_latency=$(measure_latency "${MERKLEDB_URL}/insert_batch" "$payload" 10)

    log "Average batch insert latency: ${avg_latency}ms (100 vectors/batch)"
    echo "$avg_latency"
}

# Test 4: Concurrent Query Performance
test_concurrent_performance() {
    log "TEST 4: Concurrent Query Performance (10 parallel clients)"

    local total_queries=100
    local concurrent_clients=10
    local queries_per_client=$((total_queries / concurrent_clients))

    local start_time=$(date +%s)

    for ((c=1; c<=concurrent_clients; c++)); do
        (
            for ((i=1; i<=queries_per_client; i++)); do
                query=$(random_vector)
                curl -sf -H "Authorization: Bearer ${API_KEY}" \
                    -H "Content-Type: application/json" \
                    -X POST "${MERKLEDB_URL}/search" \
                    -d "{\"vector\": ${query}, \"k\": 10}" > /dev/null 2>&1 || true
            done
        ) &
    done

    wait

    local end_time=$(date +%s)
    local duration=$((end_time - start_time))
    local throughput=$((total_queries / duration))

    log "Concurrent throughput: ${throughput} queries/second"
    echo "$throughput"
}

# Test 5: Memory Usage
test_memory_usage() {
    log "TEST 5: Memory Usage"

    local response=$(curl -sf -H "Authorization: Bearer ${API_KEY}" \
        "${MERKLEDB_URL}/health/detailed" || echo '{}')

    local memory_mb=$(echo "$response" | jq -r '.system.memory_mb // 0')

    log "Current memory usage: ${memory_mb}MB"
    echo "$memory_mb"
}

# Test 6: Cache Performance
test_cache_performance() {
    log "TEST 6: Cache Hit Rate"

    local response=$(curl -sf -H "Authorization: Bearer ${API_KEY}" \
        "${MERKLEDB_URL}/health/detailed" || echo '{}')

    local cache_hits=$(echo "$response" | jq -r '.metrics.cache_hits // 0')
    local cache_misses=$(echo "$response" | jq -r '.metrics.cache_misses // 0')
    local total=$((cache_hits + cache_misses))

    if [ $total -gt 0 ]; then
        local hit_rate=$(echo "scale=4; $cache_hits / $total" | bc)
        log "Cache hit rate: ${hit_rate}"
        echo "$hit_rate"
    else
        log "No cache data available"
        echo "0"
    fi
}

# Main baseline execution
main() {
    echo "========================================"
    echo "MerkleDB Performance Baseline"
    echo "========================================"
    echo "Target: ${MERKLEDB_URL}"
    echo "Output: ${OUTPUT_FILE}"
    echo ""

    # Check if MerkleDB is running
    if ! curl -sf "${MERKLEDB_URL}/health/live" > /dev/null 2>&1; then
        error "MerkleDB is not responding at ${MERKLEDB_URL}"
        exit 1
    fi

    log "Starting baseline measurements..."
    echo ""

    # Run tests
    insert_latency=$(test_insert_performance)
    echo ""

    search_latency=$(test_search_performance)
    echo ""

    batch_latency=$(test_batch_performance)
    echo ""

    throughput=$(test_concurrent_performance)
    echo ""

    memory_mb=$(test_memory_usage)
    echo ""

    cache_hit_rate=$(test_cache_performance)
    echo ""

    # Get system info
    system_info=$(curl -sf -H "Authorization: Bearer ${API_KEY}" \
        "${MERKLEDB_URL}/health/detailed" || echo '{}')
    vector_count=$(echo "$system_info" | jq -r '.database.vector_count // 0')

    # Create baseline report
    cat > "$OUTPUT_FILE" <<EOF
{
  "timestamp": "$(date -u +%Y-%m-%dT%H:%M:%SZ)",
  "merkledb_url": "${MERKLEDB_URL}",
  "performance": {
    "insert_latency_ms": ${insert_latency},
    "search_latency_ms": ${search_latency},
    "batch_insert_latency_ms": ${batch_latency},
    "concurrent_throughput_qps": ${throughput},
    "cache_hit_rate": ${cache_hit_rate}
  },
  "system": {
    "memory_usage_mb": ${memory_mb},
    "vector_count": ${vector_count}
  },
  "targets": {
    "insert_latency_ms": 50,
    "search_latency_ms": 100,
    "batch_insert_latency_ms": 500,
    "concurrent_throughput_qps": 100,
    "cache_hit_rate": 0.7
  }
}
EOF

    echo ""
    echo "========================================"
    echo "Baseline Results Summary"
    echo "========================================"
    echo -e "${GREEN}Insert Latency:${NC} ${insert_latency}ms (target: <50ms)"
    echo -e "${GREEN}Search Latency:${NC} ${search_latency}ms (target: <100ms)"
    echo -e "${GREEN}Batch Latency:${NC} ${batch_latency}ms (target: <500ms)"
    echo -e "${GREEN}Throughput:${NC} ${throughput} qps (target: >100 qps)"
    echo -e "${GREEN}Cache Hit Rate:${NC} ${cache_hit_rate} (target: >0.7)"
    echo -e "${GREEN}Memory Usage:${NC} ${memory_mb}MB"
    echo ""
    echo -e "${GREEN}✓ Baseline saved to ${OUTPUT_FILE}${NC}"

    # Performance warnings
    [ "$insert_latency" -gt 50 ] && echo -e "${YELLOW}⚠ Insert latency exceeds target${NC}"
    [ "$search_latency" -gt 100 ] && echo -e "${YELLOW}⚠ Search latency exceeds target${NC}"
    [ "$batch_latency" -gt 500 ] && echo -e "${YELLOW}⚠ Batch latency exceeds target${NC}"
    [ "$throughput" -lt 100 ] && echo -e "${YELLOW}⚠ Throughput below target${NC}"
}

main
