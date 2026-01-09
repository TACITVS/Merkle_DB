#!/bin/bash
# MerkleDB Query Benchmarking
# Tests different query patterns and vector dimensions

set -euo pipefail

MERKLEDB_URL="${MERKLEDB_URL:-http://localhost:4001}"
API_KEY="${MERKLEDB_API_KEY:-staging_test_key_change_in_production}"

GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m'

log() {
    echo -e "${GREEN}[$(date '+%H:%M:%S')]${NC} $1"
}

# Generate random vector of specified dimension
random_vector() {
    local dim=$1
    python3 -c "import random; print([random.random() for _ in range(${dim})])"
}

# Benchmark query with specific k value
benchmark_k_value() {
    local k=$1
    local iterations=50

    log "Benchmarking k=${k} (${iterations} queries)..."

    local total_time=0
    local success=0

    for ((i=1; i<=iterations; i++)); do
        query=$(random_vector 128)
        start=$(date +%s%N)
        response=$(curl -sf -H "Authorization: Bearer ${API_KEY}" \
            -H "Content-Type: application/json" \
            -X POST "${MERKLEDB_URL}/search" \
            -d "{\"vector\": ${query}, \"k\": ${k}}" 2>/dev/null || echo '{"error":true}')
        end=$(date +%s%N)

        if ! echo "$response" | jq -e '.error' > /dev/null 2>&1; then
            duration=$(( (end - start) / 1000000 ))
            total_time=$((total_time + duration))
            ((success++))
        fi
    done

    if [ $success -gt 0 ]; then
        local avg=$((total_time / success))
        echo -e "  k=${k}: ${avg}ms (${success}/${iterations} successful)"
        echo "$avg"
    else
        echo -e "  ${RED}k=${k}: FAILED${NC}"
        echo "0"
    fi
}

# Benchmark different vector dimensions
benchmark_dimensions() {
    log "TEST 1: Query Performance by K Value"
    echo ""

    k_values=(1 5 10 20 50 100)
    for k in "${k_values[@]}"; do
        benchmark_k_value "$k"
    done
}

# Benchmark with metadata filtering
benchmark_filtered_queries() {
    log "TEST 2: Filtered Query Performance"
    echo ""

    # Insert data with metadata
    log "Inserting 500 vectors with metadata..."
    for i in {1..500}; do
        vector=$(random_vector 128)
        category=$((RANDOM % 5))
        curl -sf -H "Authorization: Bearer ${API_KEY}" \
            -H "Content-Type: application/json" \
            -X POST "${MERKLEDB_URL}/insert" \
            -d "{\"key\": \"filtered_${i}\", \"vector\": ${vector}, \"payload\": {\"category\": ${category}}}" \
            > /dev/null 2>&1 || true
    done

    sleep 2

    # Benchmark filtered queries
    local total_time=0
    local iterations=50

    for ((i=1; i<=iterations; i++)); do
        query=$(random_vector 128)
        start=$(date +%s%N)
        curl -sf -H "Authorization: Bearer ${API_KEY}" \
            -H "Content-Type: application/json" \
            -X POST "${MERKLEDB_URL}/search" \
            -d "{\"vector\": ${query}, \"k\": 10, \"filter\": {\"category\": 1}}" \
            > /dev/null 2>&1 || true
        end=$(date +%s%N)
        duration=$(( (end - start) / 1000000 ))
        total_time=$((total_time + duration))
    done

    local avg=$((total_time / iterations))
    echo -e "  Filtered queries: ${avg}ms average"
}

# Benchmark query latency under load
benchmark_load_impact() {
    log "TEST 3: Query Latency Under Load"
    echo ""

    # Baseline (no load)
    query=$(random_vector 128)
    start=$(date +%s%N)
    curl -sf -H "Authorization: Bearer ${API_KEY}" \
        -H "Content-Type: application/json" \
        -X POST "${MERKLEDB_URL}/search" \
        -d "{\"vector\": ${query}, \"k\": 10}" \
        > /dev/null 2>&1
    end=$(date +%s%N)
    baseline=$(( (end - start) / 1000000 ))

    echo -e "  Baseline (no load): ${baseline}ms"

    # Under load (10 concurrent clients)
    log "Starting 10 concurrent background clients..."

    for c in {1..10}; do
        (
            for i in {1..100}; do
                q=$(random_vector 128)
                curl -sf -H "Authorization: Bearer ${API_KEY}" \
                    -H "Content-Type: application/json" \
                    -X POST "${MERKLEDB_URL}/search" \
                    -d "{\"vector\": ${q}, \"k\": 10}" \
                    > /dev/null 2>&1 || true
                sleep 0.1
            done
        ) &
    done

    sleep 2

    # Measure during load
    local total=0
    for i in {1..20}; do
        query=$(random_vector 128)
        start=$(date +%s%N)
        curl -sf -H "Authorization: Bearer ${API_KEY}" \
            -H "Content-Type: application/json" \
            -X POST "${MERKLEDB_URL}/search" \
            -d "{\"vector\": ${query}, \"k\": 10}" \
            > /dev/null 2>&1
        end=$(date +%s%N)
        duration=$(( (end - start) / 1000000 ))
        total=$((total + duration))
    done

    wait

    local under_load=$((total / 20))
    local impact=$(( (under_load - baseline) * 100 / baseline ))

    echo -e "  Under load (10 clients): ${under_load}ms"
    echo -e "  Latency impact: +${impact}%"
}

# Benchmark different database sizes
benchmark_database_size() {
    log "TEST 4: Query Performance vs Database Size"
    echo ""

    sizes=(100 1000 5000 10000)

    for size in "${sizes[@]}"; do
        log "Testing with ${size} vectors..."

        # Insert vectors
        for ((i=1; i<=size; i++)); do
            vector=$(random_vector 128)
            curl -sf -H "Authorization: Bearer ${API_KEY}" \
                -H "Content-Type: application/json" \
                -X POST "${MERKLEDB_URL}/insert" \
                -d "{\"key\": \"size_test_${size}_${i}\", \"vector\": ${vector}}" \
                > /dev/null 2>&1 || true

            # Progress indicator
            if [ $((i % 100)) -eq 0 ]; then
                echo -ne "  Inserted ${i}/${size}\r"
            fi
        done
        echo ""

        sleep 2

        # Benchmark
        local total=0
        local iterations=20

        for ((i=1; i<=iterations; i++)); do
            query=$(random_vector 128)
            start=$(date +%s%N)
            curl -sf -H "Authorization: Bearer ${API_KEY}" \
                -H "Content-Type: application/json" \
                -X POST "${MERKLEDB_URL}/search" \
                -d "{\"vector\": ${query}, \"k\": 10}" \
                > /dev/null 2>&1
            end=$(date +%s%N)
            duration=$(( (end - start) / 1000000 ))
            total=$((total + duration))
        done

        local avg=$((total / iterations))
        echo -e "  ${size} vectors: ${avg}ms average"
    done
}

# Main
main() {
    echo "========================================"
    echo "MerkleDB Query Benchmarks"
    echo "========================================"
    echo "Target: ${MERKLEDB_URL}"
    echo ""

    if ! curl -sf "${MERKLEDB_URL}/health/live" > /dev/null 2>&1; then
        echo -e "${RED}ERROR: MerkleDB not responding${NC}"
        exit 1
    fi

    benchmark_dimensions
    echo ""

    benchmark_filtered_queries
    echo ""

    benchmark_load_impact
    echo ""

    benchmark_database_size
    echo ""

    echo "========================================"
    echo -e "${GREEN}✓ Benchmarking complete${NC}"
    echo "========================================"
}

main
