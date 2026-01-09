#!/bin/bash
# MerkleDB API Integration Tests
# Tests all critical API endpoints in staging environment

set -euo pipefail

# Configuration
BASE_URL="${MERKLEDB_URL:-http://localhost:4001}"
API_KEY="${MERKLEDB_API_KEY:-staging_test_key_change_in_production}"
TEST_COLLECTION="test_$(date +%s)"

# Colors
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m'

# Counters
TESTS_PASSED=0
TESTS_FAILED=0

# Test helper functions
log_test() {
    echo -e "\n${YELLOW}[TEST]${NC} $1"
}

log_pass() {
    echo -e "${GREEN}✓${NC} $1"
    ((TESTS_PASSED++))
}

log_fail() {
    echo -e "${RED}✗${NC} $1"
    ((TESTS_FAILED++))
}

curl_api() {
    curl -s -H "Authorization: Bearer ${API_KEY}" -H "Content-Type: application/json" "$@"
}

# Test 1: Health checks
test_health_checks() {
    log_test "Health Check Endpoints"

    # Liveness
    if curl -sf "${BASE_URL}/health/live" > /dev/null; then
        log_pass "Liveness check"
    else
        log_fail "Liveness check failed"
    fi

    # Readiness
    if curl -sf "${BASE_URL}/health/ready" > /dev/null; then
        log_pass "Readiness check"
    else
        log_fail "Readiness check failed"
    fi

    # Detailed health
    response=$(curl -sf "${BASE_URL}/health/detailed")
    if echo "$response" | jq -e '.status == "ok"' > /dev/null 2>&1; then
        log_pass "Detailed health check"
    else
        log_fail "Detailed health check failed"
    fi
}

# Test 2: Collection operations
test_collection_operations() {
    log_test "Collection Operations"

    # Create collection
    response=$(curl_api -X POST "${BASE_URL}/collections" \
        -d "{\"name\": \"${TEST_COLLECTION}\", \"dim\": 128, \"precision\": \"f64\"}")

    if echo "$response" | jq -e '.status == "ok"' > /dev/null 2>&1; then
        log_pass "Create collection"
    else
        log_fail "Create collection: $response"
    fi

    # List collections
    response=$(curl_api -X GET "${BASE_URL}/collections")
    if echo "$response" | jq -e --arg name "$TEST_COLLECTION" '.collections[] | select(. == $name)' > /dev/null 2>&1; then
        log_pass "List collections"
    else
        log_fail "List collections"
    fi
}

# Test 3: Vector insertion
test_vector_insertion() {
    log_test "Vector Insertion"

    # Generate random 128-dim vector
    vector=$(python3 -c "import random; print([random.random() for _ in range(128)])")

    # Insert vector
    response=$(curl_api -X POST "${BASE_URL}/insert" \
        -d "{\"collection\": \"${TEST_COLLECTION}\", \"key\": \"test_vec_1\", \"vector\": ${vector}, \"payload\": {\"category\": \"test\"}}")

    if echo "$response" | jq -e '.status == "ok"' > /dev/null 2>&1; then
        log_pass "Insert vector"
    else
        log_fail "Insert vector: $response"
    fi

    # Batch insert
    batch_data="{\"collection\": \"${TEST_COLLECTION}\", \"vectors\": ["
    for i in {2..10}; do
        vector=$(python3 -c "import random; print([random.random() for _ in range(128)])")
        batch_data+="{\"key\": \"test_vec_${i}\", \"vector\": ${vector}},"
    done
    batch_data="${batch_data%,}]}"

    response=$(curl_api -X POST "${BASE_URL}/batch" -d "$batch_data")
    if echo "$response" | jq -e '.inserted == 9' > /dev/null 2>&1; then
        log_pass "Batch insert"
    else
        log_fail "Batch insert: $response"
    fi
}

# Test 4: Search operations
test_search_operations() {
    log_test "Search Operations"

    # Generate query vector
    query=$(python3 -c "import random; print([random.random() for _ in range(128)])")

    # KNN search
    response=$(curl_api -X POST "${BASE_URL}/search" \
        -d "{\"collection\": \"${TEST_COLLECTION}\", \"vector\": ${query}, \"k\": 5}")

    if echo "$response" | jq -e '.results | length > 0' > /dev/null 2>&1; then
        log_pass "KNN search"
    else
        log_fail "KNN search: $response"
    fi

    # Search with threshold
    response=$(curl_api -X POST "${BASE_URL}/search" \
        -d "{\"collection\": \"${TEST_COLLECTION}\", \"vector\": ${query}, \"k\": 5, \"threshold\": 0.5}")

    if echo "$response" | jq -e 'has("results")' > /dev/null 2>&1; then
        log_pass "Threshold search"
    else
        log_fail "Threshold search"
    fi
}

# Test 5: Metadata filtering
test_metadata_filtering() {
    log_test "Metadata Filtering"

    # Insert vectors with metadata
    for i in {1..5}; do
        vector=$(python3 -c "import random; print([random.random() for _ in range(128)])")
        curl_api -X POST "${BASE_URL}/insert" \
            -d "{\"collection\": \"${TEST_COLLECTION}\", \"key\": \"meta_vec_${i}\", \"vector\": ${vector}, \"payload\": {\"category\": \"A\", \"score\": ${i}}}" > /dev/null
    done

    query=$(python3 -c "import random; print([random.random() for _ in range(128)])")

    # Filter by category
    response=$(curl_api -X POST "${BASE_URL}/search" \
        -d "{\"collection\": \"${TEST_COLLECTION}\", \"vector\": ${query}, \"k\": 10, \"filter\": {\"category\": \"A\"}}")

    if echo "$response" | jq -e '.results | length > 0' > /dev/null 2>&1; then
        log_pass "Metadata filtering"
    else
        log_fail "Metadata filtering"
    fi
}

# Test 6: Update and delete
test_update_delete() {
    log_test "Update and Delete Operations"

    vector=$(python3 -c "import random; print([random.random() for _ in range(128)])")

    # Update vector
    response=$(curl_api -X PUT "${BASE_URL}/update" \
        -d "{\"collection\": \"${TEST_COLLECTION}\", \"key\": \"test_vec_1\", \"vector\": ${vector}}")

    if echo "$response" | jq -e '.status == "ok"' > /dev/null 2>&1; then
        log_pass "Update vector"
    else
        log_fail "Update vector"
    fi

    # Delete vector
    response=$(curl_api -X DELETE "${BASE_URL}/delete" \
        -d "{\"collection\": \"${TEST_COLLECTION}\", \"key\": \"test_vec_1\"}")

    if echo "$response" | jq -e '.status == "ok"' > /dev/null 2>&1; then
        log_pass "Delete vector"
    else
        log_fail "Delete vector"
    fi
}

# Test 7: Metrics endpoint
test_metrics() {
    log_test "Metrics Endpoint"

    response=$(curl -sf "${BASE_URL}/metrics")

    if echo "$response" | grep -q "merkledb_queries_total"; then
        log_pass "Prometheus metrics"
    else
        log_fail "Prometheus metrics"
    fi
}

# Test 8: Rate limiting (optional - may fail legitimately)
test_rate_limiting() {
    log_test "Rate Limiting (Optional)"

    # Send rapid requests
    for i in {1..150}; do
        curl -sf -H "Authorization: Bearer ${API_KEY}" "${BASE_URL}/health/live" > /dev/null 2>&1 || true
    done

    # Next request should be rate limited
    http_code=$(curl -s -o /dev/null -w "%{http_code}" -H "Authorization: Bearer ${API_KEY}" "${BASE_URL}/health/live")

    if [ "$http_code" = "429" ]; then
        log_pass "Rate limiting (got 429)"
    else
        echo -e "${YELLOW}ℹ${NC} Rate limiting not triggered or disabled (got HTTP $http_code)"
    fi
}

# Cleanup
cleanup() {
    log_test "Cleanup"

    # Drop test collection
    response=$(curl_api -X DELETE "${BASE_URL}/collections/${TEST_COLLECTION}")

    if echo "$response" | jq -e '.status == "ok"' > /dev/null 2>&1; then
        log_pass "Drop test collection"
    else
        log_fail "Cleanup failed"
    fi
}

# Main test execution
main() {
    echo "========================================"
    echo "MerkleDB API Integration Tests"
    echo "========================================"
    echo "Target: ${BASE_URL}"
    echo "Collection: ${TEST_COLLECTION}"
    echo ""

    test_health_checks
    test_collection_operations
    test_vector_insertion
    test_search_operations
    test_metadata_filtering
    test_update_delete
    test_metrics
    test_rate_limiting
    cleanup

    echo ""
    echo "========================================"
    echo "Test Results"
    echo "========================================"
    echo -e "${GREEN}Passed:${NC} ${TESTS_PASSED}"
    echo -e "${RED}Failed:${NC} ${TESTS_FAILED}"

    if [ $TESTS_FAILED -eq 0 ]; then
        echo -e "\n${GREEN}✓ All tests passed!${NC}"
        exit 0
    else
        echo -e "\n${RED}✗ Some tests failed${NC}"
        exit 1
    fi
}

# Run tests
main
