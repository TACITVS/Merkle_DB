#!/bin/bash
# MerkleDB Complete Validation Suite
# Runs all staging validation tests in sequence

set -euo pipefail

GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

MERKLEDB_URL="${MERKLEDB_URL:-http://localhost:4001}"
API_KEY="${MERKLEDB_API_KEY:-staging_test_key_change_in_production}"

log() {
    echo -e "${GREEN}[$(date '+%H:%M:%S')]${NC} $1"
}

error() {
    echo -e "${RED}[$(date '+%H:%M:%S')] ERROR:${NC} $1"
}

section() {
    echo ""
    echo -e "${BLUE}========================================${NC}"
    echo -e "${BLUE}$1${NC}"
    echo -e "${BLUE}========================================${NC}"
    echo ""
}

# Check prerequisites
check_prerequisites() {
    section "Checking Prerequisites"

    local missing=0

    # Check curl
    if ! command -v curl &> /dev/null; then
        error "curl not found"
        ((missing++))
    else
        log "curl: ✓"
    fi

    # Check jq
    if ! command -v jq &> /dev/null; then
        error "jq not found"
        ((missing++))
    else
        log "jq: ✓"
    fi

    # Check python3
    if ! command -v python3 &> /dev/null; then
        error "python3 not found"
        ((missing++))
    else
        log "python3: ✓"
    fi

    # Check k6
    if ! command -v k6 &> /dev/null; then
        error "k6 not found (load tests will be skipped)"
        log "Install from: https://k6.io/docs/getting-started/installation"
    else
        log "k6: ✓"
    fi

    # Check MerkleDB
    if curl -sf "${MERKLEDB_URL}/health/live" > /dev/null 2>&1; then
        log "MerkleDB: ✓ (${MERKLEDB_URL})"
    else
        error "MerkleDB not responding at ${MERKLEDB_URL}"
        ((missing++))
    fi

    if [ $missing -gt 0 ]; then
        error "${missing} prerequisite(s) missing"
        exit 1
    fi

    log "All prerequisites satisfied"
}

# Run integration tests
run_integration_tests() {
    section "1. Integration Tests"

    local script="../testing/integration/api_integration_test.sh"

    if [ ! -f "$script" ]; then
        error "Integration test script not found: $script"
        return 1
    fi

    log "Running API integration tests..."
    bash "$script"
}

# Run security audit
run_security_audit() {
    section "2. Security Audit"

    local script="../testing/security/security_audit.sh"

    if [ ! -f "$script" ]; then
        error "Security audit script not found: $script"
        return 1
    fi

    log "Running security audit..."
    bash "$script"
}

# Run performance baseline
run_performance_baseline() {
    section "3. Performance Baseline"

    local script="../testing/performance/baseline.sh"

    if [ ! -f "$script" ]; then
        error "Performance baseline script not found: $script"
        return 1
    fi

    log "Establishing performance baseline..."
    OUTPUT_FILE="validation_baseline_$(date +%Y%m%d_%H%M%S).json" bash "$script"
}

# Run load tests
run_load_tests() {
    section "4. Load Tests"

    if ! command -v k6 &> /dev/null; then
        echo -e "${YELLOW}⚠ k6 not installed, skipping load tests${NC}"
        return 0
    fi

    local script="../testing/load/load_test.js"

    if [ ! -f "$script" ]; then
        error "Load test script not found: $script"
        return 1
    fi

    log "Running k6 load tests (this may take 10+ minutes)..."
    k6 run "$script"
}

# Run backup validation
run_backup_validation() {
    section "5. Backup & Restore Validation"

    local script="./backup_restore_test.sh"

    if [ ! -f "$script" ]; then
        error "Backup validation script not found: $script"
        return 1
    fi

    log "Running backup & restore validation..."
    bash "$script"
}

# Run failover tests (optional - requires Docker)
run_failover_tests() {
    section "6. Failover Tests (Optional)"

    if ! command -v docker &> /dev/null; then
        echo -e "${YELLOW}⚠ Docker not installed, skipping failover tests${NC}"
        return 0
    fi

    local script="../testing/failover/raft_failover_test.sh"

    if [ ! -f "$script" ]; then
        error "Failover test script not found: $script"
        return 1
    fi

    echo -e "${YELLOW}Failover tests require a Raft cluster setup${NC}"
    echo -e "${YELLOW}Skip? (y/n)${NC}"
    read -r skip

    if [ "$skip" = "y" ] || [ "$skip" = "Y" ]; then
        log "Skipping failover tests"
        return 0
    fi

    log "Running Raft failover tests..."
    bash "$script"
}

# Generate validation report
generate_report() {
    section "Validation Report"

    local report_file="validation_report_$(date +%Y%m%d_%H%M%S).txt"

    cat > "$report_file" <<EOF
MerkleDB Staging Validation Report
Generated: $(date)
Environment: ${MERKLEDB_URL}

========================================
Test Results Summary
========================================

1. Integration Tests: ${integration_result}
2. Security Audit: ${security_result}
3. Performance Baseline: ${performance_result}
4. Load Tests: ${load_result}
5. Backup Validation: ${backup_result}
6. Failover Tests: ${failover_result}

========================================
Overall Status
========================================

Total Passed: ${passed}
Total Failed: ${failed}
Total Skipped: ${skipped}

EOF

    if [ $failed -eq 0 ]; then
        echo -e "${GREEN}✓ ALL VALIDATIONS PASSED${NC}" | tee -a "$report_file"
        echo "" | tee -a "$report_file"
        echo "MerkleDB is ready for production deployment" | tee -a "$report_file"
    else
        echo -e "${RED}✗ VALIDATION FAILURES DETECTED${NC}" | tee -a "$report_file"
        echo "" | tee -a "$report_file"
        echo "Review failed tests before production deployment" | tee -a "$report_file"
    fi

    echo "" | tee -a "$report_file"
    echo "Full report saved to: $report_file" | tee -a "$report_file"
}

# Main execution
main() {
    echo "========================================"
    echo "MerkleDB Complete Validation Suite"
    echo "========================================"
    echo "Environment: ${MERKLEDB_URL}"
    echo "Started: $(date)"
    echo ""

    passed=0
    failed=0
    skipped=0

    integration_result="NOT RUN"
    security_result="NOT RUN"
    performance_result="NOT RUN"
    load_result="NOT RUN"
    backup_result="NOT RUN"
    failover_result="NOT RUN"

    # Check prerequisites
    check_prerequisites || exit 1

    # Run test suites
    if run_integration_tests; then
        integration_result="✓ PASSED"
        ((passed++))
    else
        integration_result="✗ FAILED"
        ((failed++))
    fi

    if run_security_audit; then
        security_result="✓ PASSED"
        ((passed++))
    else
        security_result="✗ FAILED"
        ((failed++))
    fi

    if run_performance_baseline; then
        performance_result="✓ PASSED"
        ((passed++))
    else
        performance_result="✗ FAILED"
        ((failed++))
    fi

    if run_load_tests; then
        load_result="✓ PASSED"
        ((passed++))
    elif [ $? -eq 0 ]; then
        load_result="⊘ SKIPPED"
        ((skipped++))
    else
        load_result="✗ FAILED"
        ((failed++))
    fi

    if run_backup_validation; then
        backup_result="✓ PASSED"
        ((passed++))
    else
        backup_result="✗ FAILED"
        ((failed++))
    fi

    if run_failover_tests; then
        failover_result="✓ PASSED"
        ((passed++))
    elif [ $? -eq 0 ]; then
        failover_result="⊘ SKIPPED"
        ((skipped++))
    else
        failover_result="✗ FAILED"
        ((failed++))
    fi

    # Generate report
    generate_report

    # Exit with appropriate code
    if [ $failed -gt 0 ]; then
        exit 1
    else
        exit 0
    fi
}

main
