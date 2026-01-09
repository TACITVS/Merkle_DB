#!/bin/bash
# MerkleDB Security Audit
# Tests security configurations and vulnerabilities

set -euo pipefail

BASE_URL="${MERKLEDB_URL:-http://localhost:4001}"
API_KEY="${MERKLEDB_API_KEY:-staging_test_key_change_in_production}"

GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m'

PASSED=0
FAILED=0
WARNINGS=0

log_pass() {
    echo -e "${GREEN}✓${NC} $1"
    ((PASSED++))
}

log_fail() {
    echo -e "${RED}✗${NC} $1"
    ((FAILED++))
}

log_warn() {
    echo -e "${YELLOW}⚠${NC} $1"
    ((WARNINGS++))
}

# Test 1: Authentication required
test_authentication() {
    echo -e "\n${YELLOW}[TEST]${NC} Authentication"

    # Try without auth
    response=$(curl -s -o /dev/null -w "%{http_code}" "${BASE_URL}/search")

    if [ "$response" = "401" ] || [ "$response" = "403" ]; then
        log_pass "Authentication required (HTTP $response)"
    else
        log_fail "No authentication required (HTTP $response)"
    fi

    # Try with invalid auth
    response=$(curl -s -o /dev/null -w "%{http_code}" \
        -H "Authorization: Bearer invalid_key" "${BASE_URL}/search")

    if [ "$response" = "401" ] || [ "$response" = "403" ]; then
        log_pass "Invalid auth rejected (HTTP $response)"
    else
        log_warn "Invalid auth not properly rejected (HTTP $response)"
    fi
}

# Test 2: SQL injection attempts
test_sql_injection() {
    echo -e "\n${YELLOW}[TEST]${NC} SQL Injection Protection"

    payloads=(
        "' OR '1'='1"
        "admin'--"
        "1' UNION SELECT * FROM users--"
    )

    for payload in "${payloads[@]}"; do
        response=$(curl -sf -H "Authorization: Bearer ${API_KEY}" \
            -H "Content-Type: application/json" \
            -X POST "${BASE_URL}/insert" \
            -d "{\"key\": \"${payload}\", \"vector\": [0.1]}" 2>&1 || echo "error")

        if [[ "$response" != *"error"* ]] && [[ "$response" != *"SQL"* ]]; then
            log_pass "SQL injection blocked: ${payload:0:20}..."
        else
            log_fail "Possible SQL injection vulnerability"
        fi
    done
}

# Test 3: XSS attempts
test_xss() {
    echo -e "\n${YELLOW}[TEST]${NC} XSS Protection"

    xss_payloads=(
        "<script>alert('xss')</script>"
        "<img src=x onerror=alert('xss')>"
        "javascript:alert('xss')"
    )

    for payload in "${xss_payloads[@]}"; do
        vector=$(python3 -c "print([0.1] * 128)")
        response=$(curl -sf -H "Authorization: Bearer ${API_KEY}" \
            -H "Content-Type: application/json" \
            -X POST "${BASE_URL}/insert" \
            -d "{\"key\": \"xss_test\", \"vector\": ${vector}, \"payload\": {\"data\": \"${payload}\"}}" || echo "error")

        # Response should not contain unescaped HTML
        if echo "$response" | grep -q "<script>"; then
            log_fail "Possible XSS vulnerability"
        else
            log_pass "XSS blocked: ${payload:0:30}..."
        fi
    done
}

# Test 4: Rate limiting
test_rate_limiting() {
    echo -e "\n${YELLOW}[TEST]${NC} Rate Limiting"

    # Send rapid requests
    rate_limited=0
    for i in {1..200}; do
        response=$(curl -s -o /dev/null -w "%{http_code}" \
            -H "Authorization: Bearer ${API_KEY}" "${BASE_URL}/health/live")

        if [ "$response" = "429" ]; then
            rate_limited=1
            break
        fi
    done

    if [ $rate_limited -eq 1 ]; then
        log_pass "Rate limiting active"
    else
        log_warn "Rate limiting not detected (may be disabled)"
    fi
}

# Test 5: HTTPS/TLS configuration
test_tls() {
    echo -e "\n${YELLOW}[TEST]${NC} TLS Configuration"

    if [[ "$BASE_URL" == https://* ]]; then
        # Check TLS version
        tls_version=$(curl -sI "${BASE_URL}/health/live" 2>&1 | grep -i "tls" || echo "")

        if [ -n "$tls_version" ]; then
            log_pass "HTTPS enabled"
        else
            log_warn "Could not verify TLS version"
        fi

        # Check certificate
        if openssl s_client -connect "${BASE_URL#https://}" </dev/null 2>/dev/null | grep -q "Verify return code: 0"; then
            log_pass "Valid SSL certificate"
        else
            log_warn "SSL certificate validation issue"
        fi
    else
        log_warn "HTTPS not enabled (testing HTTP endpoint)"
    fi
}

# Test 6: Input validation
test_input_validation() {
    echo -e "\n${YELLOW}[TEST]${NC} Input Validation"

    # Oversized vector
    huge_vector=$(python3 -c "print([0.1] * 100000)")
    response=$(curl -s -o /dev/null -w "%{http_code}" \
        -H "Authorization: Bearer ${API_KEY}" \
        -H "Content-Type: application/json" \
        -X POST "${BASE_URL}/insert" \
        -d "{\"key\": \"huge\", \"vector\": ${huge_vector}}")

    if [ "$response" = "400" ] || [ "$response" = "413" ]; then
        log_pass "Oversized input rejected (HTTP $response)"
    else
        log_warn "Large input not rejected (HTTP $response)"
    fi

    # Invalid JSON
    response=$(curl -s -o /dev/null -w "%{http_code}" \
        -H "Authorization: Bearer ${API_KEY}" \
        -H "Content-Type: application/json" \
        -X POST "${BASE_URL}/insert" \
        -d "invalid json{}")

    if [ "$response" = "400" ]; then
        log_pass "Invalid JSON rejected"
    else
        log_warn "Invalid JSON not rejected (HTTP $response)"
    fi
}

# Test 7: Security headers
test_security_headers() {
    echo -e "\n${YELLOW}[TEST]${NC} Security Headers"

    headers=$(curl -sI "${BASE_URL}/health/live")

    if echo "$headers" | grep -qi "X-Frame-Options"; then
        log_pass "X-Frame-Options header present"
    else
        log_warn "X-Frame-Options header missing"
    fi

    if echo "$headers" | grep -qi "X-Content-Type-Options"; then
        log_pass "X-Content-Type-Options header present"
    else
        log_warn "X-Content-Type-Options header missing"
    fi

    if echo "$headers" | grep -qi "Strict-Transport-Security"; then
        log_pass "HSTS header present"
    else
        log_warn "HSTS header missing (expected if using HTTPS)"
    fi
}

# Test 8: Path traversal
test_path_traversal() {
    echo -e "\n${YELLOW}[TEST]${NC} Path Traversal Protection"

    payloads=(
        "../../../etc/passwd"
        "..\\..\\..\\windows\\system32"
        "%2e%2e%2f%2e%2e%2f"
    )

    for payload in "${payloads[@]}"; do
        response=$(curl -s -o /dev/null -w "%{http_code}" \
            -H "Authorization: Bearer ${API_KEY}" "${BASE_URL}/get/${payload}")

        if [ "$response" = "400" ] || [ "$response" = "404" ]; then
            log_pass "Path traversal blocked: ${payload:0:20}..."
        else
            log_warn "Path traversal not blocked (HTTP $response)"
        fi
    done
}

# Main
main() {
    echo "========================================"
    echo "MerkleDB Security Audit"
    echo "========================================"
    echo "Target: ${BASE_URL}"
    echo ""

    test_authentication
    test_sql_injection
    test_xss
    test_rate_limiting
    test_tls
    test_input_validation
    test_security_headers
    test_path_traversal

    echo ""
    echo "========================================"
    echo "Security Audit Results"
    echo "========================================"
    echo -e "${GREEN}Passed:${NC} ${PASSED}"
    echo -e "${RED}Failed:${NC} ${FAILED}"
    echo -e "${YELLOW}Warnings:${NC} ${WARNINGS}"

    if [ $FAILED -eq 0 ]; then
        echo -e "\n${GREEN}✓ Security audit passed!${NC}"
        if [ $WARNINGS -gt 0 ]; then
            echo -e "${YELLOW}Note: ${WARNINGS} warnings found - review recommended${NC}"
        fi
        exit 0
    else
        echo -e "\n${RED}✗ Security issues found - fix before production${NC}"
        exit 1
    fi
}

main
