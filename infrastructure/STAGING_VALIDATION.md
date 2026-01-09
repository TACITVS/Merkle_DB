# MerkleDB Staging Validation Guide

Complete guide for validating MerkleDB in staging environments before production deployment.

## Quick Start

```bash
cd infrastructure/validation
export MERKLEDB_URL="http://your-staging-server:4001"
export MERKLEDB_API_KEY="your_staging_api_key"
bash run_all_tests.sh
```

This runs the complete validation suite and generates a report.

## Prerequisites

### Required Tools
- **curl** - HTTP client for API testing
- **jq** - JSON processor for response parsing
- **python3** - For generating random test vectors
- **bash** - Shell for running test scripts

### Optional Tools
- **k6** - Load testing (install from https://k6.io)
- **docker** - For failover tests (requires Raft cluster)

### Environment Setup

1. Start MerkleDB in staging mode:
```bash
cd staging
docker-compose -f docker-compose.staging.yml up -d
```

2. Verify MerkleDB is running:
```bash
curl http://localhost:4001/health/live
```

3. Set environment variables:
```bash
export MERKLEDB_URL="http://localhost:4001"
export MERKLEDB_API_KEY="staging_test_key_change_in_production"
```

## Test Suites

### 1. Integration Tests

**Location:** `testing/integration/api_integration_test.sh`

**Purpose:** Validates all API endpoints and core functionality

**Tests:**
- Health checks (live, ready, detailed)
- Collection management (create, list, delete)
- Vector insertion (single, batch)
- Vector search (KNN, filtered)
- Metadata filtering
- Update and delete operations
- Metrics endpoint
- Rate limiting

**Run:**
```bash
bash testing/integration/api_integration_test.sh
```

**Expected Duration:** 2-3 minutes

**Success Criteria:** All 8 test categories pass

---

### 2. Security Audit

**Location:** `testing/security/security_audit.sh`

**Purpose:** Validates security configurations and protections

**Tests:**
- Authentication required (401/403 for unauthenticated requests)
- SQL injection protection
- XSS (Cross-Site Scripting) protection
- Rate limiting active
- TLS/HTTPS configuration
- Input validation (oversized vectors, invalid JSON)
- Security headers (X-Frame-Options, HSTS, X-Content-Type-Options)
- Path traversal protection

**Run:**
```bash
bash testing/security/security_audit.sh
```

**Expected Duration:** 1-2 minutes

**Success Criteria:** 0 failures, warnings acceptable for HTTPS in HTTP-only test environments

---

### 3. Performance Baseline

**Location:** `testing/performance/baseline.sh`

**Purpose:** Establishes performance metrics baseline for regression detection

**Measurements:**
- **Insert Latency:** Average time for single vector insert (target: <50ms)
- **Search Latency:** Average KNN search time (target: <100ms)
- **Batch Latency:** Average time for 100-vector batch insert (target: <500ms)
- **Throughput:** Concurrent queries per second (target: >100 qps)
- **Cache Hit Rate:** Percentage of cache hits (target: >0.7)
- **Memory Usage:** Current memory consumption

**Run:**
```bash
bash testing/performance/baseline.sh
```

**Expected Duration:** 3-5 minutes

**Output:** JSON baseline file with timestamp

**Success Criteria:** All metrics within target ranges

**Example Output:**
```json
{
  "timestamp": "2025-01-09T12:00:00Z",
  "performance": {
    "insert_latency_ms": 45,
    "search_latency_ms": 85,
    "batch_insert_latency_ms": 450,
    "concurrent_throughput_qps": 120,
    "cache_hit_rate": 0.75
  },
  "targets": {
    "insert_latency_ms": 50,
    "search_latency_ms": 100,
    "batch_insert_latency_ms": 500,
    "concurrent_throughput_qps": 100,
    "cache_hit_rate": 0.7
  }
}
```

---

### 4. Load Tests

**Location:** `testing/load/load_test.js`, `testing/load/stress_test.js`

**Purpose:** Validates performance under realistic and extreme load

**Load Test (load_test.js):**
- Ramps from 10 to 100 users over 9 minutes
- 70% searches, 30% inserts (typical workload)
- Thresholds: p95 < 500ms, p99 < 1s, <1% errors

**Stress Test (stress_test.js):**
- Ramps to 400 concurrent users
- Finds breaking point and maximum throughput
- Measures graceful degradation

**Run:**
```bash
# Load test
k6 run testing/load/load_test.js

# Stress test
k6 run testing/load/stress_test.js
```

**Expected Duration:**
- Load test: 9 minutes
- Stress test: 19 minutes

**Success Criteria:**
- p95 latency < 500ms
- p99 latency < 1s
- Error rate < 1%
- No crashes or memory leaks

---

### 5. Backup & Restore Validation

**Location:** `validation/backup_restore_test.sh`

**Purpose:** Validates backup and restore procedures

**Tests:**
1. **Backup Creation:** Creates tar.gz backup of Raft data
2. **Backup Verification:** Checks backup file exists and has reasonable size
3. **Data Integrity:** Validates vector count before/after
4. **Compression:** Verifies backup compression ratio

**Run:**
```bash
bash validation/backup_restore_test.sh
```

**Expected Duration:** 2-3 minutes

**Success Criteria:** All 4 tests pass

**Note:** Restore test requires manual MerkleDB restart. Follow instructions in test output.

---

### 6. Failover Tests (Optional)

**Location:** `testing/failover/raft_failover_test.sh`

**Purpose:** Validates Raft cluster resilience and failover

**Requirements:**
- Docker installed
- 3-node Raft cluster running
- Container names: `merkledb-node1`, `merkledb-node2`, `merkledb-node3`

**Tests:**
1. **Leader Election:** New leader elected within 10s after stopping current leader
2. **Write Availability:** Writes continue during failover
3. **Split Brain Prevention:** No leader elected without quorum
4. **Data Consistency:** Data replicated to all nodes

**Run:**
```bash
# Setup cluster first
docker-compose -f ../deployment/docker-compose.yml up -d

# Run tests
export NODES="node1:4001 node2:4002 node3:4003"
bash testing/failover/raft_failover_test.sh
```

**Expected Duration:** 5-8 minutes

**Success Criteria:** All 4 tests pass

---

## Complete Validation Workflow

### Step 1: Prepare Environment

```bash
# Start staging environment
cd infrastructure/staging
docker-compose -f docker-compose.staging.yml up -d

# Wait for startup
sleep 10

# Verify health
curl http://localhost:4001/health/ready
```

### Step 2: Run Complete Validation

```bash
cd ../validation
export MERKLEDB_URL="http://localhost:4001"
export MERKLEDB_API_KEY="staging_test_key_change_in_production"

# Run all tests
bash run_all_tests.sh
```

### Step 3: Review Results

The validation suite generates a report file: `validation_report_YYYYMMDD_HHMMSS.txt`

**Example Report:**
```
MerkleDB Staging Validation Report
Generated: Thu Jan  9 12:00:00 PST 2025
Environment: http://localhost:4001

========================================
Test Results Summary
========================================

1. Integration Tests: ✓ PASSED
2. Security Audit: ✓ PASSED
3. Performance Baseline: ✓ PASSED
4. Load Tests: ✓ PASSED
5. Backup Validation: ✓ PASSED
6. Failover Tests: ⊘ SKIPPED

========================================
Overall Status
========================================

Total Passed: 5
Total Failed: 0
Total Skipped: 1

✓ ALL VALIDATIONS PASSED

MerkleDB is ready for production deployment
```

### Step 4: Production Checklist

Before deploying to production, verify:

- [ ] All validation tests passed
- [ ] Performance metrics within targets
- [ ] Security audit passed (0 failures)
- [ ] Load tests show acceptable latency under 100+ concurrent users
- [ ] Backup/restore procedure validated
- [ ] Failover tests passed (if running Raft cluster)
- [ ] Monitoring configured (Prometheus + Grafana)
- [ ] Alerts configured (Alertmanager)
- [ ] SSL/TLS certificates configured
- [ ] Production API keys generated (not default staging keys)
- [ ] Rate limiting configured appropriately
- [ ] Backup schedule configured (daily recommended)
- [ ] Disaster recovery plan documented

---

## Troubleshooting

### Integration Tests Failing

**Symptom:** Tests fail with connection errors

**Fix:**
```bash
# Check MerkleDB is running
curl http://localhost:4001/health/live

# Check logs
docker logs merkledb-staging

# Restart if needed
docker-compose restart
```

### Performance Below Targets

**Symptom:** Latency exceeds targets or throughput too low

**Diagnosis:**
1. Check system resources: `docker stats`
2. Check memory usage: `curl http://localhost:4001/health/detailed | jq .system.memory_mb`
3. Check cache hit rate: `curl http://localhost:4001/health/detailed | jq .metrics`

**Common Causes:**
- Insufficient memory allocation
- CPU throttling
- Disk I/O bottleneck
- Cache size too small

**Fix:**
```bash
# Increase memory limit in docker-compose.yml
services:
  merkledb:
    deploy:
      resources:
        limits:
          memory: 4G  # Increase from 2G
```

### Security Audit Warnings

**Symptom:** Warnings about HTTPS, security headers

**Expected Behavior:** Warnings are acceptable in HTTP-only staging environments

**Production Fix:**
- Configure SSL/TLS with valid certificates
- Add security headers in Nginx/load balancer
- Enable HSTS for HTTPS enforcement

### Load Tests Failing

**Symptom:** High error rates or timeouts during k6 tests

**Diagnosis:**
```bash
# Check if rate limiting is too aggressive
curl -I http://localhost:4001/health/live | grep -i rate

# Check connection limits
netstat -an | grep 4001 | wc -l
```

**Fix:**
- Adjust rate limits in configuration
- Increase connection pool size
- Scale horizontally (add nodes)

### Backup Tests Failing

**Symptom:** Backup creation fails or restore doesn't work

**Common Issues:**
1. Insufficient disk space
2. Permission denied on backup directory
3. Raft data directory not found

**Fix:**
```bash
# Check disk space
df -h

# Create backup directory with correct permissions
mkdir -p /backup/merkledb
chmod 755 /backup/merkledb

# Verify Raft data directory exists
ls -la /var/lib/merkledb/raft/
```

---

## Continuous Validation

### Automated Nightly Runs

Setup cron job for nightly validation:

```bash
# Edit crontab
crontab -e

# Add nightly validation at 2 AM
0 2 * * * cd /path/to/infrastructure/validation && bash run_all_tests.sh >> /var/log/merkledb_validation.log 2>&1
```

### Monitoring Validation Results

Setup alerts for validation failures:

```yaml
# alertmanager config
routes:
  - match:
      job: merkledb_validation
      severity: critical
    receiver: ops-team
```

### Regression Detection

Compare baseline metrics over time:

```bash
# Save baselines with timestamps
bash testing/performance/baseline.sh
# Creates: baseline_YYYYMMDD_HHMMSS.json

# Compare with previous baseline
python3 compare_baselines.py baseline_20250109.json baseline_20250108.json
```

---

## Contact & Support

For issues or questions:
- GitHub Issues: https://github.com/TACITVS/merkle_db/issues
- Documentation: https://github.com/TACITVS/merkle_db/docs

---

## Appendix: Performance Targets

| Metric | Target | Excellent | Poor |
|--------|--------|-----------|------|
| Insert Latency | <50ms | <30ms | >100ms |
| Search Latency | <100ms | <50ms | >200ms |
| Batch Insert (100) | <500ms | <300ms | >1000ms |
| Throughput | >100 qps | >200 qps | <50 qps |
| Cache Hit Rate | >0.7 | >0.85 | <0.5 |
| Memory Usage | Stable | Stable | Growing |
| Error Rate | <1% | <0.1% | >5% |
| P95 Latency | <500ms | <300ms | >1000ms |
| P99 Latency | <1000ms | <600ms | >2000ms |

---

**Version:** 1.0
**Last Updated:** 2025-01-09
**Maintainer:** MerkleDB Team
