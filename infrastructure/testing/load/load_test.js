// MerkleDB Load Testing with k6
// Run: k6 run --vus 50 --duration 5m load_test.js

import http from 'k6/http';
import { check, sleep } from 'k6';
import { Rate, Trend, Counter } from 'k6/metrics';

// Custom metrics
const errorRate = new Rate('errors');
const searchLatency = new Trend('search_latency');
const insertLatency = new Trend('insert_latency');
const searchSuccess = new Counter('search_success');
const insertSuccess = new Counter('insert_success');

// Configuration
const BASE_URL = __ENV.MERKLEDB_URL || 'http://localhost:4001';
const API_KEY = __ENV.MERKLEDB_API_KEY || 'staging_test_key_change_in_production';
const COLLECTION = 'load_test';
const DIM = 128;

// Test configuration - ramp up load
export const options = {
  stages: [
    { duration: '1m', target: 10 },   // Ramp up to 10 users
    { duration: '2m', target: 50 },   // Ramp up to 50 users
    { duration: '3m', target: 50 },   // Stay at 50 for 3 minutes
    { duration: '1m', target: 100 },  // Spike to 100
    { duration: '2m', target: 0 },    // Ramp down
  ],
  thresholds: {
    'http_req_duration': ['p(95)<500', 'p(99)<1000'], // 95% < 500ms, 99% < 1s
    'http_req_failed': ['rate<0.01'],  // <1% errors
    'errors': ['rate<0.05'],           // <5% errors
  },
};

// Generate random vector
function randomVector(dim) {
  const vec = [];
  for (let i = 0; i < dim; i++) {
    vec.push(Math.random());
  }
  return vec;
}

// Setup - create collection
export function setup() {
  const url = `${BASE_URL}/collections`;
  const payload = JSON.stringify({
    name: COLLECTION,
    dim: DIM,
    precision: 'f64'
  });

  const params = {
    headers: {
      'Content-Type': 'application/json',
      'Authorization': `Bearer ${API_KEY}`
    }
  };

  const res = http.post(url, payload, params);

  if (res.status !== 200) {
    console.error(`Setup failed: ${res.status} ${res.body}`);
  }

  return { collection: COLLECTION };
}

// Main test function
export default function(data) {
  const params = {
    headers: {
      'Content-Type': 'application/json',
      'Authorization': `Bearer ${API_KEY}`
    }
  };

  // 70% searches, 30% inserts (typical read-heavy workload)
  if (Math.random() < 0.7) {
    // Search test
    const query = randomVector(DIM);
    const searchPayload = JSON.stringify({
      collection: data.collection,
      vector: query,
      k: 10
    });

    const searchStart = Date.now();
    const searchRes = http.post(`${BASE_URL}/search`, searchPayload, params);
    const searchDuration = Date.now() - searchStart;

    const searchOk = check(searchRes, {
      'search status 200': (r) => r.status === 200,
      'search has results': (r) => JSON.parse(r.body).results !== undefined,
    });

    errorRate.add(!searchOk);
    searchLatency.add(searchDuration);
    if (searchOk) searchSuccess.add(1);

  } else {
    // Insert test
    const vector = randomVector(DIM);
    const key = `load_test_${__VU}_${Date.now()}`;
    const insertPayload = JSON.stringify({
      collection: data.collection,
      key: key,
      vector: vector,
      payload: { category: 'load_test', vu: __VU }
    });

    const insertStart = Date.now();
    const insertRes = http.post(`${BASE_URL}/insert`, insertPayload, params);
    const insertDuration = Date.now() - insertStart;

    const insertOk = check(insertRes, {
      'insert status 200': (r) => r.status === 200,
      'insert success': (r) => JSON.parse(r.body).status === 'ok',
    });

    errorRate.add(!insertOk);
    insertLatency.add(insertDuration);
    if (insertOk) insertSuccess.add(1);
  }

  // Small random sleep to simulate real usage
  sleep(Math.random() * 0.5);
}

// Teardown - cleanup
export function teardown(data) {
  const url = `${BASE_URL}/collections/${data.collection}`;
  const params = {
    headers: {
      'Authorization': `Bearer ${API_KEY}`
    }
  };

  http.del(url, null, params);
}
