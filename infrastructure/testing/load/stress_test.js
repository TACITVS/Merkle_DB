// MerkleDB Stress Test - Push to limits
// Run: k6 run stress_test.js

import http from 'k6/http';
import { check, sleep } from 'k6';

const BASE_URL = __ENV.MERKLEDB_URL || 'http://localhost:4001';
const API_KEY = __ENV.MERKLEDB_API_KEY || 'staging_test_key_change_in_production';
const DIM = 128;

// Stress test - ramp to breaking point
export const options = {
  stages: [
    { duration: '2m', target: 100 },   // Ramp to 100 users
    { duration: '5m', target: 200 },   // Ramp to 200 users
    { duration: '5m', target: 300 },   // Ramp to 300 users
    { duration: '5m', target: 400 },   // Ramp to 400 users (stress)
    { duration: '2m', target: 0 },     // Recovery
  ],
};

function randomVector(dim) {
  return Array.from({length: dim}, () => Math.random());
}

export default function() {
  const params = {
    headers: {
      'Content-Type': 'application/json',
      'Authorization': `Bearer ${API_KEY}`
    },
    timeout: '30s'
  };

  const query = randomVector(DIM);
  const payload = JSON.stringify({
    vector: query,
    k: 10
  });

  const res = http.post(`${BASE_URL}/search`, payload, params);

  check(res, {
    'status 200 or 429': (r) => r.status === 200 || r.status === 429,
  });

  sleep(0.1);
}
