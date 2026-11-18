// Grafana k6 distributed load test: 50,000 concurrent VUs (design.md Gate 15/16).
// Thresholds follow design.md Appendix A: p50<=240, p95<=450, p99<=590 (stricter KPI values win over the spec's 250/600).
// Run (single node, scaled):  k6 run -e BASE=http://localhost:8000 -e PEAK=500 load_tests/k6_50k_load.js
// Run (distributed, k6-operator): 50K VUs split across >= 10 runners; see README "Load testing".
import http from 'k6/http';
import { check, sleep } from 'k6';
import { Counter } from 'k6/metrics';

const BASE = __ENV.BASE || 'http://localhost:8000';
const PEAK = parseInt(__ENV.PEAK || '50000');
const THINK = parseFloat(__ENV.THINK || '1');          // spec uses sleep(1); see design.md Appendix A #9 (Little's law)
const leaks = new Counter('tenant_leaks');

const QUERIES = [
  'Verify batch LOT-88001 safety protocol and return triage status',   // spec query
  'What is the unit price of SKU-007?',
  'Who signs off on big purchases of the Tundra Gauze?',
  'What is the shelf life of SKU-010?',
  'Summarize policy POL-03.',
];

export const options = {
  stages: [
    { duration: '2m', target: Math.round(PEAK * 0.2) },  // ramp to 20% (10K at PEAK=50000)
    { duration: '3m', target: PEAK },                    // ramp to peak concurrent VUs
    { duration: '5m', target: PEAK },                    // sustain peak
    { duration: '2m', target: 0 },                       // graceful ramp-down
  ],
  thresholds: {
    'http_req_duration': ['p(50)<240', 'p(95)<450', 'p(99)<590'],   // KPI-31/32/33
    'http_req_failed': ['rate<0.0005'],                              // KPI-34: < 0.05%
    'checks': ['rate>0.999'],
    'tenant_leaks': ['count==0'],                                    // KPI-28
  },
};

export default function () {
  const vuId = __VU;
  const tenant = `tenant_${vuId % 50}`;
  const payload = JSON.stringify({
    tenant_id: tenant,
    user_id: `user_${vuId}`,
    query: QUERIES[(__ITER + vuId) % QUERIES.length],
  });
  const params = { headers: { 'Content-Type': 'application/json', 'X-Tenant-ID': tenant } };
  const res = http.post(`${BASE}/api/v1/agent/query`, payload, params);
  const ok = check(res, {
    'status is 200': (r) => r.status === 200,
    'latency within SLA (<=590ms)': (r) => r.timings.duration <= 590,
    'no cross-tenant leak': (r) => r.status !== 200 || r.json('tenant_id') === tenant,
  });
  if (res.status === 200 && res.json('tenant_id') !== tenant) leaks.add(1);
  sleep(THINK);
}
