const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');
const source = fs.readFileSync(path.join(__dirname, '../webapp/dist/app.js'), 'utf8');
const code = source.slice(source.indexOf('async function readJob('), source.indexOf('\nasync function reconstruct('));
function reader(fetch) {
  const context = {fetch, AbortSignal, setTimeout: fn => setImmediate(fn), $: () => ({textContent: ''})};
  vm.createContext(context);
  vm.runInContext(code + ';this.read=readJob;', context);
  return context.read;
}
(async () => {
  let calls = 0;
  const read = reader(async (url) => {
    assert.equal(url, '/api/jobs/example');
    calls++;
    if (calls < 3) return {status: 502};
    return {status: 200, ok: true, json: async () => ({status: 'complete'})};
  });
  assert.equal((await read('example')).status, 'complete');
  assert.equal(calls, 3);
  calls = 0;
  await assert.rejects(reader(async () => {
    calls++;
    return {status: 404, ok: false, json: async () => ({error: 'Run not found'})};
  })('missing'), /Run not found/);
  assert.equal(calls, 1);
  calls = 0;
  await assert.rejects(reader(async () => {calls++; throw new TypeError('Network failure');})('offline'), /host may still be processing/);
  assert.equal(calls, 12);
  console.log('3 polling checks passed: transient recovery, permanent error, bounded retries.');
})().catch(error => {console.error(error); process.exitCode = 1;});
