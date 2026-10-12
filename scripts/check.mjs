import assert from 'node:assert/strict';
import { readFileSync, existsSync } from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import vm from 'node:vm';
import { createRequire } from 'node:module';
import { createHash } from 'node:crypto';
const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const { selectPapers, parseState, defaults, escapeHTML, hasResource } = createRequire(import.meta.url)('../app.js');
const sandbox = { window: {} };
vm.runInNewContext(readFileSync(path.join(root, 'data.js'), 'utf8'), sandbox);
const data = JSON.parse(JSON.stringify(sandbox.window.ATLAS_DATA));
const source = readFileSync(path.join(root, 'README.md'), 'utf8');
const bibliography = source.slice(source.search(/^\s*\*\s+\*\*/m));
const originalResources = [...bibliography.matchAll(/\[([^\[\]]+)\]\((https?:\/\/[^\s)]+)\)/g)].map(m => ({ label: m[1], url: m[2] }));
const resourceKey = link => `${link.label}\n${link.url}`;
const originalResourcesSet = new Set(originalResources.map(resourceKey));
const retainedResources = new Set(data.papers.flatMap(p => p.links.map(resourceKey)));
const retainedLinks = new Set(data.papers.flatMap(p => p.links.map(l => l.url)));
assert.equal(data.tracks.length, 10);
assert.equal(data.sourceCount, (source.match(/^\s*\*\s+\*\*/gm) || []).length);
assert.equal(data.papers.reduce((n,p) => n + p.tracks.length, 0), data.sourceCount, 'Every source category entry must be retained');
assert.deepEqual(retainedResources, originalResourcesSet, 'Every source resource label and URL must be preserved');
assert.equal(new Set(data.papers.map(p => p.id)).size, data.papers.length);
for (const p of data.papers) {
  assert.ok(p.title && p.venue && p.year >= 2000 && p.year <= 2030);
  assert.ok(p.links.every(l => l.label.trim() && /^https?:\/\//.test(l.url)), `Malformed resource link: ${p.title}`);
  assert.ok(p.tracks.every(id => data.tracks.some(t => t.id === id)));
}
const v2x = selectPapers(data, { ...defaults, q: 'v2x-r', year: '2025', resource: 'code', track: 'dataset' });
assert.ok(v2x.some(p => p.title.startsWith('V2X-R: ')), 'Combined filters should include V2X-R even when V2X-Radar also matches');
assert.ok(v2x.every(p => p.year === 2025 && p.tracks.includes('dataset') && hasResource(p, 'code')), 'Every combined-filter result must satisfy the year, track and resource constraints');
const pendingProject = { links: [{ label: 'paper', url: 'https://example.org/paper' }, { label: 'arXiv', url: 'https://arxiv.org/abs/2512.13107' }, { label: 'project (code pending)', url: 'https://example.org/project' }] };
assert.equal(hasResource(pendingProject, 'code'), false, 'A project with pending code must not count as available code');
assert.equal(hasResource(pendingProject, 'data'), false);
const combinedResource = { links: [{ label: 'code&data', url: 'https://example.org/resources' }] };
assert.equal(hasResource(combinedResource, 'code'), true);
assert.equal(hasResource(combinedResource, 'data'), true);
const restoration = data.papers.find(p => p.title === 'Weather-Robust LiDAR Perception: Point Cloud Restoration from Adverse Weather');
assert.ok(restoration.tracks.includes('dataset') && restoration.tracks.includes('lidarde'), 'Repeated restoration record retains both categories');
assert.equal(selectPapers(data, { ...defaults, q: 'no-such-paper-983426' }).length, 0);
assert.equal(selectPapers(data, { ...defaults, q: '  V2X-R  ' }).length, selectPapers(data, { ...defaults, q: 'v2x r' }).length);
const oldest = selectPapers(data, { ...defaults, sort: 'oldest' });
assert.equal(oldest[0].year, Math.min(...data.papers.map(p => p.year)));
const saved = selectPapers(data, { ...defaults, collection: 'saved' }, new Set([v2x[0].id]));
assert.equal(saved.length, 1); assert.equal(saved[0].id, v2x[0].id);
assert.equal(selectPapers(data, { ...defaults, collection: 'saved' }).length, 0);
assert.deepEqual(parseState(new URLSearchParams('track=wrong&year=9999&sort=wrong&resource=no&view=wrong'), data), defaults);
assert.equal(escapeHTML('<img src=x onerror="oops">'), '&lt;img src=x onerror=&quot;oops&quot;&gt;');
const html = readFileSync(path.join(root, 'index.html'), 'utf8');
const ids = [...html.matchAll(/\bid="([^"]+)"/g)].map(m => m[1]);
assert.equal(ids.length, new Set(ids).size, 'DOM IDs must be unique');
for (const m of html.matchAll(/(?:src|href)="([^"#:]+)"/g)) if (!/^(http|mailto)/.test(m[1])) {
  const asset = m[1].split(/[?#]/)[0];
  assert.ok(existsSync(path.join(root, 'dist', asset)), `Missing asset: ${asset}`);
}
for (const asset of ['styles.css', 'app.js', 'scene.js', 'data.js']) {
  const version = createHash('sha256').update(readFileSync(path.join(root, asset))).digest('hex').slice(0, 12);
  assert.ok(html.includes(`"${asset}?v=${version}"`), `Missing current cache version: ${asset}`);
}
console.log(`PASS: ${data.sourceCount} source records and ${retainedLinks.size} source links retained; compound search, sorting, saved lists, URL validation, HTML safety and local assets checked.`);
