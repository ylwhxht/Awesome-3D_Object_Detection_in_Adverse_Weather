import { readFileSync, writeFileSync, mkdirSync, copyFileSync, cpSync } from 'node:fs';
import { createHash } from 'node:crypto';
import { fileURLToPath } from 'node:url';
import path from 'node:path';

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const readme = readFileSync(path.join(root, 'README.md'), 'utf8');
const tracks = [
  { id: 'surveys', name: 'Surveys & reviews', short: 'Survey', icon: 'book', color: 'sand', description: 'Get oriented. Start with the big picture of all-weather perception.' },
  { id: 'dataset', name: 'Datasets & benchmarks', short: 'Dataset', icon: 'database', color: 'green', description: 'Find real and synthetic data for the conditions that matter.' },
  { id: 'eva', name: 'Weather analysis', short: 'Analysis', icon: 'chart', color: 'sand', description: 'Understand how weather changes sensor measurements.' },
  { id: 'sim', name: 'Weather simulation', short: 'Simulation', icon: 'cloud', color: 'blue', description: 'Bring rare weather into training with physics and generation.' },
  { id: 'lidarde', name: 'LiDAR restoration', short: 'Restoration', icon: 'sparkles', color: 'purple', description: 'Recover useful geometry from weather-corrupted point clouds.' },
  { id: 'lidar', name: 'LiDAR & camera detection', short: 'Detection', icon: 'scan', color: 'green', description: 'Explore robust detection and adaptive multimodal perception.' },
  { id: 'radar', name: '4D radar & camera', short: '4D radar', icon: 'radar', color: 'blue', description: 'Use range, motion, and radar spectra for reliable detection.' },
  { id: 'l3r', name: 'LiDAR + 3D radar', short: '3D radar fusion', icon: 'layers', color: 'purple', description: 'Combine complementary LiDAR and radar measurements.' },
  { id: 'l4r', name: 'LiDAR + 4D radar', short: '4D radar fusion', icon: 'box', color: 'green', description: 'Connect detailed geometry with rich radar representations.' },
  { id: 'icp', name: 'Cooperative perception', short: 'Cooperation', icon: 'network', color: 'sand', description: 'See together through vehicle and infrastructure collaboration.' }
];
const clean = s => s.replace(/<[^>]*>/g, ' ').replace(/\s+/g, ' ').replace(/^,/, '').trim();
const records = [];
let track = '', headingYear = 0, current = null;
function flush() {
  if (!current) return;
  const links = [...current.lines.join('\n').matchAll(/\[([^\[\]]+)\]\((https?:\/\/[^\s)]+)\)/g)].map(m => ({ label: m[1], url: m[2] }));
  const metadata = clean(current.lines.filter(s => !s.includes('](http')).join(' '));
  const years = [...metadata.matchAll(/\b(19\d{2}|20\d{2})\b/g)];
  const year = years.length ? Number(years[years.length - 1][1]) : current.headingYear;
  let title = clean(current.rawTitle.replace(/^\[[^\]]+\]:?\s*/, ''));
  const alias = current.rawTitle.match(/^\[([^\]]+)\]/)?.[1] || '';
  if (!title) title = alias || clean(current.rawTitle);
  const sourceUrl = links.find(l => l.label === 'paper')?.url || links[0]?.url;
  if (!sourceUrl) throw new Error(`Missing link: ${title}`);
  const key = `${title.toLowerCase().replace(/[^a-z0-9]/g, '')}|${year}|${sourceUrl}`;
  const previous = records.find(p => p.key === key);
  if (previous) {
    if (!previous.tracks.includes(current.track)) previous.tracks.push(current.track);
    if (alias && !previous.aliases.includes(alias)) previous.aliases.push(alias);
    for (const link of links) if (!previous.links.some(l => l.url === link.url && l.label === link.label)) previous.links.push(link);
  } else {
    records.push({ id: createHash('sha1').update(key).digest('hex').slice(0, 12), key, title, aliases: alias ? [alias] : [], year, venue: metadata || String(year), tracks: [current.track], links });
  }
  current = null;
}
for (const line of readme.split(/\r?\n/)) {
  const section = line.match(/^##\s+<a id="([^"]+)"/);
  if (section) { flush(); track = tracks.some(t => t.id === section[1]) ? section[1] : ''; continue; }
  const year = line.match(/^###\s+\*(\d{4})\*/);
  if (year) { flush(); headingYear = Number(year[1]); continue; }
  const entry = line.match(/^\s*\*\s+\*\*(.*?)\*\*(.*)$/);
  if (entry && track) { flush(); current = { rawTitle: entry[1], lines: [entry[2]], track, headingYear }; continue; }
  if (current && line.trim() !== '----') current.lines.push(line);
}
flush();
const sourceCount = (readme.match(/^\s*\*\s+\*\*/gm) || []).length;
const categoryEntries = records.reduce((n, p) => n + p.tracks.length, 0);
if (categoryEntries !== sourceCount) throw new Error(`Source entries lost: ${categoryEntries}/${sourceCount}`);
for (const p of records) delete p.key;
const updates = [...readme.matchAll(/^\* \[(\d{4}\.\d{1,2}\.\d{1,2})\] (.*)$/gm)].map(m => ({ date: m[1], text: m[2] }));
const data = { tracks, papers: records, updates, sourceCount, source: 'https://github.com/ylwhxht/Awesome-3D_Object_Detection_in_Adverse_Weather', updated: updates.at(-1)?.date };
writeFileSync(path.join(root, 'data.js'), `/* Generated from README.md. Run node scripts/build.mjs to refresh. */\nwindow.ATLAS_DATA = ${JSON.stringify(data, null, 2)};\n`);
const htmlPath = path.join(root, 'index.html');
const html = readFileSync(htmlPath, 'utf8').replace(/((?:src|href)=")(styles\.css|app\.js|scene\.js|data\.js)(?:\?v=[^"]*)?"/g, (_, prefix, asset) => {
  const version = createHash('sha256').update(readFileSync(path.join(root, asset))).digest('hex').slice(0, 12);
  return `${prefix}${asset}?v=${version}"`;
});
writeFileSync(htmlPath, html);
mkdirSync(path.join(root, 'dist', 'images'), { recursive: true });
for (const file of ['index.html', 'styles.css', 'app.js', 'scene.js', 'data.js', 'favicon.svg', '.nojekyll']) copyFileSync(path.join(root, file), path.join(root, 'dist', file));
copyFileSync(path.join(root, 'images', 'kradar_examples.png'), path.join(root, 'dist', 'images', 'kradar_examples.png'));
for (const weather of ['clear', 'fog', 'rain', 'snow']) for (const extension of ['mp4', 'jpg']) {
  const file = `camera-${weather}.${extension}`;
  copyFileSync(path.join(root, 'images', file), path.join(root, 'dist', 'images', file));
}
copyFileSync(path.join(root, 'images', 'CAMERA-SOURCES.md'), path.join(root, 'dist', 'images', 'CAMERA-SOURCES.md'));
cpSync(path.join(root, 'fonts'), path.join(root, 'dist', 'fonts'), { recursive: true });
console.log(JSON.stringify({ papers: records.length, sourceEntries: sourceCount, tracks: tracks.length, sourceUpdated: data.updated, output: path.join(root, 'dist') }));
