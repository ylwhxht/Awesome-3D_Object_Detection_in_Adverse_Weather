import http from 'node:http';
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..', 'dist');
const types = { '.html': 'text/html; charset=utf-8', '.js': 'text/javascript; charset=utf-8', '.css': 'text/css; charset=utf-8', '.png': 'image/png', '.jpg': 'image/jpeg', '.mp4': 'video/mp4', '.md': 'text/plain; charset=utf-8', '.svg': 'image/svg+xml', '.woff2': 'font/woff2', '.txt': 'text/plain; charset=utf-8' };
const server = http.createServer((req, res) => {
  let pathname;
  try { pathname = decodeURIComponent(new URL(req.url, 'http://localhost').pathname); } catch { res.writeHead(400).end(); return; }
  const file = path.resolve(root, '.' + (pathname.endsWith('/') ? pathname + 'index.html' : pathname));
  if (!file.startsWith(root + path.sep)) { res.writeHead(403).end(); return; }
  fs.readFile(file, (error, contents) => {
    if (error) { res.writeHead(404).end('Not found'); return; }
    const headers = { 'Content-Type': types[path.extname(file)] || 'application/octet-stream', 'Cache-Control': 'no-cache', 'Accept-Ranges': 'bytes' };
    const range = req.headers.range?.match(/^bytes=(\d*)-(\d*)$/);
    if (range && (range[1] || range[2])) {
      const start = range[1] ? Number(range[1]) : Math.max(0, contents.length - Number(range[2]));
      const end = range[1] && range[2] ? Math.min(Number(range[2]), contents.length - 1) : contents.length - 1;
      if (start > end || start >= contents.length) { res.writeHead(416, { 'Content-Range': `bytes */${contents.length}` }).end(); return; }
      res.writeHead(206, { ...headers, 'Content-Range': `bytes ${start}-${end}/${contents.length}`, 'Content-Length': end - start + 1 });
      res.end(req.method === 'HEAD' ? undefined : contents.subarray(start, end + 1));
    } else {
      res.writeHead(200, { ...headers, 'Content-Length': contents.length });
      res.end(req.method === 'HEAD' ? undefined : contents);
    }
  });
});
server.listen(4173, '127.0.0.1', () => console.log('Local: http://127.0.0.1:4173'));
