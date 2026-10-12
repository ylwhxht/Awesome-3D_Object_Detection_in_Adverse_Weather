(() => {
  'use strict';
  const canvas = document.getElementById('scene');
  const ctx = canvas.getContext('2d');
  if (!ctx) return;
  const reduceMotion = matchMedia('(prefers-reduced-motion: reduce)');
  let width = 0, height = 0, weather = 'fog', paused = reduceMotion.matches, inView = true, frame = 0, last = 0, phase = 0;
  const random = n => { const x = Math.sin(n * 127.1 + 311.7) * 43758.5453; return x - Math.floor(x); };
  const ground = Array.from({ length: 2700 }, (_, i) => ({ x: (random(i + 1) - .5) * 28, z: random(i + 3300) * 100, size: .35 + random(i + 4000) * .95 }));
  const foliage = Array.from({ length: 800 }, (_, i) => ({ x: (6 + random(i + 5000) * 8) * (i % 2 ? 1 : -1), y: random(i + 7000) * 4, z: 3 + random(i + 6000) * 78 }));
  const particles = Array.from({ length: 120 }, (_, i) => ({ x: random(i + 9000), y: random(i + 10000), size: .3 + random(i + 11000), speed: .3 + random(i + 12000) }));
  const vehicles = [{ x: -1.75, z: 12, width: 1.8, length: 3.8, height: 1.45 }, { x: 1.65, z: 24, width: 2, length: 4.3, height: 1.9 }, { x: -1.8, z: 44, width: 1.8, length: 3.8, height: 1.4 }];
  function project(x, y, z) {
    const scale = 1 / (1 + z * .064);
    return { x: width * .51 + (x + (z - 22) * .115) * width * .09 * scale, y: height * .31 + (7.5 - y - z * .032) * height * .095 * scale, scale };
  }
  function segment(a, b, color, lineWidth = 1) { ctx.strokeStyle = color; ctx.lineWidth = lineWidth; ctx.beginPath(); ctx.moveTo(a.x, a.y); ctx.lineTo(b.x, b.y); ctx.stroke(); }
  function point(x, y, z, color, size = 1) { const p = project(x, y, z); ctx.fillStyle = color; ctx.fillRect(p.x, p.y, Math.max(.4, size * p.scale), Math.max(.4, size * p.scale)); }
  function vehicle(v, index) {
    const sides = [-1, 1];
    for (let i = 0; i < 230; i++) {
      const a = random(i + 13 + index * 751), b = random(i + 731 + index * 600);
      let x, y, z;
      if (i % 3 === 0) { x = v.x + (a - .5) * v.width; y = v.height * .64; z = v.z + b * v.length; }
      else if (i % 3 === 1) { x = v.x + sides[i % 2] * v.width / 2; y = a * v.height * .65; z = v.z + b * v.length; }
      else { x = v.x + (a - .5) * v.width * .82; y = v.height; z = v.z + .8 + b * (v.length - 1.7); }
      point(x, y, z, `rgba(202,224,187,${.44 + a * .45})`, 1.7);
    }
    const corners = [
      project(v.x - v.width / 2, 0, v.z), project(v.x + v.width / 2, 0, v.z), project(v.x + v.width / 2, 0, v.z + v.length), project(v.x - v.width / 2, 0, v.z + v.length),
      project(v.x - v.width / 2, v.height, v.z), project(v.x + v.width / 2, v.height, v.z), project(v.x + v.width / 2, v.height, v.z + v.length), project(v.x - v.width / 2, v.height, v.z + v.length)
    ];
    for (const [a,b] of [[0,1],[1,2],[2,3],[3,0],[4,5],[5,6],[6,7],[7,4],[0,4],[1,5],[2,6],[3,7]]) segment(corners[a], corners[b], index === 0 ? '#e1eeb5' : '#badfa6dd', 1.05);
    const top = corners[7];
    ctx.font = `${Math.max(7, width * .014)}px monospace`; ctx.fillStyle = '#d5e4bcbf'; ctx.fillText('VEHICLE / 0' + (index + 1), top.x, top.y - 9);
  }
  function draw() {
    if (!width || !height) return;
    ctx.clearRect(0, 0, width, height);
    // Perspective guides and points are an illustration, never measured detections.
    for (let x = -12; x <= 12; x += 3) segment(project(x, 0, 0), project(x, 0, 110), '#8ba98710', .6);
    for (let z = 0; z < 90; z += 9) segment(project(-15, 0, z), project(15, 0, z), '#8ba9870c', .6);
    for (const p of ground) {
      const road = Math.abs(p.x) < 4.7;
      const boundary = Math.abs(Math.abs(p.x) - 4.5) < .28;
      const fade = weather === 'fog' ? Math.max(.1, 1 - p.z / 135) : 1;
      point(p.x, 0, p.z, `rgba(${road ? '170,204,168' : '115,162,130'},${(boundary ? 1 : road ? .68 : .44) * fade})`, boundary ? p.size * 2.5 : p.size * 1.9);
    }
    for (let z = 0; z < 90; z += .42) {
      const lane = ((z + phase * 1.5) % 10) < 5;
      if (lane) for (let x = -.04; x <= .04; x += .08) point(x, .015, z, '#c1d2ac90', 1.6);
      for (const x of [-4.5, 4.5]) point(x, .04, z, '#9bc6a677', 1.7);
    }
    for (const p of foliage) point(p.x, p.y, p.z, '#a6c39777', 1.8);
    vehicles.slice().reverse().forEach(v => vehicle(v, vehicles.indexOf(v)));
    const ring = 8 + (phase * 2 % 30);
    ctx.beginPath(); for (let i = 0; i <= 100; i++) { const a = Math.PI * i / 100; const p = project(Math.cos(a) * ring, .02, Math.sin(a) * ring); if (!i) ctx.moveTo(p.x,p.y); else ctx.lineTo(p.x,p.y); } ctx.strokeStyle = '#b7d49a18'; ctx.lineWidth = 1; ctx.stroke();
    if (weather === 'fog') {
      const fog = ctx.createRadialGradient(width * .61, height * .37, 1, width * .6, height * .36, width * .6);
      fog.addColorStop(0, '#9aae9236'); fog.addColorStop(.5, '#93b19510'); fog.addColorStop(1, '#182e2600'); ctx.fillStyle = fog; ctx.fillRect(0, 0, width, height);
      for (let i = 0; i < 35; i++) { const p = particles[i]; const x = (p.x * width + phase * 6) % width; const y = height * .2 + p.y * height * .6; ctx.fillStyle = '#d1ddc617'; ctx.beginPath(); ctx.arc(x, y, p.size, 0, Math.PI * 2); ctx.fill(); }
    }
    if (weather === 'rain' || weather === 'snow') for (const p of particles) {
      const y = (p.y * height + phase * (weather === 'rain' ? 140 : 18) * p.speed) % height;
      const x = (p.x * width + Math.sin(phase + p.y * 5) * (weather === 'snow' ? 10 : 3)) % width;
      if (weather === 'rain') segment({ x, y }, { x: x - 3, y: y + 9 * p.size }, '#a9cebd40', .7);
      else { ctx.fillStyle = '#e8edd788'; ctx.beginPath(); ctx.arc(x,y,p.size * 1.4,0,Math.PI*2); ctx.fill(); }
    }
  }
  function loop(time) {
    frame = 0;
    if (paused || !inView || document.hidden) return;
    if (time - last >= 50) { phase += Math.min((time - last) / 1000, .06); last = time; draw(); }
    frame = requestAnimationFrame(loop);
  }
  function schedule() { if (frame) cancelAnimationFrame(frame); frame = 0; if (!paused && inView && !document.hidden) { last = performance.now(); frame = requestAnimationFrame(loop); } }
  function resize() { const rect = canvas.getBoundingClientRect(); width = rect.width; height = rect.height; const dpr = Math.min(devicePixelRatio || 1, 2); canvas.width = Math.round(width * dpr); canvas.height = Math.round(height * dpr); ctx.setTransform(dpr, 0, 0, dpr, 0, 0); draw(); }
  const descriptions = { clear: 'A clear view. A starting point.', fog: 'Find the signal through the fog.', rain: 'Read the scene beyond the rain.', snow: 'Keep the geometry. Lose the noise.' };
  document.querySelectorAll('[data-weather]').forEach(button => button.addEventListener('click', () => { weather = button.dataset.weather; document.querySelectorAll('[data-weather]').forEach(el => el.setAttribute('aria-pressed', String(el === button))); document.getElementById('weather-caption').textContent = descriptions[weather]; canvas.setAttribute('aria-label', `Illustration of a 3D point-cloud road with bounding boxes around vehicles in ${weather === 'clear' ? 'clear weather' : weather}`); draw(); }));
  const pauseButton = document.getElementById('pause-scene');
  function syncPause() { pauseButton.setAttribute('aria-label', paused ? 'Play illustration' : 'Pause illustration'); pauseButton.title = paused ? 'Play illustration' : 'Pause illustration'; pauseButton.querySelector('use').setAttribute('href', paused ? '#i-play' : '#i-pause'); schedule(); }
  pauseButton.addEventListener('click', () => { paused = !paused; syncPause(); });
  reduceMotion.addEventListener('change', event => { paused = event.matches; syncPause(); draw(); });
  document.addEventListener('visibilitychange', schedule);
  new IntersectionObserver(entries => { inView = entries[0].isIntersecting; schedule(); }).observe(canvas);
  new ResizeObserver(resize).observe(canvas);
  resize(); syncPause();
})();
