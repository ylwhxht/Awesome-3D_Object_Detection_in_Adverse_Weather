(() => {
  'use strict';
  const canvas = document.getElementById('scene');
  const ctx = canvas.getContext('2d');
  if (!ctx) return;
  const lab = canvas.closest('.weather-lab');
  const reduceMotion = matchMedia('(prefers-reduced-motion: reduce)');
  const pauseButton = document.getElementById('pause-scene');
  const status = document.getElementById('scene-status');
  const camera = document.getElementById('weather-camera');
  const cameraFeed = document.getElementById('camera-feed');
  const cameraPoster = document.getElementById('camera-poster');
  const cameraMessage = document.getElementById('camera-message');
  const cycleProgress = document.getElementById('cycle-progress');
  const conditions = ['clear', 'fog', 'rain', 'snow'];
  const cycleSeconds = 8;
  const footage = {
    clear: { name: 'Boreas', sequence: '2021-03-09-14-23', url: 'https://www.boreas.utias.utoronto.ca/#/download', license: 'CC BY 4.0' },
    fog: { name: 'RADIATE', sequence: 'tiny_foggy / fog_6_0', url: 'https://github.com/marcelsheeny/radiate_sdk/tree/master/data/radiate/tiny_foggy', license: 'CC BY-NC-SA 4.0' },
    rain: { name: 'Boreas', sequence: '2021-06-29-18-53', url: 'https://www.boreas.utias.utoronto.ca/#/download', license: 'CC BY 4.0' },
    snow: { name: 'Boreas', sequence: '2021-01-26-11-22', url: 'https://www.boreas.utias.utoronto.ca/#/download', license: 'CC BY 4.0' }
  };
  let motionChoice = null;
  try { motionChoice = localStorage.getItem('adverse-weather-motion'); } catch { /* Storage may be unavailable. */ }
  const random = n => { const x = Math.sin(n * 127.1 + 311.7) * 43758.5453; return x - Math.floor(x); };
  const wrap = (value, range) => ((value % range) + range) % range;
  const edges = [[0,1],[1,2],[2,3],[3,0],[4,5],[5,6],[6,7],[7,4],[0,4],[1,5],[2,6],[3,7]];
  let width = 0, height = 0, weather = 'clear', paused = motionChoice === 'play' ? false : motionChoice === 'pause' ? true : reduceMotion.matches;
  let weatherIndex = 0, weatherElapsed = 0, cameraRevision = 0;
  let inView = true, frame = 0, last = 0, phase = 0, yaw = 0, tilt = 0, scanZ = 10;
  let pointerX = 0, pointerY = 0, cameraX = 0, cameraY = 0;
  const ground = Array.from({ length: 2200 }, (_, i) => ({ x: (random(i + 1) - .5) * 34, z: random(i + 3300) * 84, size: .55 + random(i + 4000) }));
  const roadside = Array.from({ length: 700 }, (_, i) => ({ x: (6 + random(i + 5000) * 13) * (i % 2 ? 1 : -1), y: random(i + 7000) * 6, z: random(i + 6000) * 84 }));
  const particles = Array.from({ length: 150 }, (_, i) => ({ x: random(i + 9000), y: random(i + 10000), size: .4 + random(i + 11000), speed: .4 + random(i + 12000) }));
  const vehicles = [
    { x: -2.2, z: 9, width: 2, length: 4.4, height: 1.65 },
    { x: 2.1, z: 22, width: 2.2, length: 4.8, height: 2 },
    { x: -2, z: 40, width: 2, length: 4.4, height: 1.6 }
  ].map((v, index) => ({ ...v, points: Array.from({ length: 190 }, (_, i) => {
    const a = random(i + 13 + index * 751), b = random(i + 731 + index * 600);
    if (i % 3 === 0) return { x: (a - .5) * v.width, y: v.height * .63, z: b * v.length };
    if (i % 3 === 1) return { x: (i % 2 ? 1 : -1) * v.width / 2, y: a * v.height * .65, z: b * v.length };
    return { x: (a - .5) * v.width * .8, y: v.height, z: .8 + b * (v.length - 1.7) };
  }) }));
  function project(x, y, z) {
    const scale = 12 / (z + 12);
    return { x: width * .5 + (x + (z - 20) * yaw) * width * .071 * scale,
      y: height * (.23 + tilt) + (4.7 - y) * height * .145 * scale, scale };
  }
  function segment(a, b, color, lineWidth = 1) {
    ctx.strokeStyle = color; ctx.lineWidth = lineWidth; ctx.beginPath();
    ctx.moveTo(a.x, a.y); ctx.lineTo(b.x, b.y); ctx.stroke();
  }
  function point(x, y, z, size = 1) {
    const p = project(x, y, z), s = Math.max(.55, size * p.scale);
    ctx.fillRect(p.x, p.y, s, s);
  }
  function polygon(points, color) {
    ctx.beginPath(); points.forEach((p, i) => i ? ctx.lineTo(p.x, p.y) : ctx.moveTo(p.x, p.y));
    ctx.closePath(); ctx.fillStyle = color; ctx.fill();
  }
  function vehicle(v, index) {
    const z = v.z + Math.sin(phase * .52 + index * 1.7) * (2 + index);
    const x = v.x + Math.sin(phase * .35 + index) * .22;
    const hit = Math.exp(-Math.abs(z - scanZ) / 7);
    ctx.fillStyle = '#d4efc1'; ctx.globalAlpha = .66 + hit * .3;
    for (const p of v.points) point(x + p.x, p.y, z + p.z, 1.9);
    ctx.globalAlpha = 1;
    const corners = [
      project(x - v.width / 2, 0, z), project(x + v.width / 2, 0, z),
      project(x + v.width / 2, 0, z + v.length), project(x - v.width / 2, 0, z + v.length),
      project(x - v.width / 2, v.height, z), project(x + v.width / 2, v.height, z),
      project(x + v.width / 2, v.height, z + v.length), project(x - v.width / 2, v.height, z + v.length)
    ];
    polygon([corners[4], corners[5], corners[6], corners[7]], `rgba(187,235,166,${.04 + hit * .16})`);
    for (const [a, b] of edges) segment(corners[a], corners[b], index === 0 ? '#deefab' : '#86d7b7', 1.1 + hit * .6);
    // A moving reconstruction slice makes each tracked volume legible.
    const slice = z + wrap(phase * 1.35 + index, v.length);
    segment(project(x - v.width / 2, v.height, slice), project(x + v.width / 2, v.height, slice), '#ecffd9', 1.4);
    const top = corners[index === 1 ? 6 : 7];
    const label = { x: top.x + (index === 1 ? 15 : index === 2 ? -35 : -16), y: top.y + (index === 1 ? 1 : -19) };
    segment(top, { x: label.x, y: label.y + 4 }, '#b4d69d88', .7);
    ctx.font = `${width < 380 ? 9 : 10}px monospace`; ctx.fillStyle = '#d3e7bc';
    ctx.fillText(`TRACK / 0${index + 1}`, label.x, label.y);
    if (index === 0) {
      const center = project(x, 0, z + v.length / 2);
      ctx.strokeStyle = '#cfeaab66'; ctx.lineWidth = .8;
      ctx.beginPath(); ctx.ellipse(center.x, center.y + 4, 30 + Math.sin(phase * 1.7) * 4, 8, 0, 0, Math.PI * 2); ctx.stroke();
    }
  }
  function drawScan() {
    for (let i = 0; i < 2; i++) {
      const radius = 4 + wrap(phase * 10 + i * 33, 66);
      ctx.beginPath();
      for (let j = 0; j <= 64; j++) {
        const angle = Math.PI * j / 64, p = project(Math.cos(angle) * radius, .03, Math.sin(angle) * radius);
        if (!j) ctx.moveTo(p.x, p.y); else ctx.lineTo(p.x, p.y);
      }
      ctx.strokeStyle = `rgba(102,226,199,${.32 * (1 - radius / 76)})`; ctx.lineWidth = 1.1; ctx.stroke();
    }
    const a = project(-7, 0, scanZ), b = project(7, 0, scanZ);
    const c = project(7, 3.7, scanZ), d = project(-7, 3.7, scanZ);
    const glow = ctx.createLinearGradient(a.x, a.y, d.x, d.y);
    glow.addColorStop(0, '#7bf4cc24'); glow.addColorStop(1, '#7bf4cc00');
    polygon([a, b, c, d], glow);
    ctx.shadowColor = '#7bf4cc'; ctx.shadowBlur = 10;
    segment(a, b, '#8dffe0aa', 1.5); ctx.shadowBlur = 0;
    segment(a, d, '#7bf4cc33', .7); segment(b, c, '#7bf4cc33', .7);
  }
  function drawRadar() {
    const r = width < 380 ? 22 : 28, x = width - r - 25, y = height * .25 + 15;
    const angle = phase * .8 - Math.PI / 2;
    ctx.save(); ctx.translate(x, y);
    ctx.strokeStyle = '#9bbfb032'; ctx.lineWidth = .7;
    for (const radius of [r * .5, r]) { ctx.beginPath(); ctx.arc(0, 0, radius, 0, Math.PI * 2); ctx.stroke(); }
    segment({x:-r-5,y:0}, {x:r+5,y:0}, '#9bbfb027', .7);
    segment({x:0,y:-r-5}, {x:0,y:r+5}, '#9bbfb027', .7);
    ctx.beginPath(); ctx.moveTo(0, 0); ctx.arc(0, 0, r, angle - .75, angle); ctx.closePath(); ctx.fillStyle = '#8deac523'; ctx.fill();
    segment({x:0,y:0}, {x:Math.cos(angle)*r,y:Math.sin(angle)*r}, '#a9f3cfbb', 1);
    for (let i = 0; i < 3; i++) {
      ctx.fillStyle = `rgba(216,241,171,${.5 + .4 * Math.sin(phase * 1.5 + i)})`;
      ctx.beginPath(); ctx.arc(Math.cos(i * 2.1 + .4) * r * .65, Math.sin(i * 2.1 + .4) * r * .65, 1.8, 0, Math.PI * 2); ctx.fill();
    }
    ctx.fillStyle = '#9ecbbb'; ctx.font = '9px monospace'; ctx.textAlign = 'center';
    ctx.fillText(`SCAN ${String(Math.floor(wrap(phase * .8 * 180 / Math.PI, 360))).padStart(3, '0')}°`, 0, r + 17);
    ctx.restore();
  }
  function atmosphere() {
    if (weather === 'fog') {
      for (let i = 0; i < 3; i++) {
        const x = width * (.2 + i * .3) + Math.sin(phase * .24 + i * 2) * width * .13;
        const y = height * (.24 + i * .09) + Math.cos(phase * .3 + i) * 12;
        const fog = ctx.createRadialGradient(x, y, 0, x, y, width * .38);
        fog.addColorStop(0, '#b7d0b51c'); fog.addColorStop(.5, '#a3c5b20e'); fog.addColorStop(1, '#9ecdb000');
        ctx.fillStyle = fog; ctx.fillRect(0, 0, width, height);
      }
    }
    const count = weather === 'clear' ? 28 : weather === 'fog' ? 60 : particles.length;
    for (let i = 0; i < count; i++) {
      const p = particles[i];
      if (weather === 'rain') {
        const y = wrap(p.y * height + phase * 245 * p.speed, height);
        const x = wrap(p.x * width - phase * 37 * p.speed, width);
        segment({x,y}, {x:x-4,y:y+13*p.size}, '#a9decc55', .8);
      } else {
        const snow = weather === 'snow';
        const x = wrap(p.x * width + phase * (snow ? 12 : 18) * p.speed + Math.sin(phase * .8 + i) * 9, width);
        const y = wrap(p.y * height + phase * (snow ? 38 : -8) * p.speed, height);
        ctx.fillStyle = snow ? '#e4f4dcaa' : '#bcebd546';
        ctx.beginPath(); ctx.arc(x, y, p.size * (snow ? 1.25 : .65), 0, Math.PI * 2); ctx.fill();
      }
    }
  }
  function draw() {
    if (!width || !height) return;
    ctx.clearRect(0, 0, width, height);
    yaw = .04 + Math.sin(phase * .23) * .028 + cameraX * .05;
    tilt = Math.sin(phase * .31) * .008 + cameraY * .018;
    scanZ = 3 + wrap(phase * 12, 72);
    // All geometry, tracking and scanning are illustrative, not sensor measurements.
    for (let x = -18; x <= 18; x += 3) segment(project(x, 0, 0), project(x, 0, 90), '#80c8ac13', .7);
    for (let i = 0; i < 12; i++) {
      const z = wrap(i * 7 - phase * 6.5, 84);
      segment(project(-20, 0, z), project(20, 0, z), '#80c8ac17', .7);
    }
    for (const p of ground) {
      const z = wrap(p.z - phase * 6.5, 84), road = Math.abs(p.x) < 5.2;
      const scan = Math.exp(-Math.abs(z - scanZ) / 3.2);
      ctx.fillStyle = scan > .45 ? '#c3ffe0' : road ? '#9edac0' : '#6aab94';
      ctx.globalAlpha = Math.min(1, (road ? .6 : .36) + scan * .65) * (weather === 'fog' ? 1 - z / 125 : 1);
      point(p.x, 0, z, p.size * (1.5 + scan));
    }
    ctx.globalAlpha = 1;
    ctx.fillStyle = '#c2dab377';
    for (let i = 0; i < 190; i++) {
      const z = wrap(i * .44 - phase * 6.5, 84);
      if (z % 10 < 4.7) point(0, .02, z, 2);
      for (const x of [-5.2, 5.2]) point(x, .02, z, 1.6);
    }
    ctx.fillStyle = '#8bd4b36b';
    for (const p of roadside) point(p.x, p.y, wrap(p.z - phase * 6.5, 84), 1.65);
    drawScan();
    for (let i = vehicles.length - 1; i >= 0; i--) vehicle(vehicles[i], i);
    atmosphere(); drawRadar();
  }
  function loop(time) {
    frame = 0;
    if (paused || !inView || document.hidden) return;
    if (time - last >= 1000 / 30) {
      const elapsed = Math.min((time - last) / 1000, .25); last = time;
      phase += elapsed;
      weatherElapsed += elapsed;
      if (weatherElapsed >= cycleSeconds) {
        weatherElapsed %= cycleSeconds;
        setWeather((weatherIndex + 1) % conditions.length);
      }
      if (cycleProgress) cycleProgress.style.transform = `scaleX(${weatherElapsed / cycleSeconds})`;
      cameraX += (pointerX - cameraX) * .07; cameraY += (pointerY - cameraY) * .07;
      draw();
    }
    frame = requestAnimationFrame(loop);
  }
  function schedule() {
    if (frame) cancelAnimationFrame(frame);
    frame = 0;
    if (!paused && inView && !document.hidden) { last = performance.now(); frame = requestAnimationFrame(loop); }
    syncCamera();
  }
  function resize() {
    const rect = canvas.getBoundingClientRect(); width = rect.width; height = rect.height;
    const dpr = Math.min(devicePixelRatio || 1, 2);
    canvas.width = Math.round(width * dpr); canvas.height = Math.round(height * dpr);
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0); draw();
  }
  const descriptions = { clear: 'A clear view. A starting point.', fog: 'Find the signal through the fog.', rain: 'Read the scene beyond the rain.', snow: 'Keep the geometry. Lose the noise.' };
  function syncCamera() {
    if (!camera) return; // Older cached HTML can briefly load the newly deployed script.
    if (paused || !inView || document.hidden) { camera.pause(); return; }
    if (!camera.paused || camera.error) return;
    const revision = cameraRevision;
    camera.play().catch(error => {
      if (revision !== cameraRevision || error.name === 'AbortError') return;
      if (error.name === 'NotAllowedError') {
        paused = true;
        syncPause();
      } else showCameraError();
    });
  }
  function showCameraError() {
    cameraFeed.dataset.ready = 'false';
    cameraMessage.textContent = 'Clip unavailable · view source ↗';
    cameraMessage.hidden = false;
  }
  function setWeather(index) {
    weatherIndex = index;
    weather = conditions[index];
    const source = footage[weather];
    lab.dataset.weather = weather;
    document.querySelectorAll('[data-condition]').forEach(el => {
      if (el.dataset.condition === weather) el.setAttribute('aria-current', 'step');
      else el.removeAttribute('aria-current');
    });
    document.getElementById('weather-caption').textContent = descriptions[weather];
    for (const [id, text] of Object.entries({
      'cycle-position': `AUTO / 0${index + 1}`,
      'cycle-next': `NEXT: ${conditions[(index + 1) % conditions.length].toUpperCase()}`,
      'camera-weather': weather[0].toUpperCase() + weather.slice(1)
    })) {
      const label = document.getElementById(id);
      if (label) label.textContent = text;
    }
    canvas.setAttribute('aria-label', `Illustrated moving 3D point-cloud road, scanning waves and tracked vehicles in ${weather === 'clear' ? 'clear weather' : weather}`);
    if (!camera) {
      document.querySelectorAll('.weather-switch [data-weather]').forEach(el => el.setAttribute('aria-pressed', String(el.dataset.weather === weather)));
      draw(); return;
    }
    const credit = document.getElementById('camera-source');
    credit.textContent = `${source.name} ↗`; credit.href = source.url;
    credit.title = `${source.name} · ${source.sequence} · ${source.license}`;
    cameraRevision++;
    camera.pause();
    cameraFeed.dataset.ready = 'false'; cameraMessage.hidden = true;
    cameraPoster.src = `images/camera-${weather}.jpg`;
    camera.poster = cameraPoster.src;
    camera.setAttribute('aria-label', `${source.name} real front-camera footage in ${weather === 'clear' ? 'clear weather' : weather}`);
    camera.src = `images/camera-${weather}.mp4`;
    camera.load(); syncCamera();
    draw();
  }
  if (camera) {
    camera.muted = true;
    camera.addEventListener('loadeddata', () => { cameraFeed.dataset.ready = 'true'; cameraMessage.hidden = true; });
    camera.addEventListener('error', showCameraError);
  }
  lab.addEventListener('pointermove', event => {
    if (event.pointerType === 'touch') return;
    const rect = lab.getBoundingClientRect();
    pointerX = (event.clientX - rect.left) / rect.width - .5;
    pointerY = (event.clientY - rect.top) / rect.height - .5;
  }, { passive: true });
  lab.addEventListener('pointerleave', () => { pointerX = 0; pointerY = 0; });
  function syncPause() {
    const label = paused ? 'Play weather cycle and camera' : 'Pause weather cycle and camera';
    pauseButton.setAttribute('aria-label', label); pauseButton.title = label;
    pauseButton.querySelector('use').setAttribute('href', paused ? '#i-play' : '#i-pause');
    status.textContent = paused ? 'PAUSED · PRESS PLAY' : 'AUTO WEATHER · 8s';
    lab.dataset.motion = paused ? 'paused' : 'running';
    schedule();
  }
  pauseButton.addEventListener('click', () => {
    paused = !paused;
    try { localStorage.setItem('adverse-weather-motion', paused ? 'pause' : 'play'); } catch { /* Playback still works without storage. */ }
    syncPause();
  });
  reduceMotion.addEventListener('change', event => {
    paused = event.matches;
    try { localStorage.removeItem('adverse-weather-motion'); } catch { /* Follow the system preference for this visit. */ }
    syncPause(); draw();
  });
  document.addEventListener('visibilitychange', schedule);
  new IntersectionObserver(entries => { inView = entries[0].isIntersecting; schedule(); }).observe(lab);
  new ResizeObserver(resize).observe(canvas);
  if (camera?.readyState >= 2) cameraFeed.dataset.ready = 'true';
  if (!camera) setWeather(0);
  resize(); syncPause();
})();
