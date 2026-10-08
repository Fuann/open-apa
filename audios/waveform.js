/* The radial envelope is measured from sample.wav; rotation is decorative. */
(() => {
    'use strict';
    const audio = document.querySelector('audio');
    const canvas = document.querySelector('#waveform');
    const status = document.querySelector('#waveform-status');
    const context = canvas.getContext('2d');
    if (!context) {
        status.textContent = 'Waveform preview is unavailable. Use the audio player below.';
        return;
    }
    const controls = document.querySelector('.waveform-controls');
    const play = document.querySelector('#waveform-play');
    const seek = document.querySelector('#waveform-seek');
    const time = document.querySelector('#waveform-time');
    const mute = document.querySelector('#waveform-mute');
    const formatTime = value => `${Math.floor(value / 60)}:${String(Math.floor(value % 60)).padStart(2, '0')}`;
    function updateControls() {
        const length = Number.isFinite(audio.duration) ? audio.duration : 0;
        seek.disabled = !length;
        seek.max = length || 1;
        seek.value = audio.currentTime;
        seek.setAttribute('aria-valuetext', `${audio.currentTime.toFixed(1)} of ${length.toFixed(1)} seconds`);
        time.textContent = `${formatTime(audio.currentTime)} / ${formatTime(length)}`;
        play.textContent = audio.paused ? '▶' : 'Ⅱ';
        play.setAttribute('aria-label', audio.paused ? 'Play recording' : 'Pause recording');
        mute.textContent = audio.muted ? '×' : '♪';
        mute.setAttribute('aria-label', audio.muted ? 'Unmute recording' : 'Mute recording');
        mute.setAttribute('aria-pressed', String(audio.muted));
    }
    play.addEventListener('click', async () => {
        if (!audio.paused) { audio.pause(); return; }
        try { await audio.play(); }
        catch { status.textContent = 'Unable to play the recording. Please try again.'; }
    });
    seek.addEventListener('input', () => {
        if (Number.isFinite(audio.duration)) audio.currentTime = Number(seek.value);
        updateControls(); draw();
    });
    mute.addEventListener('click', () => { audio.muted = !audio.muted; });
    audio.addEventListener('volumechange', updateControls);
    audio.addEventListener('durationchange', updateControls);
    audio.addEventListener('error', () => { status.textContent = 'Unable to load the recording. Please reload the page.'; });
    let dragging = null;
    function seekAt(event) {
        if (!Number.isFinite(audio.duration)) return;
        const rect = canvas.getBoundingClientRect();
        audio.currentTime = clamp((event.clientX - rect.left - 16) / (rect.width - 32), 0, 1) * audio.duration;
        updateControls(); draw();
    }
    canvas.addEventListener('pointerdown', event => {
        if (!event.isPrimary || event.button !== 0) return;
        dragging = event.pointerId;
        canvas.setPointerCapture(event.pointerId);
        seekAt(event);
    });
    canvas.addEventListener('pointermove', event => { if (dragging === event.pointerId) seekAt(event); });
    for (const name of ['pointerup', 'pointercancel', 'lostpointercapture']) {
        canvas.addEventListener(name, () => { dragging = null; });
    }
    canvas.classList.add('waveform-interactive');
    controls.hidden = false;
    audio.controls = false;
    audio.hidden = true;
    updateControls();
    const motion = matchMedia('(prefers-reduced-motion: reduce)');
    let envelope, duration, width = 0, height = 0, frame = 0;
    const clamp = (v, low, high) => Math.max(low, Math.min(high, v));

    // Plot bounds in the original 1660 × 548 image, excluding axes/margins.
    const spectrogram = new Image();
    spectrogram.addEventListener('load', draw);
    spectrogram.addEventListener('error', () => {
        status.textContent = 'Spectrogram could not load. The waveform and audio controls remain available.';
    });
    spectrogram.src = 'sample-spectrogram.png';

    function draw() {
        updateControls();
        if (!envelope || !width) return;
        context.clearRect(0, 0, width, height);
        const left = 16, span = width - 32;
        // Keep the waveform height; the spectrogram is 70% as tall.
        const gap = 18, referenceBottom = 60;
        const waveTop = referenceBottom + gap;
        const plotHeight = (height - 28 - waveTop - gap) / 1.7;
        const waveBottom = waveTop + plotHeight;
        const center = waveTop + plotHeight / 2;
        const progress = clamp(audio.currentTime / duration, 0, 1);
        const rotation = motion.matches ? .45 : audio.currentTime * .85 + .45;
        const radius = plotHeight * .48;
        const rings = 32;
        const points = envelope.map((amplitude, i) => {
            const x = i / (envelope.length - 1);
            return { x, r: (0.012 + amplitude * .988) * radius };
        });
        const project = (p, angle) => {
            const theta = angle + rotation + p.x * 3.4;
            const depth = Math.cos(theta);
            return [left + p.x * span + depth * p.r * .24,
                center + Math.sin(theta) * p.r * (.85 + depth * .10)];
        };
        // Draw the back half first so the front strands stay legible.
        const strands = Array.from({length: rings}, (_, i) => i * Math.PI * 2 / rings)
            .sort((a, b) => Math.cos(a + rotation) - Math.cos(b + rotation));
        for (const angle of strands) {
            const gradient = context.createLinearGradient(left, 0, width - left, height);
            gradient.addColorStop(0, '#75daca');
            gradient.addColorStop(.5, '#7ebeea');
            gradient.addColorStop(1, '#b6a5dc');
            context.strokeStyle = gradient;
            context.globalAlpha = .27 + .32 * (Math.cos(angle + rotation) + 1) / 2;
            context.lineWidth = .8;
            context.beginPath();
            points.forEach((p, i) => {
                const [x, y] = project(p, angle);
                if (i === 0) context.moveTo(x, y); else context.lineTo(x, y);
            });
            context.stroke();
        }
        // Cross-sections join the longitudinal strands into a wire surface.
        context.strokeStyle = '#8ccbdc';
        context.globalAlpha = .16;
        context.lineWidth = .6;
        for (let i = 0; i < points.length; i += 4) {
            context.beginPath();
            for (let j = 0; j <= rings; j++) {
                const [x, y] = project(points[i], j / rings * Math.PI * 2);
                if (j === 0) context.moveTo(x, y); else context.lineTo(x, y);
            }
            context.stroke();
        }
        context.globalAlpha = 1;
        const specTop = waveBottom + gap;
        const baseline = height - 28;
        const specHeight = baseline - specTop;
        if (spectrogram.complete && spectrogram.naturalWidth) {
            context.drawImage(spectrogram, 130, 63, 1400, 422, left, specTop, span, specHeight);
            context.fillStyle = 'rgba(8, 16, 25, .18)';
            context.fillRect(left + span * progress, specTop, span * (1 - progress), specHeight);
        }
        context.font = '14px system-ui, sans-serif';
        context.textAlign = 'left';
        context.fillStyle = '#b6c9d7';
        context.fillText('Reference text', left, 20);
        const words = document.querySelector('.reference-text p').textContent.trim().split(/\s+/);
        const textLeft = left + span * .1, textSpan = span * .8;
        let wordSize = 15;
        context.font = `${wordSize}px system-ui, sans-serif`;
        // Keep equal word gaps within a 10% inset on each side of the plot.
        let total = words.reduce((sum, word) => sum + context.measureText(word).width, 0);
        if (total + (words.length - 1) * 3 > textSpan) {
            wordSize *= (textSpan - (words.length - 1) * 3) / total;
            context.font = `${wordSize}px system-ui, sans-serif`;
            total = words.reduce((sum, word) => sum + context.measureText(word).width, 0);
        }
        const wordGap = (textSpan - total) / (words.length - 1);
        let wordX = textLeft;
        context.fillStyle = '#e0e9ef';
        words.forEach(word => {
            context.fillText(word, wordX, 56);
            wordX += context.measureText(word).width + wordGap;
        });
        context.fillStyle = '#343f50';
        context.fillRect(left, baseline, span, 1);
        context.fillStyle = '#99d9e4';
        context.fillRect(left, baseline, span * progress, 2);
        const playhead = left + span * progress;
        context.strokeStyle = '#c9eff4';
        context.globalAlpha = .9;
        context.beginPath();
        context.moveTo(playhead, waveTop);
        context.lineTo(playhead, waveBottom);
        context.moveTo(playhead, specTop);
        context.lineTo(playhead, baseline);
        context.stroke();
        context.globalAlpha = 1;
        context.font = '11px system-ui, sans-serif';
        context.fillStyle = '#98a6bb';
        context.textAlign = 'left';
        context.fillText('0:00', left, height - 8);
        context.textAlign = 'right';
        context.fillText(`${duration.toFixed(2)} s`, width - left, height - 8);
    }
    function stop() { cancelAnimationFrame(frame); frame = 0; }
    function tick() {
        frame = 0;
        draw();
        if (!audio.paused && !audio.ended && !document.hidden && !motion.matches)
            frame = requestAnimationFrame(tick);
    }
    function sync() {
        stop();
        draw();
        if (!audio.paused && !audio.ended && !document.hidden && !motion.matches)
            frame = requestAnimationFrame(tick);
    }
    function resize() {
        const rect = canvas.getBoundingClientRect();
        width = rect.width; height = rect.height;
        const ratio = Math.min(devicePixelRatio || 1, 2);
        canvas.width = Math.round(width * ratio);
        canvas.height = Math.round(height * ratio);
        context.setTransform(ratio, 0, 0, ratio, 0, 0);
        draw();
    }
    new ResizeObserver(resize).observe(canvas);
    for (const event of ['play', 'pause', 'ended', 'seeking', 'seeked', 'loadedmetadata'])
        audio.addEventListener(event, sync);
    audio.addEventListener('timeupdate', () => { if (!frame) draw(); });
    document.addEventListener('visibilitychange', sync);
    motion.addEventListener('change', sync);
    fetch('waveform-data.json').then(response => {
        if (!response.ok) throw new Error('Could not load waveform');
        return response.json();
    }).then(data => {
        envelope = data.envelope;
        duration = data.duration;
        status.textContent = 'Press play, or click and drag either view to explore the recording.';
        resize(); sync();
    }).catch(() => {
        status.textContent = 'Waveform preview could not load. Use the controls to play the recording.';
    });
})();
