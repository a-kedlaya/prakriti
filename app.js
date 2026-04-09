// Automatically use production URL when not running locally
const API = (location.hostname === "localhost" || location.hostname === "127.0.0.1")
    ? "http://localhost:5001"
    : "https://prakriti-ai.onrender.com";  // ← your Render URL

// ── Dosha card expand/collapse ──────────────────────────────────────────────
const expanded = { vata: false, pitta: false, kapha: false };

function expandDosha(name) {
    const body = document.getElementById(`${name}-body`);
    const card = body.closest('.dosha-card');
    const toggle = card.querySelector('.card-toggle');
    expanded[name] = !expanded[name];
    body.classList.toggle('open', expanded[name]);
    toggle.textContent = expanded[name] ? 'Show Less ↑' : 'Learn More ↓';
}

// ── Upload & real ML prediction ──────────────────────────────────────────────
let uploadedFile = null;

function handleFile(e) {
    const file = e.target.files[0];
    if (!file) return;
    uploadedFile = file;

    const reader = new FileReader();
    reader.onload = (ev) => {
        document.getElementById('uploadInner').style.display = 'none';
        document.getElementById('uploadPreview').style.display = 'block';
        document.getElementById('previewImg').src = ev.target.result;
        document.getElementById('spinner').style.display = 'flex';
        document.getElementById('resultBox').style.display = 'none';
        runPrediction(file);
    };
    reader.readAsDataURL(file);
}

async function runPrediction(file) {
    const formData = new FormData();
    formData.append("image", file);

    try {
        const res = await fetch(`${API}/predict`, { method: "POST", body: formData });
        const data = await res.json();

        if (data.error) throw new Error(data.error);
        showResult(data);
    } catch (err) {
        showError(err.message);
    }
}

const DOSHA_META = {
    'Vata': { emoji: '💨', color: '#a78bfa', desc: 'Air + Space · Movement & Creativity' },
    'Pitta': { emoji: '🔥', color: '#fb923c', desc: 'Fire + Water · Transformation & Focus' },
    'Kapha': { emoji: '🌍', color: '#4ade80', desc: 'Earth + Water · Structure & Stability' },
    'Vata-Kapha': { emoji: '💨🌍', color: '#a78bfa', desc: 'Dual Dosha · Air + Earth' },
    'Pitta-Kapha': { emoji: '🔥🌍', color: '#fb923c', desc: 'Dual Dosha · Fire + Earth' },
    'Vata-Pitta': { emoji: '💨🔥', color: '#f5c842', desc: 'Dual Dosha · Air + Fire' },
};

function showResult(data) {
    const meta = DOSHA_META[data.dosha] || { emoji: '🕉️', color: '#fff', desc: '' };
    const top3Html = data.top3.map(t => {
        const m = DOSHA_META[t.dosha] || {};
        return `<div style="display:flex;justify-content:space-between;font-size:0.75rem;color:#8888aa;margin-top:0.3rem">
      <span>${m.emoji || ''} ${t.dosha}</span><span>${t.confidence}%</span>
    </div>`;
    }).join('');

    const box = document.getElementById('resultBox');
    box.innerHTML = `
    <div class="result-label">Your Prakriti</div>
    <div class="result-dosha" style="color:${meta.color}">${meta.emoji} ${data.dosha}</div>
    <div class="result-conf">${meta.desc}</div>
    <div style="margin-top:0.6rem;font-size:0.75rem;color:#8888aa">Confidence: ${data.confidence}%</div>
    <div style="margin-top:0.8rem;border-top:1px solid rgba(255,255,255,0.08);padding-top:0.6rem">${top3Html}</div>
  `;
    document.getElementById('spinner').style.display = 'none';
    box.style.display = 'block';
}

function showError(msg) {
    const box = document.getElementById('resultBox');
    box.innerHTML = `
    <div style="color:#f87171;font-size:0.85rem">⚠️ ${msg || 'Could not connect to server.'}</div>
    <div style="color:#8888aa;font-size:0.75rem;margin-top:0.5rem">Make sure <code>server.py</code> is running.</div>
  `;
    document.getElementById('spinner').style.display = 'none';
    box.style.display = 'block';
}

function resetUpload() {
    document.getElementById('uploadInner').style.display = 'block';
    document.getElementById('uploadPreview').style.display = 'none';
    document.getElementById('fileInput').value = '';
    uploadedFile = null;
}

// ── Bot Modal ────────────────────────────────────────────────────────────────
function openBot() {
    document.getElementById('botOverlay').classList.add('open');
    document.getElementById('botInput').focus();
    checkBotStatus();
}

function closeBot(e) {
    if (!e || e.target === document.getElementById('botOverlay')) {
        document.getElementById('botOverlay').classList.remove('open');
    }
}

// ── Launch real-time camera bot ───────────────────────────────────────────────
async function launchCameraBot() {
    const btn = document.getElementById('launchBotBtn');
    btn.disabled = true;
    btn.textContent = '⏳ Launching...';

    try {
        const res = await fetch(`${API}/launch-bot`, { method: 'POST' });
        const data = await res.json();
        appendBotMsg('bot', data.status === 'launched'
            ? '✅ Camera bot launched! An OpenCV window should open on your screen showing live Prakriti detection.'
            : `ℹ️ ${data.message}`
        );
        btn.textContent = '📷 Camera Bot Running';
        setTimeout(checkBotStatus, 2000);
    } catch {
        appendBotMsg('bot', '⚠️ Could not reach the server. Make sure <strong>server.py</strong> is running:\n<code>python server.py</code>');
        btn.disabled = false;
        btn.textContent = '📷 Launch Camera Bot';
    }
}

async function checkBotStatus() {
    try {
        const res = await fetch(`${API}/bot-status`);
        const data = await res.json();
        const btn = document.getElementById('launchBotBtn');
        if (data.running) {
            btn.textContent = '📷 Camera Bot Running';
            btn.disabled = true;
        } else {
            btn.textContent = '📷 Launch Camera Bot';
            btn.disabled = false;
        }
    } catch { /* server not running */ }
}

// ── Bot knowledge base ───────────────────────────────────────────────────────
const kb = [
    {
        keys: ['vata', 'air', 'space', 'wind'],
        reply: `🌬️ <strong>Vata Dosha</strong> — Air + Space<br><br>
    Controls all movement: breathing, blood flow, nerve signals, and thoughts.<br><br>
    <strong>Traits:</strong> Thin, fast-talking, creative, imaginative.<br>
    <strong>Imbalance:</strong> Anxiety, dry skin, insomnia, constipation.`
    },
    {
        keys: ['pitta', 'fire', 'heat', 'hot'],
        reply: `🔥 <strong>Pitta Dosha</strong> — Fire + Water<br><br>
    Controls digestion, metabolism, body temperature, and sharp intelligence.<br><br>
    <strong>Traits:</strong> Medium build, strong digestion, ambitious, natural leader.<br>
    <strong>Imbalance:</strong> Anger, acidity, ulcers, excess sweating.`
    },
    {
        keys: ['kapha', 'earth', 'water', 'heavy', 'stable'],
        reply: `🌍 <strong>Kapha Dosha</strong> — Earth + Water<br><br>
    Provides structure — bones, muscles, immunity — and emotional stability.<br><br>
    <strong>Traits:</strong> Strong, heavier body, calm, patient, loyal, great stamina.<br>
    <strong>Imbalance:</strong> Laziness, weight gain, congestion, cold.`
    },
    {
        keys: ['balance', 'fix', 'heal', 'remedy', 'cure'],
        reply: `⚖️ <strong>Balancing your Dosha:</strong><br><br>
    • <strong>Vata:</strong> Warm foods, routine, rest, oil massage<br>
    • <strong>Pitta:</strong> Cool foods, avoid spicy/oily, meditation, nature<br>
    • <strong>Kapha:</strong> Light foods, exercise, stimulating activities`
    },
    {
        keys: ['food', 'diet', 'eat', 'nutrition'],
        reply: `🥗 <strong>Dosha-based Diet:</strong><br><br>
    • <strong>Vata:</strong> Warm, moist, oily — soups, stews, ghee, nuts<br>
    • <strong>Pitta:</strong> Cool, sweet, bitter — salads, coconut, cucumber, mint<br>
    • <strong>Kapha:</strong> Light, dry, spicy — legumes, ginger, honey, greens`
    },
    {
        keys: ['prakriti', 'constitution', 'body type', 'what is'],
        reply: `🕉️ <strong>Prakriti</strong> is your unique mind-body constitution in Ayurveda.<br><br>
    Determined at birth, it's a blend of Vata, Pitta, and Kapha in unique proportions. Understanding it helps you make better choices about diet, lifestyle, and health.`
    },
    {
        keys: ['dual', 'combination', 'mixed', 'vata-kapha', 'pitta-kapha', 'vata-pitta'],
        reply: `⚡ <strong>Dual Doshas</strong> are very common!<br><br>
    • <strong>Vata-Pitta:</strong> Creative + Focused<br>
    • <strong>Pitta-Kapha:</strong> Ambitious + Stable<br>
    • <strong>Vata-Kapha:</strong> Creative + Grounded<br><br>
    Our ML model recognizes all 5 Prakriti types including dual combinations.`
    },
    {
        keys: ['camera', 'live', 'real time', 'realtime', 'webcam', 'launch', 'open bot', 'start bot'],
        reply: `📷 Click the <strong>"Launch Camera Bot"</strong> button above to start live Prakriti detection using your webcam!<br><br>
    It uses your trained <code>prakriti_model.h5</code> + OpenCV to detect your face and predict your Dosha in real time.`
    },
    {
        keys: ['hello', 'hi', 'hey', 'namaste'],
        reply: `Namaste! 🙏 I'm your Prakriti guide.<br><br>
    I can answer questions about Doshas, Ayurvedic diet, and lifestyle — or you can launch the live camera bot to detect your Prakriti in real time!`
    },
    {
        keys: ['ayurveda', 'ancient', 'science', 'history'],
        reply: `📜 <strong>Ayurveda</strong> is one of the world's oldest healing systems, originating in India over 5,000 years ago.<br><br>
    "Ayur" = life, "Veda" = knowledge. It focuses on balance between mind, body, and spirit through diet, lifestyle, herbs, and yoga.`
    },
];

function getBotReply(msg) {
    const lower = msg.toLowerCase();
    for (const entry of kb) {
        if (entry.keys.some(k => lower.includes(k))) return entry.reply;
    }
    return `🤔 Great question! Try asking me about:<br>
  • <em>Vata, Pitta, or Kapha dosha</em><br>
  • <em>How to balance your dosha</em><br>
  • <em>Ayurvedic diet tips</em><br>
  • <em>"Launch camera bot"</em> for live detection`;
}

function appendBotMsg(who, html) {
    const msgs = document.getElementById('botMessages');
    const div = document.createElement('div');
    div.className = `bot-msg ${who}`;
    div.innerHTML = `<div class="msg-bubble">${html}</div>`;
    msgs.appendChild(div);
    msgs.scrollTop = msgs.scrollHeight;
}

function sendMsg() {
    const input = document.getElementById('botInput');
    const text = input.value.trim();
    if (!text) return;

    appendBotMsg('user', escapeHtml(text));
    input.value = '';

    const typingDiv = document.createElement('div');
    typingDiv.className = 'bot-msg bot';
    typingDiv.innerHTML = `<div class="msg-bubble" style="opacity:0.5">✦ typing...</div>`;
    document.getElementById('botMessages').appendChild(typingDiv);
    document.getElementById('botMessages').scrollTop = 9999;

    setTimeout(() => {
        typingDiv.remove();
        appendBotMsg('bot', getBotReply(text));
    }, 700);
}

function escapeHtml(str) {
    return str.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
}

// ── Drag-and-drop ─────────────────────────────────────────────────────────────
const uploadBox = document.getElementById('uploadBox');
uploadBox.addEventListener('dragover', e => { e.preventDefault(); uploadBox.style.borderColor = 'rgba(124,106,247,0.8)'; });
uploadBox.addEventListener('dragleave', () => { uploadBox.style.borderColor = ''; });
uploadBox.addEventListener('drop', e => {
    e.preventDefault();
    uploadBox.style.borderColor = '';
    const file = e.dataTransfer.files[0];
    if (file && file.type.startsWith('image/')) {
        handleFile({ target: { files: [file] } });
    }
});
