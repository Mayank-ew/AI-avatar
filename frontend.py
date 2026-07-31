"""
Demo control panel — a Modal-hosted web UI to exercise the full pipeline end to end.

Serves a single self-contained HTML page (onboard + generate forms, aspect-ratio picker, all
override fields, inline video playback) and proxies its actions to the existing /host/onboard
and /generate endpoints. The sensitive credentials (Modal proxy Key/Secret) live only here,
server-side — the browser never sees them. The page itself is gated by APP_BEARER_TOKEN.

Env it needs (from .env via from_dotenv, or named secrets):
  APP_BEARER_TOKEN            — the page password + the app-level bearer sent upstream
  MODAL_PROXY_KEY             — proxy-auth token id (wk-...), sent as Modal-Key upstream
  MODAL_PROXY_SECRET          — proxy-auth token secret (ws-...), sent as Modal-Secret upstream
  ONBOARD_URL / GENERATE_URL  — the deployed endpoint base URLs (from `modal deploy` output)
"""

import os

import modal

from app import app, frontend_image, secrets_for


def _verify_token(x_app_token: str | None):
    from fastapi import HTTPException

    expected = os.environ.get("APP_BEARER_TOKEN")
    if not expected:
        # No token configured — allow (dev only). In practice APP_BEARER_TOKEN is always set.
        return
    if x_app_token != expected:
        raise HTTPException(status_code=401, detail="Wrong password (APP_BEARER_TOKEN).")


def _upstream_headers() -> dict:
    """Headers for calling the proxy-auth-protected pipeline endpoints."""
    h = {}
    key, sec = os.environ.get("MODAL_PROXY_KEY"), os.environ.get("MODAL_PROXY_SECRET")
    if key and sec:
        h["Modal-Key"] = key
        h["Modal-Secret"] = sec
    token = os.environ.get("APP_BEARER_TOKEN")
    if token:
        h["Authorization"] = f"Bearer {token}"
    return h


@app.function(
    image=frontend_image,
    # In dotenv mode this ships all .env vars (APP_BEARER_TOKEN, MODAL_PROXY_KEY/SECRET,
    # ONBOARD_URL, GENERATE_URL). In named mode you'd add a secret carrying those.
    secrets=secrets_for("proxy-auth-token"),
    timeout=2700,  # generation can take minutes; keep the proxy request alive
)
@modal.asgi_app()  # public page; actions are gated by the password (APP_BEARER_TOKEN) in-app
def control_panel():
    import requests
    from fastapi import FastAPI, File, Form, Header, Request, UploadFile
    from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse

    web = FastAPI(title="EasyWebinar Avatar — Demo Control Panel")

    onboard_url = os.environ.get("ONBOARD_URL", "").rstrip("/")
    generate_url = os.environ.get("GENERATE_URL", "").rstrip("/")

    @web.get("/", response_class=HTMLResponse)
    async def index():
        return _PAGE

    @web.post("/api/onboard")
    async def api_onboard(
        x_app_token: str = Header(None),
        host_id: str = Form(...),
        consent_attested: str = Form(...),
        video: UploadFile = File(...),
        voice_character_hint: str = Form(None),
        scene_prompt: str = Form(None),
    ):
        _verify_token(x_app_token)
        if not onboard_url:
            return JSONResponse(status_code=500, content={"error": "ONBOARD_URL not configured"})
        content = await video.read()
        data = {"host_id": host_id, "consent_attested": consent_attested}
        if voice_character_hint:
            data["voice_character_hint"] = voice_character_hint
        if scene_prompt:
            data["scene_prompt"] = scene_prompt
        try:
            r = requests.post(
                f"{onboard_url}/host/onboard",
                headers=_upstream_headers(),
                data=data,
                files={"video": (video.filename, content, video.content_type or "video/mp4")},
                timeout=600,
            )
            return JSONResponse(status_code=r.status_code, content=_safe_json(r))
        except Exception as e:  # noqa: BLE001
            return JSONResponse(status_code=502, content={"error": repr(e)})

    @web.post("/api/reference/{host_id}")
    async def api_reference(host_id: str, request: Request, x_app_token: str = Header(None)):
        """Re-roll the studio reference with a new scene prompt (no voice re-clone)."""
        _verify_token(x_app_token)
        if not onboard_url:
            return JSONResponse(status_code=500, content={"error": "ONBOARD_URL not configured"})
        payload = await request.json()
        try:
            r = requests.post(
                f"{onboard_url}/host/{host_id}/reference",
                headers=_upstream_headers(),
                json=payload,
                timeout=600,
            )
            return JSONResponse(status_code=r.status_code, content=_safe_json(r))
        except Exception as e:  # noqa: BLE001
            return JSONResponse(status_code=502, content={"error": repr(e)})

    @web.post("/api/onboard_url")
    async def api_onboard_url(request: Request, x_app_token: str = Header(None)):
        """JSON onboarding (no file) — forwards to the server-side URL-download endpoint."""
        _verify_token(x_app_token)
        if not onboard_url:
            return JSONResponse(status_code=500, content={"error": "ONBOARD_URL not configured"})
        payload = await request.json()
        try:
            r = requests.post(
                f"{onboard_url}/host/onboard-url",
                headers=_upstream_headers(),
                json=payload,
                timeout=600,
            )
            return JSONResponse(status_code=r.status_code, content=_safe_json(r))
        except Exception as e:  # noqa: BLE001
            return JSONResponse(status_code=502, content={"error": repr(e)})

    @web.get("/api/profile/{host_id}")
    async def api_profile(host_id: str, x_app_token: str = Header(None)):
        _verify_token(x_app_token)
        try:
            r = requests.get(f"{generate_url}/profile/{host_id}",
                             headers=_upstream_headers(), timeout=60)
            return JSONResponse(status_code=r.status_code, content=_safe_json(r))
        except Exception as e:  # noqa: BLE001
            return JSONResponse(status_code=502, content={"error": repr(e)})

    @web.get("/api/hosts")
    async def api_hosts(x_app_token: str = Header(None)):
        _verify_token(x_app_token)
        try:
            r = requests.get(f"{generate_url}/hosts", headers=_upstream_headers(), timeout=60)
            return JSONResponse(status_code=r.status_code, content=_safe_json(r))
        except Exception as e:  # noqa: BLE001
            return JSONResponse(status_code=502, content={"error": repr(e)})

    @web.post("/api/generate")
    async def api_generate(request: Request, x_app_token: str = Header(None)):
        """Submit a generation job — returns {job_id} fast (Wan runs async)."""
        _verify_token(x_app_token)
        if not generate_url:
            return JSONResponse(status_code=500, content={"error": "GENERATE_URL not configured"})
        payload = await request.json()
        try:
            r = requests.post(
                f"{generate_url}/generate",
                headers=_upstream_headers(),
                json=payload,
                timeout=120,
            )
            return JSONResponse(status_code=r.status_code, content=_safe_json(r))
        except Exception as e:  # noqa: BLE001
            return JSONResponse(status_code=502, content={"error": repr(e)})

    @web.get("/api/status/{job_id}")
    async def api_status(job_id: str, x_app_token: str = Header(None)):
        """Poll a generation job: pending | done (+ base64 video) | error."""
        _verify_token(x_app_token)
        if not generate_url:
            return JSONResponse(status_code=500, content={"error": "GENERATE_URL not configured"})
        try:
            r = requests.get(
                f"{generate_url}/status/{job_id}",
                headers=_upstream_headers(),
                timeout=120,
            )
            return JSONResponse(status_code=r.status_code, content=_safe_json(r))
        except Exception as e:  # noqa: BLE001
            return JSONResponse(status_code=502, content={"error": repr(e)})

    @web.get("/api/video/{job_id}")
    async def api_video(job_id: str, x_app_token: str = Header(None)):
        """Stream the finished mp4 from the pipeline (binary, not base64) so any size renders."""
        _verify_token(x_app_token)
        if not generate_url:
            return JSONResponse(status_code=500, content={"error": "GENERATE_URL not configured"})
        try:
            r = requests.get(f"{generate_url}/video/{job_id}", headers=_upstream_headers(),
                             timeout=300, stream=True)
            if r.status_code >= 400:
                return JSONResponse(status_code=r.status_code, content=_safe_json(r))
            return StreamingResponse(r.iter_content(chunk_size=1024 * 1024), media_type="video/mp4")
        except Exception as e:  # noqa: BLE001
            return JSONResponse(status_code=502, content={"error": repr(e)})

    def _safe_json(resp):
        try:
            return resp.json()
        except Exception:  # noqa: BLE001
            return {"error": f"upstream {resp.status_code}", "raw": resp.text[:1000]}

    return web


# ---------------------------------------------------------------------------
# The page. Self-contained (inline CSS/JS). Kept deliberately simple.
# ---------------------------------------------------------------------------
_PAGE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>AI Avatar — Demo Control Panel</title>
<style>
  :root { color-scheme: dark; }
  * { box-sizing: border-box; }
  body { margin:0; font-family: system-ui, -apple-system, Segoe UI, Roboto, sans-serif;
         background:#0d1117; color:#e6edf3; line-height:1.5; }
  header { padding:20px 24px; border-bottom:1px solid #21262d; display:flex; align-items:center;
           gap:16px; flex-wrap:wrap; }
  header h1 { font-size:18px; margin:0; font-weight:600; }
  .pw { margin-left:auto; display:flex; gap:8px; align-items:center; }
  .wrap { max-width:1100px; margin:0 auto; padding:24px; display:grid;
          grid-template-columns:1fr 1fr; gap:24px; }
  @media (max-width:820px){ .wrap{ grid-template-columns:1fr; } }
  .card { background:#161b22; border:1px solid #21262d; border-radius:12px; padding:20px; }
  .card h2 { margin:0 0 4px; font-size:16px; }
  .card p.sub { margin:0 0 16px; color:#8b949e; font-size:13px; }
  label { display:block; font-size:13px; margin:12px 0 4px; color:#c9d1d9; }
  input[type=text], input[type=password], textarea, select, input[type=file] {
    width:100%; padding:9px 11px; background:#0d1117; border:1px solid #30363d;
    border-radius:8px; color:#e6edf3; font-size:14px; }
  textarea { min-height:90px; resize:vertical; font-family:inherit; }
  .row { display:flex; gap:12px; } .row > * { flex:1; }
  .check { display:flex; align-items:center; gap:8px; margin-top:12px; font-size:14px; }
  .check input { width:auto; }
  button { margin-top:18px; width:100%; padding:11px; border:0; border-radius:8px;
           background:#238636; color:#fff; font-size:14px; font-weight:600; cursor:pointer; }
  button:hover { background:#2ea043; } button:disabled { background:#30363d; cursor:not-allowed; }
  .out { margin-top:16px; font-size:13px; white-space:pre-wrap; word-break:break-word;
         background:#0d1117; border:1px solid #30363d; border-radius:8px; padding:12px;
         min-height:20px; max-height:260px; overflow:auto; }
  .ok { color:#3fb950; } .err { color:#f85149; } .muted { color:#8b949e; }
  video { width:100%; margin-top:14px; border-radius:8px; border:1px solid #30363d; display:none; }
  a.dl { display:none; margin-top:10px; font-size:13px; color:#58a6ff; }
  img.preview { width:100%; max-width:320px; margin-top:14px; border-radius:8px;
                border:1px solid #30363d; display:none; }
  .previewwrap { display:flex; gap:14px; flex-wrap:wrap; align-items:flex-start; }
  .previewwrap figure { margin:0; } .previewwrap figcaption { font-size:12px; color:#8b949e; }
  button.secondary { background:#1f6feb; } button.secondary:hover { background:#388bfd; }
</style>
</head>
<body>
<header>
  <h1>🎬 AI Avatar — Demo Control Panel</h1>
  <div class="pw">
    <input type="password" id="pw" placeholder="Access password (APP_BEARER_TOKEN)" style="width:280px"/>
  </div>
</header>
<div class="wrap">

  <div class="card">
    <h2>1 · Onboard a host</h2>
    <p class="sub">One-time per host. Uploads a short video → clones the voice → reimagines a clean studio reference → saves the profile.</p>
    <label>Host ID</label>
    <input type="text" id="o_host" placeholder="e.g. host_123" value="host_demo_1"/>
    <label>Video (short clip, clear face + clean audio)</label>
    <input type="file" id="o_video" accept="video/*"/>
    <label>— or — Video URL (server-side download; use this if uploads are blocked)</label>
    <input type="text" id="o_url" placeholder="https://www.youtube.com/watch?v=..."/>
    <label>Clip range for URL download (start → end, max 3 min slice)</label>
    <div class="row">
      <div>
        <span class="muted" style="font-size:12px">Start: <b id="o_start_lbl">0:00</b></span>
        <input type="range" id="o_start" min="0" max="600" value="0" step="1" oninput="updRange()"/>
      </div>
      <div>
        <span class="muted" style="font-size:12px">End: <b id="o_end_lbl">1:30</b></span>
        <input type="range" id="o_end" min="1" max="600" value="90" step="1" oninput="updRange()"/>
      </div>
    </div>
    <label>Studio scene / background (optional — the AI reference is re-staged into this)</label>
    <textarea id="o_scene" placeholder="e.g. seated at a modern podcast desk with a microphone, brand-blue backdrop, soft studio lighting" style="min-height:60px"></textarea>
    <label>Voice character hint (optional)</label>
    <input type="text" id="o_hint" placeholder="e.g. warm, calm narrator"/>
    <div class="check">
      <input type="checkbox" id="o_consent"/>
      <label for="o_consent" style="margin:0">I confirm I have the right/consent to clone this voice.</label>
    </div>
    <button id="o_btn" onclick="onboard()">Onboard host (upload file)</button>
    <button id="o_btn_url" onclick="onboardUrl()" style="background:#1f6feb">Onboard host (from URL)</button>
    <div class="out muted" id="o_out">No request sent yet.</div>
    <div class="previewwrap">
      <figure><img class="preview" id="o_preview"/><figcaption id="o_preview_cap"></figcaption></figure>
    </div>
    <div id="o_reroll" style="display:none">
      <label>Not happy with the reference? Re-stage it (new scene, same voice):</label>
      <textarea id="o_scene2" placeholder="describe a different background/scene" style="min-height:50px"></textarea>
      <button class="secondary" id="o_reroll_btn" onclick="rerollReference()">🔄 Re-roll studio reference</button>
    </div>
  </div>

  <div class="card">
    <h2>2 · Generate a video</h2>
    <p class="sub">Uses only the host_id — the stored profile supplies the voice + studio reference. Wan2.2-S2V renders async (minutes); this polls until it's ready.</p>
    <label>Host ID <a href="#" onclick="listHosts();return false" style="color:#58a6ff;font-size:12px;margin-left:8px">(list onboarded hosts)</a></label>
    <input type="text" id="g_host" placeholder="same host_id you onboarded" value="host_demo_1"/>
    <button id="g_prof" onclick="viewProfile()" style="background:#30363d;margin-top:8px">🔍 View this host's stored profile</button>
    <label>Script</label>
    <textarea id="g_script" placeholder="Type the narration script here...">Welcome to today's webinar on serverless AI pipelines. We're excited to show you what's possible.</textarea>
    <div class="row">
      <div>
        <label>Aspect ratio</label>
        <select id="g_ratio">
          <option value="16:9">16:9 (landscape)</option>
          <option value="9:16">9:16 (vertical / reels)</option>
          <option value="1:1">1:1 (square)</option>
        </select>
      </div>
      <div>
        <label>Voice character hint override (optional)</label>
        <input type="text" id="g_hint" placeholder="overrides the onboarded hint"/>
      </div>
    </div>
    <label>Scene prompt override (optional — describes the visual scene/motion)</label>
    <input type="text" id="g_wanprompt" placeholder="e.g. speaking to camera in a bright studio, subtle head movement"/>
    <div class="row">
      <div>
        <label>TTS provider (A/B test)</label>
        <select id="g_tts" onchange="ttsToggle()">
          <option value="elevenlabs">ElevenLabs (v3, uses profile voice)</option>
          <option value="fishaudio">Fish Audio (s2.1-pro)</option>
        </select>
      </div>
      <div>
        <label>Fish voice id (reference_id) — Fish Audio only</label>
        <input type="text" id="g_fishref" placeholder="Fish reference_id" disabled/>
      </div>
    </div>
    <div class="check">
      <input type="checkbox" id="g_director" checked/>
      <label for="g_director" style="margin:0">Enable LLM audio director (ElevenLabs only; ignored for Fish)</label>
    </div>
    <button id="g_btn" onclick="generate()">Generate video</button>
    <div class="out muted" id="g_out">No request sent yet.</div>
    <video id="g_video" controls></video>
    <a class="dl" id="g_dl" download="avatar.mp4">⬇ Download video</a>
  </div>

</div>
<script>
const pw = () => document.getElementById('pw').value.trim();
function setOut(id, msg, cls){ const el=document.getElementById(id); el.className='out '+(cls||''); el.textContent=msg; }
function showReference(j){
  // Display the reimagined studio portrait (if the backend produced one) + the re-roll controls.
  const img=document.getElementById('o_preview'), cap=document.getElementById('o_preview_cap');
  if(j.reference_preview_b64){
    img.src='data:image/png;base64,'+j.reference_preview_b64; img.style.display='block';
    cap.textContent='AI studio reference — this is what will speak your scripts.';
    document.getElementById('o_reroll').style.display='block';
  } else {
    img.style.display='none'; cap.textContent='';
    if(j.reference_error){ cap.textContent='⚠ reference not generated: '+j.reference_error; }
  }
}
function ttsToggle(){
  const isFish = document.getElementById('g_tts').value==='fishaudio';
  document.getElementById('g_fishref').disabled = !isFish;
}
function fmt(s){ s=Math.round(s); return Math.floor(s/60)+':'+String(s%60).padStart(2,'0'); }
function updRange(){
  let a=+document.getElementById('o_start').value, b=+document.getElementById('o_end').value;
  if(b<=a){ b=a+1; document.getElementById('o_end').value=b; }
  if(b-a>180){ b=a+180; document.getElementById('o_end').value=b; }   // enforce max 3-min slice
  document.getElementById('o_start_lbl').textContent=fmt(a);
  document.getElementById('o_end_lbl').textContent=fmt(b)+'  ('+(b-a)+'s)';
}

async function onboard(){
  if(!pw()){ setOut('o_out','Enter the access password up top first.','err'); return; }
  const f = document.getElementById('o_video').files[0];
  if(!f){ setOut('o_out','Choose a video file.','err'); return; }
  if(!document.getElementById('o_consent').checked){ setOut('o_out','You must tick the consent box.','err'); return; }
  const fd = new FormData();
  fd.append('host_id', document.getElementById('o_host').value.trim());
  fd.append('consent_attested', 'true');
  fd.append('video', f);
  const hint = document.getElementById('o_hint').value.trim();
  const scene = document.getElementById('o_scene').value.trim();
  if(hint) fd.append('voice_character_hint', hint);
  if(scene) fd.append('scene_prompt', scene);
  const btn=document.getElementById('o_btn'); btn.disabled=true;
  setOut('o_out','Uploading + cloning voice + reimagining a studio reference… (~1-2 min)','muted');
  try{
    const r = await fetch('/api/onboard', { method:'POST', headers:{'x-app-token':pw()}, body:fd });
    const j = await r.json();
    if(r.ok && j.status==='success'){
      setOut('o_out','✓ Onboarded.\\nvoice_id: '+j.voice_id+'\\nrequires_verification: '+j.requires_verification+(j.reference_error?'\\n⚠ '+j.reference_error:''), 'ok');
      document.getElementById('g_host').value = document.getElementById('o_host').value.trim();
      showReference(j);
    } else {
      setOut('o_out','✗ '+JSON.stringify(j,null,2),'err');
    }
  }catch(e){ setOut('o_out','✗ '+e,'err'); }
  btn.disabled=false;
}

async function onboardUrl(){
  if(!pw()){ setOut('o_out','Enter the access password up top first.','err'); return; }
  const url = document.getElementById('o_url').value.trim();
  if(!url){ setOut('o_out','Paste a video URL (or use the file-upload button instead).','err'); return; }
  if(!document.getElementById('o_consent').checked){ setOut('o_out','You must tick the consent box.','err'); return; }
  const payload = {
    host_id: document.getElementById('o_host').value.trim(),
    video_url: url,
    consent_attested: true,
    start_sec: +document.getElementById('o_start').value,
    end_sec: +document.getElementById('o_end').value
  };
  const hint = document.getElementById('o_hint').value.trim();
  const scene = document.getElementById('o_scene').value.trim();
  if(hint) payload.voice_character_hint = hint;
  if(scene) payload.scene_prompt = scene;
  const btn=document.getElementById('o_btn_url'); btn.disabled=true;
  setOut('o_out','Downloading the video on Modal + cloning voice + reimagining a studio reference… (~2 min)','muted');
  try{
    const r = await fetch('/api/onboard_url', { method:'POST', headers:{'x-app-token':pw(),'content-type':'application/json'}, body:JSON.stringify(payload) });
    const j = await r.json();
    if(r.ok && j.status==='success'){
      setOut('o_out','✓ Onboarded.\\nvoice_id: '+j.voice_id+'\\nrequires_verification: '+j.requires_verification+(j.reference_error?'\\n⚠ '+j.reference_error:''), 'ok');
      document.getElementById('g_host').value = document.getElementById('o_host').value.trim();
      showReference(j);
    } else {
      setOut('o_out','✗ stage='+(j.stage||'?')+'\\n'+JSON.stringify(j,null,2),'err');
    }
  }catch(e){ setOut('o_out','✗ '+e,'err'); }
  btn.disabled=false;
}

async function rerollReference(){
  if(!pw()){ setOut('o_out','Enter the access password up top first.','err'); return; }
  const h = document.getElementById('o_host').value.trim();
  const scene = document.getElementById('o_scene2').value.trim();
  const btn=document.getElementById('o_reroll_btn'); btn.disabled=true;
  setOut('o_out','Re-staging the studio reference for '+h+'… (~1 min)','muted');
  try{
    const r = await fetch('/api/reference/'+encodeURIComponent(h), { method:'POST', headers:{'x-app-token':pw(),'content-type':'application/json'}, body:JSON.stringify({scene_prompt:scene}) });
    const j = await r.json();
    if(r.ok && j.status==='success'){
      setOut('o_out','✓ New studio reference staged.\\nscene: '+j.scene, 'ok');
      showReference(j);
    } else { setOut('o_out','✗ '+JSON.stringify(j,null,2),'err'); }
  }catch(e){ setOut('o_out','✗ '+e,'err'); }
  btn.disabled=false;
}

async function viewProfile(){
  if(!pw()){ setOut('g_out','Enter the access password up top first.','err'); return; }
  const h = document.getElementById('g_host').value.trim();
  setOut('g_out','Looking up profile for '+h+'…','muted');
  try{
    const r = await fetch('/api/profile/'+encodeURIComponent(h), { headers:{'x-app-token':pw()} });
    const j = await r.json();
    if(r.ok && j.status==='success'){
      setOut('g_out','Profile for '+h+':\\n'
        +'  voice_id: '+j.voice_id+'\\n'
        +'  base_video_path: '+j.base_video_path+'\\n'
        +'  reference_image_path: '+(j.reference_image_path||'(none — run set_reference / re-onboard)')+'\\n'
        +'  voice_character_hint: '+(j.voice_character_hint||'(none)')+'\\n'
        +'  created_at: '+j.created_at, 'ok');
    } else { setOut('g_out','✗ '+JSON.stringify(j,null,2),'err'); }
  }catch(e){ setOut('g_out','✗ '+e,'err'); }
}

async function listHosts(){
  if(!pw()){ setOut('g_out','Enter the access password up top first.','err'); return; }
  try{
    const r = await fetch('/api/hosts', { headers:{'x-app-token':pw()} });
    const j = await r.json();
    if(r.ok && j.status==='success'){
      setOut('g_out','Onboarded hosts ('+j.host_ids.length+'):\\n  '+(j.host_ids.join('\\n  ')||'(none yet)'),'ok');
    } else { setOut('g_out','✗ '+JSON.stringify(j,null,2),'err'); }
  }catch(e){ setOut('g_out','✗ '+e,'err'); }
}

let _pollTimer=null;
async function generate(){
  if(!pw()){ setOut('g_out','Enter the access password up top first.','err'); return; }
  const payload = {
    host_id: document.getElementById('g_host').value.trim(),
    script_text: document.getElementById('g_script').value,
    target_ratio: document.getElementById('g_ratio').value,
    enable_director: document.getElementById('g_director').checked,
    tts_provider: document.getElementById('g_tts').value
  };
  const hint=document.getElementById('g_hint').value.trim();
  const wp=document.getElementById('g_wanprompt').value.trim();
  const fref=document.getElementById('g_fishref').value.trim();
  if(hint) payload.voice_character_hint=hint;
  if(wp) payload.wan_prompt=wp;
  if(payload.tts_provider==='fishaudio' && fref) payload.fish_reference_id=fref;
  const btn=document.getElementById('g_btn'); btn.disabled=true;
  if(_pollTimer){ clearInterval(_pollTimer); _pollTimer=null; }
  document.getElementById('g_video').style.display='none';
  document.getElementById('g_dl').style.display='none';
  setOut('g_out','Submitting job… (director → TTS → Wan2.2-S2V). Rendering runs async on the GPU and can take SEVERAL MINUTES — this will poll automatically. You can leave the tab open.','muted');
  try{
    const r = await fetch('/api/generate', { method:'POST', headers:{'x-app-token':pw(),'content-type':'application/json'}, body:JSON.stringify(payload) });
    const j = await r.json();
    if(!(r.ok && (j.status==='submitted'||j.job_id))){
      setOut('g_out','✗ stage='+(j.stage||'?')+'\\n'+JSON.stringify(j,null,2),'err'); btn.disabled=false; return;
    }
    pollJob(j.job_id, Date.now());
  }catch(e){ setOut('g_out','✗ '+e,'err'); btn.disabled=false; }
}

function pollJob(jobId, t0){
  const btn=document.getElementById('g_btn');
  const tick = async () => {
    try{
      const r = await fetch('/api/status/'+encodeURIComponent(jobId), { headers:{'x-app-token':pw()} });
      const j = await r.json();
      const secs = Math.round((Date.now()-t0)/1000);
      if(j.status==='pending'){
        setOut('g_out','⏳ Rendering on the GPU… '+secs+'s elapsed (job '+jobId.slice(0,8)+'). Polling every 5s.','muted');
        return;
      }
      clearInterval(_pollTimer); _pollTimer=null; btn.disabled=false;
      if(j.status==='done'){
        setOut('g_out','✓ Done in ~'+secs+'s — '+j.resolution+', '+j.duration_seconds+'s\\nTTS: '+(j.tts_provider||'?')+' ('+j.tts_model+')\\n\\ndirected_text:\\n'+j.directed_text, 'ok');
        // Fetch the video as a binary blob (streamed, with auth) — robust for any size.
        try{
          const vr = await fetch('/api/video/'+encodeURIComponent(jobId), { headers:{'x-app-token':pw()} });
          if(!vr.ok){ setOut('g_out','✓ rendered, but video fetch failed (HTTP '+vr.status+'). Pull it from the volume: '+(j.video_volume_path||''),'err'); return; }
          const url = URL.createObjectURL(await vr.blob());
          const v=document.getElementById('g_video'); v.src=url; v.style.display='block';
          const dl=document.getElementById('g_dl'); dl.href=url; dl.style.display='inline-block';
          // Make the result unmissable in a live demo: jump to it + autoplay.
          v.scrollIntoView({behavior:'smooth', block:'center'});
          v.muted=false; v.play().catch(()=>{});
        }catch(e){ setOut('g_out','✓ rendered, but video load error: '+e,'err'); }
      } else {
        setOut('g_out','✗ stage='+(j.stage||'?')+'\\n'+JSON.stringify(j,null,2),'err');
      }
    }catch(e){ /* transient — keep polling */ }
  };
  tick();
  _pollTimer=setInterval(tick, 5000);
}
</script>
</body>
</html>"""
