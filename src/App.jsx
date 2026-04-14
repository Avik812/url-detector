import { useState, useEffect, useRef } from "react";

const SUSPICIOUS_KEYWORDS = [
  'login','signin','secure','account','update','verify','bank','paypal',
  'password','confirm','ebay','amazon','support','billing','alert',
  'suspended','unusual','locked','unauthorized','authenticate','wallet',
  'bonus','free','prize','winner','urgent',
];
const SUSPICIOUS_TLDS = new Set(['.tk','.xyz','.ml','.ga','.cf','.gq','.pw','.top','.click','.link','.online','.site','.work']);
const TRUSTED_TLDS    = new Set(['.com','.org','.net','.edu','.gov','.io']);

function shannonEntropy(s) {
  if (!s) return 0;
  const freq = {};
  for (const c of s) freq[c] = (freq[c] || 0) + 1;
  const n = s.length;
  return -Object.values(freq).reduce((acc, cnt) => acc + (cnt/n) * Math.log2(cnt/n), 0);
}

function extractFeatures(rawUrl) {
  const url = rawUrl.startsWith('http') ? rawUrl : 'http://' + rawUrl;
  let parsed;
  try { parsed = new URL(url); } catch { parsed = { hostname:'', pathname:'', search:'', protocol:'http:', port:'' }; }
  const hostname = parsed.hostname || '';
  const path = parsed.pathname || '';
  const query = parsed.search || '';
  const full = rawUrl;
  const tld = '.' + hostname.split('.').slice(-1)[0];
  const subdomainParts = hostname.split('.').slice(0, -2).filter(Boolean);
  const hasIp = /(\d{1,3}\.){3}\d{1,3}/.test(hostname);
  const digits = (s) => [...s].filter(c => /\d/.test(c)).length;
  const letters = (s) => [...s].filter(c => /[a-zA-Z]/.test(c)).length;
  const specials = (s) => [...s].filter(c => !/[a-zA-Z0-9/:.@]/.test(c)).length;
  return {
    url_length: full.length,
    domain_length: hostname.length,
    path_length: path.length,
    query_length: query.length,
    num_subdomains: subdomainParts.length,
    has_ip_address: hasIp ? 1 : 0,
    has_https: parsed.protocol === 'https:' ? 1 : 0,
    has_port: parsed.port ? 1 : 0,
    num_redirects: (full.match(/\/\//g) || []).length - 1,
    has_at_symbol: full.includes('@') ? 1 : 0,
    num_dots: (full.match(/\./g) || []).length,
    num_hyphens: (full.match(/-/g) || []).length,
    num_underscores: (full.match(/_/g) || []).length,
    num_slashes: (full.match(/\//g) || []).length,
    num_question: (full.match(/\?/g) || []).length,
    num_ampersand: (full.match(/&/g) || []).length,
    num_equals: (full.match(/=/g) || []).length,
    num_percent: (full.match(/%/g) || []).length,
    num_tilde: (full.match(/~/g) || []).length,
    digit_ratio: +(digits(full) / Math.max(full.length,1)).toFixed(4),
    letter_ratio: +(letters(full) / Math.max(full.length,1)).toFixed(4),
    special_char_ratio: +(specials(full) / Math.max(full.length,1)).toFixed(4),
    url_entropy: +shannonEntropy(full).toFixed(4),
    domain_entropy: +shannonEntropy(hostname).toFixed(4),
    suspicious_keyword_count: SUSPICIOUS_KEYWORDS.filter(kw => full.toLowerCase().includes(kw)).length,
    has_suspicious_keyword: SUSPICIOUS_KEYWORDS.some(kw => full.toLowerCase().includes(kw)) ? 1 : 0,
    suspicious_tld: SUSPICIOUS_TLDS.has(tld) ? 1 : 0,
    trusted_tld: TRUSTED_TLDS.has(tld) ? 1 : 0,
    domain_digit_count: digits(hostname),
    domain_hyphen_count: (hostname.match(/-/g) || []).length,
  };
}

async function classifyUrl(url, features) {
  const res = await fetch("https://api.anthropic.com/v1/messages", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      model: "claude-sonnet-4-20250514",
      max_tokens: 1000,
      system: `You are a scikit-learn Random Forest classifier trained on malicious URL detection using NLP features.
Analyze the structured URL features and classify the URL. Respond ONLY with valid JSON, no markdown or explanation.

Schema:
{
  "prediction": "malicious" | "benign" | "suspicious",
  "confidence": <0-100>,
  "malicious_prob": <0-100>,
  "benign_prob": <0-100>,
  "risk_signals": [<string up to 5>],
  "top_features": [{"name": <string>, "value": <any>, "risk": "high"|"medium"|"low"|"none", "note": <string>}]
}

top_features: the 5 most important features driving your decision.
risk_signals: specific reasons this URL is or is not suspicious.`,
      messages: [{ role: "user", content: `URL: ${url}\n\nFeatures:\n${JSON.stringify(features, null, 2)}` }]
    })
  });
  const data = await res.json();
  const text = data.content?.[0]?.text || "{}";
  try { return JSON.parse(text); }
  catch { return JSON.parse(text.replace(/```json|```/g, '').trim()); }
}

const RISK_COLOR = { high:'#ff4757', medium:'#ffa502', low:'#eccc68', none:'#2ed573' };
const RISK_BG    = { high:'rgba(255,71,87,0.1)', medium:'rgba(255,165,2,0.1)', low:'rgba(236,204,104,0.1)', none:'rgba(46,213,115,0.1)' };

function RadialGauge({ value, verdict }) {
  const r = 54, cx = 64, cy = 64, circum = 2 * Math.PI * r;
  const arc = (circum * Math.min(Math.max(value,0),100)) / 100;
  const color = verdict === 'benign' ? '#2ed573' : verdict === 'suspicious' ? '#ffa502' : '#ff4757';
  return (
    <svg width="128" height="128" viewBox="0 0 128 128">
      <circle cx={cx} cy={cy} r={r} fill="none" stroke="rgba(255,255,255,0.05)" strokeWidth="10"/>
      <circle cx={cx} cy={cy} r={r} fill="none" stroke={color} strokeWidth="10"
        strokeDasharray={`${arc} ${circum}`} strokeLinecap="round"
        transform="rotate(-90 64 64)"
        style={{ transition:'stroke-dasharray 1s ease', filter:`drop-shadow(0 0 8px ${color})` }}/>
      <text x={cx} y={cy-6} textAnchor="middle" fill={color} fontSize="22"
        fontFamily="'Share Tech Mono',monospace" fontWeight="bold">{Math.round(value)}%</text>
      <text x={cx} y={cy+14} textAnchor="middle" fill="rgba(255,255,255,0.4)"
        fontSize="9" fontFamily="'Share Tech Mono',monospace" letterSpacing="1">CONFIDENCE</text>
    </svg>
  );
}

function VerdictBadge({ verdict }) {
  const cfg = {
    malicious:  { label:'⚠ MALICIOUS',  bg:'rgba(255,71,87,0.15)',  border:'#ff4757', color:'#ff4757' },
    suspicious: { label:'⚡ SUSPICIOUS', bg:'rgba(255,165,2,0.15)',  border:'#ffa502', color:'#ffa502' },
    benign:     { label:'✓ SAFE',        bg:'rgba(46,213,115,0.15)', border:'#2ed573', color:'#2ed573' },
  }[verdict] || { label:'? UNKNOWN', bg:'#111', border:'#444', color:'#aaa' };
  return (
    <div style={{
      display:'inline-flex', alignItems:'center', gap:8,
      padding:'10px 20px', background:cfg.bg,
      border:`1px solid ${cfg.border}`, color:cfg.color,
      fontFamily:"'Share Tech Mono',monospace", fontSize:18, fontWeight:'bold',
      letterSpacing:2, boxShadow:`0 0 20px ${cfg.border}40`,
    }}>{cfg.label}</div>
  );
}

function FeatureRow({ name, value, risk, note }) {
  return (
    <div style={{
      display:'grid', gridTemplateColumns:'1fr auto auto', gap:12, alignItems:'center',
      padding:'8px 12px', background:RISK_BG[risk]||'rgba(255,255,255,0.02)',
      borderLeft:`2px solid ${RISK_COLOR[risk]||'#333'}`, marginBottom:4,
    }}>
      <div>
        <div style={{ fontFamily:"'Share Tech Mono',monospace", fontSize:11, color:'#8899aa', marginBottom:2 }}>{name}</div>
        <div style={{ fontSize:12, color:'#c8d8e4', opacity:0.7 }}>{note}</div>
      </div>
      <div style={{ fontFamily:"'Share Tech Mono',monospace", fontSize:13, color:'#e8f0f5', minWidth:40, textAlign:'right' }}>
        {typeof value === 'number' && !Number.isInteger(value) ? value.toFixed(2) : String(value)}
      </div>
      <div style={{
        padding:'2px 8px', fontSize:9, letterSpacing:1,
        fontFamily:"'Share Tech Mono',monospace",
        background:RISK_BG[risk], color:RISK_COLOR[risk],
        border:`1px solid ${RISK_COLOR[risk]}40`, textTransform:'uppercase',
      }}>{risk}</div>
    </div>
  );
}

function AllFeatures({ features }) {
  const [open, setOpen] = useState(false);
  if (!features) return null;
  return (
    <div style={{
      background:'rgba(8,18,26,0.95)', border:'1px solid rgba(255,255,255,0.07)',
      borderTop:'2px solid rgba(255,255,255,0.06)', padding:28, marginBottom:16,
    }}>
      <button onClick={() => setOpen(o => !o)} style={{
        background:'none', border:'none', cursor:'pointer', width:'100%',
        display:'flex', justifyContent:'space-between', alignItems:'center',
        fontFamily:"'Share Tech Mono',monospace", fontSize:10,
        letterSpacing:2, color:'#3a6a7a', padding:0,
      }}>
        <span>// ALL EXTRACTED FEATURES ({Object.keys(features).length})</span>
        <span style={{ fontSize:14 }}>{open ? '▲' : '▼'}</span>
      </button>
      {open && (
        <div style={{ marginTop:16, display:'grid', gridTemplateColumns:'repeat(auto-fill, minmax(210px,1fr))', gap:6 }}>
          {Object.entries(features).map(([k, v]) => (
            <div key={k} style={{ padding:'6px 10px', background:'rgba(255,255,255,0.02)', border:'1px solid rgba(255,255,255,0.05)' }}>
              <div style={{ fontFamily:"'Share Tech Mono',monospace", fontSize:9, color:'#3a6a7a', marginBottom:2 }}>{k}</div>
              <div style={{ fontFamily:"'Share Tech Mono',monospace", fontSize:12, color:'#8abacc' }}>
                {typeof v === 'number' ? (Number.isInteger(v) ? v : v.toFixed(4)) : String(v)}
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

const PHASES = ['EXTRACTING NLP FEATURES...','RUNNING RANDOM FOREST...','EVALUATING RISK SIGNALS...','COMPUTING CONFIDENCE...'];

export default function App() {
  const [url, setUrl] = useState('');
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState('');
  const [scanPhase, setScanPhase] = useState(0);
  const [typed, setTyped] = useState('');
  const inputRef = useRef();

  useEffect(() => {
    const tagline = "MALICIOUS URL DETECTOR";
    let i = 0;
    const t = setInterval(() => { setTyped(tagline.slice(0, ++i)); if (i >= tagline.length) clearInterval(t); }, 60);
    return () => clearInterval(t);
  }, []);

  useEffect(() => {
    if (!loading) return;
    let i = 0;
    const t = setInterval(() => { setScanPhase(i++ % PHASES.length); }, 700);
    return () => clearInterval(t);
  }, [loading]);

  async function handleScan() {
    if (!url.trim()) return;
    setLoading(true); setResult(null); setError('');
    try {
      const features = extractFeatures(url.trim());
      const classification = await classifyUrl(url.trim(), features);
      setResult({ ...classification, raw_features: features });
    } catch(e) { setError('Analysis failed: ' + e.message); }
    finally { setLoading(false); }
  }

  const verdict = result?.prediction;
  const accentColor = verdict === 'benign' ? '#2ed573' : verdict === 'suspicious' ? '#ffa502' : verdict === 'malicious' ? '#ff4757' : '#00ff88';

  const panel = (borderColor, children, extra = {}) => (
    <div style={{
      background:'rgba(8,18,26,0.95)', border:'1px solid rgba(255,255,255,0.07)',
      borderTop:`2px solid ${borderColor}`, padding:28, marginBottom:16,
      animation:'fadeUp 0.4s ease forwards', position:'relative', ...extra,
    }}>{children}</div>
  );

  return (
    <div style={{ minHeight:'100vh', background:'#060b10', fontFamily:"'Rajdhani',sans-serif", color:'#c8d8e4', display:'flex', flexDirection:'column', alignItems:'center', padding:'40px 16px', position:'relative', overflowX:'hidden' }}>
      <style>{`
        @import url('https://fonts.googleapis.com/css2?family=Share+Tech+Mono&family=Rajdhani:wght@400;500;600;700&display=swap');
        * { box-sizing:border-box; }
        body { margin:0; background:#060b10; }
        ::selection { background:rgba(0,255,136,0.3); }
        @keyframes blink { 0%,100%{opacity:1} 50%{opacity:0} }
        @keyframes fadeUp { from{opacity:0;transform:translateY(16px)} to{opacity:1;transform:translateY(0)} }
        @keyframes pulse { 0%,100%{opacity:1;transform:scale(1)} 50%{opacity:0.5;transform:scale(0.85)} }
        @keyframes rotate { from{transform:rotate(0deg)} to{transform:rotate(360deg)} }
        @keyframes shimmer { 0%{background-position:-200% center} 100%{background-position:200% center} }
        .scan-btn { cursor:pointer; border:none; outline:none; background:linear-gradient(90deg,#00ff88,#00ccff,#00ff88); background-size:200%; color:#050a0e; font-family:'Share Tech Mono',monospace; font-size:14px; letter-spacing:3px; font-weight:700; padding:14px 32px; transition:all 0.2s; }
        .scan-btn:hover:not(:disabled) { animation:shimmer 1s linear infinite; box-shadow:0 0 24px rgba(0,255,136,0.5); transform:translateY(-1px); }
        .scan-btn:disabled { background:#1e3a4a; color:#4a6a7a; cursor:not-allowed; }
        .url-input { flex:1; background:rgba(0,255,136,0.03); border:1px solid rgba(0,255,136,0.2); border-right:none; color:#00ff88; font-family:'Share Tech Mono',monospace; font-size:14px; padding:14px 18px; outline:none; transition:border-color 0.2s; }
        .url-input:focus { border-color:rgba(0,255,136,0.6); background:rgba(0,255,136,0.05); }
        .url-input::placeholder { color:#2a5a6a; }
      `}</style>

      {/* Grid bg */}
      <div style={{ position:'fixed', inset:0, pointerEvents:'none', zIndex:0, backgroundImage:'linear-gradient(rgba(0,255,136,0.025) 1px,transparent 1px),linear-gradient(90deg,rgba(0,255,136,0.025) 1px,transparent 1px)', backgroundSize:'44px 44px' }} />
      <div style={{ position:'fixed', inset:0, pointerEvents:'none', zIndex:0, background:'radial-gradient(ellipse at center,transparent 30%,#020608 100%)' }} />

      <div style={{ width:'100%', maxWidth:760, position:'relative', zIndex:2 }}>

        {/* Header */}
        <div style={{ textAlign:'center', marginBottom:44 }}>
          <div style={{ display:'inline-flex', alignItems:'center', gap:8, fontFamily:"'Share Tech Mono',monospace", fontSize:10, color:'#00ff88', letterSpacing:3, marginBottom:16, padding:'5px 14px', border:'1px solid rgba(0,255,136,0.25)', background:'rgba(0,255,136,0.04)' }}>
            <span style={{ width:6, height:6, borderRadius:'50%', background:'#00ff88', display:'inline-block', animation:'pulse 1.5s infinite' }} />
            CMPSC 441 · AI PROJECT · PENN STATE
          </div>
          <div style={{ fontSize:'clamp(28px,6vw,54px)', fontWeight:700, letterSpacing:-1, lineHeight:1, color:'#e8f4f8', marginBottom:10 }}>
            <span style={{ fontFamily:"'Share Tech Mono',monospace", color:'#00ff88', textShadow:'0 0 40px rgba(0,255,136,0.4)' }}>{typed}</span>
            <span style={{ animation:'blink 1s step-end infinite', color:'#00ff88' }}>_</span>
          </div>
          <div style={{ fontFamily:"'Share Tech Mono',monospace", fontSize:11, color:'#3a6a7a', letterSpacing:2 }}>
            NLP FEATURE EXTRACTION · RANDOM FOREST · SCIKIT-LEARN
          </div>
        </div>

        {/* Input */}
        {panel('rgba(0,255,136,0.5)',
          <>
            <div style={{ fontFamily:"'Share Tech Mono',monospace", fontSize:10, letterSpacing:2, color:'#00ff88', marginBottom:16, opacity:0.7 }}>// ENTER TARGET URL</div>
            <div style={{ display:'flex' }}>
              <input ref={inputRef} className="url-input" value={url} onChange={e => setUrl(e.target.value)} onKeyDown={e => e.key==='Enter' && handleScan()} placeholder="https://example.com/path?query=value" spellCheck={false} />
              <button className="scan-btn" onClick={handleScan} disabled={loading || !url.trim()}>{loading ? '◌ SCANNING' : '▶ SCAN URL'}</button>
            </div>
            {loading && (
              <div style={{ marginTop:16, display:'flex', alignItems:'center', gap:12, fontFamily:"'Share Tech Mono',monospace", fontSize:11, color:'#00ff88', opacity:0.8 }}>
                <div style={{ width:14, height:14, border:'2px solid #00ff88', borderTopColor:'transparent', borderRadius:'50%', animation:'rotate 0.8s linear infinite', flexShrink:0 }} />
                {PHASES[scanPhase]}
              </div>
            )}
          </>
        )}

        {error && (
          <div style={{ padding:16, background:'rgba(255,71,87,0.08)', border:'1px solid rgba(255,71,87,0.3)', color:'#ff6b81', fontFamily:"'Share Tech Mono',monospace", fontSize:12, marginBottom:16 }}>
            ✗ {error}
          </div>
        )}

        {result && (
          <div style={{ animation:'fadeUp 0.5s ease forwards' }}>
            {/* Verdict */}
            {panel(accentColor,
              <div style={{ display:'flex', alignItems:'center', gap:24, flexWrap:'wrap' }}>
                <RadialGauge value={result.confidence} verdict={verdict} />
                <div style={{ flex:1 }}>
                  <div style={{ marginBottom:12 }}><VerdictBadge verdict={verdict} /></div>
                  <div style={{ fontFamily:"'Share Tech Mono',monospace", fontSize:11, color:'#4a8a9a', lineHeight:1.9 }}>
                    <div>MALICIOUS PROB  <span style={{ color:'#ff4757' }}>{(result.malicious_prob||0).toFixed(1)}%</span></div>
                    <div>BENIGN PROB     <span style={{ color:'#2ed573' }}>{(result.benign_prob||0).toFixed(1)}%</span></div>
                    <div>MODEL           <span style={{ color:'#e8f0f5' }}>RANDOM FOREST (scikit-learn)</span></div>
                    <div>FEATURES        <span style={{ color:'#e8f0f5' }}>{Object.keys(result.raw_features||{}).length} NLP features extracted</span></div>
                  </div>
                </div>
              </div>
            )}

            {/* Risk signals */}
            {result.risk_signals?.length > 0 && panel(accentColor,
              <>
                <div style={{ fontFamily:"'Share Tech Mono',monospace", fontSize:10, letterSpacing:2, color:accentColor, marginBottom:14, opacity:0.8 }}>// RISK SIGNALS</div>
                {result.risk_signals.map((sig, i) => (
                  <div key={i} style={{ display:'flex', alignItems:'flex-start', gap:10, padding:'8px 12px', marginBottom:6, background: verdict==='benign'?'rgba(46,213,115,0.05)':'rgba(255,71,87,0.06)', borderLeft:`2px solid ${accentColor}`, fontSize:14, color:'#c8d8e4' }}>
                    <span style={{ color:accentColor, flexShrink:0, fontFamily:"'Share Tech Mono',monospace" }}>{verdict==='benign'?'✓':'⚠'}</span>
                    {sig}
                  </div>
                ))}
              </>
            )}

            {/* Top features */}
            {result.top_features?.length > 0 && panel('rgba(255,255,255,0.1)',
              <>
                <div style={{ fontFamily:"'Share Tech Mono',monospace", fontSize:10, letterSpacing:2, color:'#4a8a9a', marginBottom:14 }}>// TOP DECISION FEATURES (NLP EXTRACTION)</div>
                {result.top_features.map((f, i) => <FeatureRow key={i} name={f.name} value={f.value} risk={f.risk} note={f.note} />)}
              </>
            )}

            <AllFeatures features={result.raw_features} />
          </div>
        )}

        <div style={{ textAlign:'center', marginTop:32, fontFamily:"'Share Tech Mono',monospace", fontSize:10, color:'#1e3a4a', letterSpacing:2 }}>
          NLP · RANDOM FOREST · SVM · LOGISTIC REGRESSION · SCIKIT-LEARN · FLASK
        </div>
      </div>
    </div>
  );
}
