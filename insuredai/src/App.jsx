import { useState, useEffect, useRef, useCallback } from 'react'
import {
  LayoutDashboard, MessageSquare, Search, BarChart3,
  ArrowUp, AlertTriangle, CheckCircle, Clock, FileText,
  Zap, TrendingUp, Shield, Activity, Users, MapPin,
  AlertCircle, Sparkles, BookOpen, Database
} from 'lucide-react'
import {
  BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer,
  PieChart, Pie, Cell, AreaChart, Area, CartesianGrid,
  RadarChart, Radar, PolarGrid, PolarAngleAxis
} from 'recharts'

const API = '/api'
const post = (url, body) => fetch(url, {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify(body)
}).then(r => r.json())

const CHART_COLORS = ['#2563eb','#059669','#7c3aed','#d97706','#dc2626']
const TT = {
  contentStyle: {
    background: '#fff', border: '1px solid #e4e7ec', borderRadius: 10,
    fontSize: 12, fontFamily: 'Inter,sans-serif',
    boxShadow: '0 4px 12px rgba(0,0,0,0.1)', color: '#0f1117'
  },
  cursor: { fill: 'rgba(37,99,235,0.04)' }
}

function LogoMark({ size = 18 }) {
  return (
    <svg width={size} height={size} viewBox="0 0 24 24" fill="none">
      <path d="M12 2L4 6v6c0 5.25 3.5 10.15 8 11.35C16.5 22.15 20 17.25 20 12V6L12 2z"
        fill="currentColor" opacity="0.95"/>
      <path d="M9 12l2 2 4-4" stroke="white" strokeWidth="2"
        strokeLinecap="round" strokeLinejoin="round"/>
    </svg>
  )
}

function Sidebar({ page, setPage, online }) {
  const items = [
    { id: 'dashboard', label: 'Dashboard',      icon: LayoutDashboard, badge: null },
    { id: 'chat',      label: 'Ask InsuredAI',  icon: MessageSquare,   badge: 'AI' },
    { id: 'claim',     label: 'Claim Analyser', icon: Search,          badge: null },
    { id: 'analytics', label: 'Analytics',      icon: BarChart3,       badge: null },
  ]
  return (
    <aside className="sidebar">
      <div className="sidebar-top">
        <div className="logo">
          <div className="logo-mark"><LogoMark size={18} /></div>
          <div>
            <div className="logo-name">InsuredAI</div>
            <div className="logo-tag">Claims Intelligence</div>
          </div>
        </div>
        <div className="nav-label">Menu</div>
        <nav className="nav">
          {items.map(({ id, label, icon: Icon, badge }) => (
            <div key={id}
              className={`nav-btn ${page === id ? 'active' : ''}`}
              onClick={() => setPage(id)}>
              <Icon size={16} />
              {label}
              {badge && <span className="nav-badge">{badge}</span>}
            </div>
          ))}
        </nav>
      </div>
      <div className="sidebar-sep" />
      <div className="sidebar-bottom">
        <div className="api-status">
          <div className={`status-dot ${online ? 'on' : 'off'}`} />
          <div className="status-info">
            <div className="s1">{online ? 'System Online' : 'System Offline'}</div>
            <div className="s2">LangGraph · Groq · XGBoost</div>
          </div>
        </div>
      </div>
    </aside>
  )
}

function Dashboard() {
  const [stats,   setStats]   = useState(null)
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    fetch(`${API}/claims/stats`).then(r => r.json())
      .then(s => { setStats(s); setLoading(false) })
      .catch(() => setLoading(false))
  }, [])

  const typeData = [
    { n:'Vehicle', v:2000 }, { n:'Property', v:2071 },
    { n:'Liability', v:1977 }, { n:'Health', v:1956 }, { n:'Travel', v:1996 },
  ]
  const statusData = [
    { n:'Closed', v:55, c:'#059669' }, { n:'Open', v:25, c:'#2563eb' },
    { n:'In Review', v:15, c:'#d97706' }, { n:'Litigated', v:5, c:'#dc2626' },
  ]
  const regionData = [
    {r:'London',s:23.7},{r:'Birmingham',s:21.7},{r:'Manchester',s:21.1},
    {r:'Liverpool',s:18.8},{r:'Sheffield',s:18.7},{r:'Bristol',s:18.5},
    {r:'Edinburgh',s:18.4},{r:'Cardiff',s:18.4},{r:'Belfast',s:18.4},{r:'Leeds',s:18.3},
  ]
  const trendData = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']
    .map((m, i) => ({ m, v: Math.round(650 + Math.sin(i / 2) * 150 + i * 12) }))

  const kpis = stats ? [
    { label:'Total Claims',    val: stats.total_claims.toLocaleString(),       sub: 'All time',           cls: 'kpi-blue'   },
    { label:'Open Claims',     val: stats.open_claims.toLocaleString(),         sub: 'Needs action',       cls: 'kpi-purple' },
    { label:'Fraud Flagged',   val: stats.fraud_flagged.toLocaleString(),       sub: `${(stats.fraud_flagged/stats.total_claims*100).toFixed(1)}% of total`, cls: 'kpi-red' },
    { label:'Litigation Risk', val: stats.litigation_flagged.toLocaleString(),  sub: `${(stats.litigation_flagged/stats.total_claims*100).toFixed(1)}% of total`, cls: 'kpi-amber' },
    { label:'Avg Claim Value', val: `£${Math.round(stats.avg_claim_amount/1000)}k`, sub: `~${Math.round(stats.avg_resolution_days)} days avg`, cls: 'kpi-green' },
  ] : []

  return (
    <div className="page">
      <div className="page-head">
        <div className="page-title">Operations overview</div>
        <div className="page-sub">Live claims intelligence · {new Date().toLocaleDateString('en-GB', { weekday:'long', day:'numeric', month:'long' })}</div>
      </div>

      {loading ? (
        <div className="kpi-grid">{[...Array(5)].map((_,i) => <div key={i} className="skel" style={{height:88}}/>)}</div>
      ) : (
        <div className="kpi-grid">
          {kpis.map(({ label, val, sub, cls }) => (
            <div key={label} className={`kpi ${cls}`}>
              <div className="kpi-stripe" />
              <div className="kpi-label">{label}</div>
              <div className="kpi-val">{val}</div>
              <div className="kpi-sub">{sub}</div>
            </div>
          ))}
        </div>
      )}

      <div className="grid-2">
        <div className="card">
          <div className="card-head"><BarChart3 size={15} className="card-icon"/><span className="card-title">Claims by type</span></div>
          <div className="card-body">
            <ResponsiveContainer width="100%" height={200}>
              <BarChart data={typeData} barSize={28}>
                <CartesianGrid strokeDasharray="3 3" stroke="#f0f2f5" vertical={false}/>
                <XAxis dataKey="n" tick={{fill:'#9ca3af',fontSize:11}} axisLine={false} tickLine={false}/>
                <YAxis tick={{fill:'#9ca3af',fontSize:10}} axisLine={false} tickLine={false}/>
                <Tooltip {...TT}/>
                <Bar dataKey="v" radius={[5,5,0,0]} name="Claims">
                  {typeData.map((_,i) => <Cell key={i} fill={CHART_COLORS[i]}/>)}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>

        <div className="card">
          <div className="card-head"><Activity size={15} className="card-icon"/><span className="card-title">Status breakdown</span></div>
          <div className="card-body" style={{display:'flex',flexDirection:'column',alignItems:'center'}}>
            <ResponsiveContainer width="100%" height={160}>
              <PieChart>
                <Pie data={statusData} cx="50%" cy="50%" innerRadius={48} outerRadius={72} paddingAngle={3} dataKey="v">
                  {statusData.map((d,i) => <Cell key={i} fill={d.c}/>)}
                </Pie>
                <Tooltip contentStyle={TT.contentStyle}/>
              </PieChart>
            </ResponsiveContainer>
            <div style={{display:'flex',flexWrap:'wrap',gap:'8px 16px',justifyContent:'center',marginTop:4}}>
              {statusData.map(d => (
                <div key={d.n} style={{display:'flex',alignItems:'center',gap:5}}>
                  <div style={{width:8,height:8,borderRadius:'50%',background:d.c}}/>
                  <span style={{fontSize:'0.75rem',color:'#4b5563'}}>{d.n} · {d.v}%</span>
                </div>
              ))}
            </div>
          </div>
        </div>
      </div>

      <div className="card" style={{marginBottom:14}}>
        <div className="card-head"><TrendingUp size={15} className="card-icon"/><span className="card-title">Claims volume trend</span></div>
        <div className="card-body">
          <ResponsiveContainer width="100%" height={160}>
            <AreaChart data={trendData}>
              <defs>
                <linearGradient id="ag" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="0%" stopColor="#2563eb" stopOpacity={0.12}/>
                  <stop offset="100%" stopColor="#2563eb" stopOpacity={0}/>
                </linearGradient>
              </defs>
              <CartesianGrid strokeDasharray="3 3" stroke="#f0f2f5" vertical={false}/>
              <XAxis dataKey="m" tick={{fill:'#9ca3af',fontSize:11}} axisLine={false} tickLine={false}/>
              <YAxis tick={{fill:'#9ca3af',fontSize:10}} axisLine={false} tickLine={false}/>
              <Tooltip {...TT}/>
              <Area type="monotone" dataKey="v" stroke="#2563eb" strokeWidth={2} fill="url(#ag)" name="Claims"/>
            </AreaChart>
          </ResponsiveContainer>
        </div>
      </div>

      <div className="card">
        <div className="card-head"><MapPin size={15} className="card-icon"/><span className="card-title">Average fraud score by region (%)</span></div>
        <div className="card-body">
          <ResponsiveContainer width="100%" height={200}>
            <BarChart data={regionData} layout="vertical" barSize={12}>
              <CartesianGrid strokeDasharray="3 3" stroke="#f0f2f5" horizontal={false}/>
              <XAxis type="number" domain={[15,25]} tick={{fill:'#9ca3af',fontSize:10}} axisLine={false} tickLine={false} tickFormatter={v=>`${v}%`}/>
              <YAxis type="category" dataKey="r" tick={{fill:'#4b5563',fontSize:11}} axisLine={false} tickLine={false} width={85}/>
              <Tooltip {...TT} formatter={v=>[`${v}%`,'Fraud score']}/>
              <Bar dataKey="s" radius={[0,5,5,0]}>
                {regionData.map((_,i) => <Cell key={i} fill={`hsl(${220-i*8},${85-i*3}%,${55-i*2}%)`}/>)}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        </div>
      </div>
    </div>
  )
}

const EXAMPLES = [
  'Which claim type has the longest average resolution time?',
  'What is subrogation in insurance?',
  'Show top 5 adjusters by fraud-flagged claims',
  'Score claim CLM0000100',
  'What does excess mean on a policy?',
  'Compare average claim amounts across all regions',
  'What happens if a policyholder misses the claim deadline?',
  'Which month had the highest claims in 2023?',
]

const PILLS = {
  sql:          { l: 'SQL Agent', c: 'pill-sql'  },
  rag:          { l: 'RAG Agent', c: 'pill-rag'  },
  ml:           { l: 'ML Agent',  c: 'pill-ml'   },
  both_sql_rag: { l: 'SQL + RAG', c: 'pill-both' },
}

function SourceBadge({ sourceType }) {
  if (sourceType === 'ai_knowledge') {
    return (
      <span style={{
        fontSize:'0.68rem', padding:'2px 8px', borderRadius:'99px',
        background:'#fffbeb', color:'#92400e',
        border:'1px solid #fcd34d', fontWeight:600,
        display:'inline-flex', alignItems:'center', gap:3
      }}>
        <span style={{fontSize:10}}>✦</span> AI Knowledge
      </span>
    )
  }
  if (sourceType === 'policy_documents') {
    return (
      <span style={{
        fontSize:'0.68rem', padding:'2px 8px', borderRadius:'99px',
        background:'#ecfdf5', color:'#065f46',
        border:'1px solid #6ee7b7', fontWeight:600,
        display:'inline-flex', alignItems:'center', gap:3
      }}>
        <span style={{fontSize:10}}>📄</span> Policy Docs
      </span>
    )
  }
  return null
}

function Chat() {
  const [msgs,    setMsgs]    = useState([])
  const [input,   setInput]   = useState('')
  const [loading, setLoading] = useState(false)
  const endRef = useRef(null)

  useEffect(() => { endRef.current?.scrollIntoView({ behavior: 'smooth' }) }, [msgs])

  const send = useCallback(async (q) => {
    if (!q.trim() || loading) return
    setMsgs(m => [...m, { role: 'user', content: q }])
    setInput('')
    setLoading(true)
    try {
      const r = await post(`${API}/query`, { question: q })
      setMsgs(m => [...m, {
        role:        'ai',
        content:     r.answer,
        route:       r.route,
        sources:     r.sources || [],
        source_type: r.source_type || null,
      }])
    } catch(e) {
      setMsgs(m => [...m, { role:'ai', content:`Error: ${e.message}`, route:'sql', sources:[], source_type:null }])
    }
    setLoading(false)
  }, [loading])

  const onKey = e => { if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); send(input) } }

  return (
    <div className="chat-wrap">
      <div className="chat-header">
        <div className="page-title">Ask InsuredAI</div>
        <div className="page-sub">Claims data · Policy documents · Insurance knowledge · ML risk scoring</div>
      </div>

      {msgs.length === 0 && !loading ? (
        <div className="chat-empty">
          <div className="chat-empty-icon"><Sparkles size={24}/></div>
          <div className="chat-empty-title">What would you like to know?</div>
          <div className="chat-empty-sub">
            Ask about claims data, policy terms, insurance definitions, or get a risk score for any claim.
            InsuredAI searches policy documents first — and falls back to its insurance knowledge when needed.
          </div>
          <div className="chips">
            {EXAMPLES.map((ex, i) => (
              <button key={i} className="chip" onClick={() => send(ex)}>{ex}</button>
            ))}
          </div>
        </div>
      ) : (
        <div className="chat-messages">
          {msgs.map((msg, i) => (
            <div key={i} className={`msg ${msg.role}`}>
              <div className={`msg-av ${msg.role}`}>
                {msg.role === 'ai' ? <LogoMark size={14}/> : 'U'}
              </div>
              <div className={`bubble ${msg.role}`}>
                {msg.role === 'ai' && msg.route && (
                  <div className="bubble-top">
                    <span className={`agent-pill ${PILLS[msg.route]?.c || 'pill-sql'}`}>
                      {PILLS[msg.route]?.l || msg.route}
                    </span>
                    <SourceBadge sourceType={msg.source_type} />
                    <span style={{fontSize:'0.72rem',color:'#9ca3af',marginLeft:'auto'}}>InsuredAI</span>
                  </div>
                )}
                <div style={{whiteSpace:'pre-wrap'}}>{msg.content}</div>
                {msg.sources?.length > 0 && (
                  <div className="src-list">
                    {msg.sources.map((s,si) => (
                      <div key={si} className="src-item"><FileText size={11}/>{s}</div>
                    ))}
                  </div>
                )}
              </div>
            </div>
          ))}
          {loading && (
            <div className="msg ai">
              <div className="msg-av ai"><LogoMark size={14}/></div>
              <div className="bubble ai">
                <div className="thinking"><b/><b/><b/></div>
              </div>
            </div>
          )}
          <div ref={endRef}/>
        </div>
      )}

      <div className="chat-bottom">
        {msgs.length > 0 && (
          <div className="quick-bar">
            {EXAMPLES.slice(0,4).map((ex,i) => (
              <button key={i} className="qchip" onClick={() => send(ex)}>{ex}</button>
            ))}
          </div>
        )}
        <div className="input-box">
          <textarea className="input-ta"
            placeholder="Ask about claims, policy terms, insurance definitions, or type a claim ID..."
            value={input}
            onChange={e => setInput(e.target.value)}
            onKeyDown={onKey}
            rows={1}
          />
          <button className="send-btn" onClick={() => send(input)}
            disabled={loading || !input.trim()} aria-label="Send">
            <ArrowUp size={16}/>
          </button>
        </div>
        <div className="input-hint">Press Enter to send · Shift+Enter for new line</div>
      </div>
    </div>
  )
}

function ClaimAnalyser() {
  const [id,      setId]      = useState('')
  const [result,  setResult]  = useState(null)
  const [loading, setLoading] = useState(false)
  const [error,   setError]   = useState(null)

  const analyse = async () => {
    if (!id.trim()) return
    setLoading(true); setError(null); setResult(null)
    try {
      const r = await fetch(`${API}/claim/${id.trim().toUpperCase()}`)
      if (!r.ok) { const e = await r.json(); throw new Error(e.detail || 'Not found') }
      setResult(await r.json())
    } catch(e) { setError(e.message) }
    setLoading(false)
  }

  const radarData = result ? [
    { s:'Fraud',       v: +(result.fraud_score * 100).toFixed(1) },
    { s:'Litigation',  v: +(result.litigation_score * 100).toFixed(1) },
    { s:'Complexity',  v: +Math.min(result.resolution_days_forecast / 3.65, 100).toFixed(1) },
    { s:'Severity',    v: +Math.min(result.claim_amount / 1200, 100).toFixed(1) },
    { s:'Risk Tier',   v: result.risk_tier==='high' ? 80 : result.risk_tier==='medium' ? 50 : 25 },
  ] : []

  return (
    <div style={{display:'flex',flexDirection:'column',height:'100%',overflow:'hidden'}}>
      <div className="analyser-top">
        <div className="page-head">
          <div className="page-title">Claim analyser</div>
          <div className="page-sub">AI-powered fraud detection, litigation prediction, and policy context</div>
        </div>
        <div className="search-row">
          <input className="inp" placeholder="Enter claim ID — e.g. CLM0000042"
            value={id} onChange={e => setId(e.target.value)}
            onKeyDown={e => e.key==='Enter' && analyse()}/>
          <button className="btn" onClick={analyse} disabled={loading || !id.trim()}>
            {loading ? 'Analysing…' : 'Analyse →'}
          </button>
        </div>
        <div className="hint">Try: CLM0000001 · CLM0000042 · CLM0000100 · CLM0001234 · CLM0005000</div>
      </div>

      <div className="analyser-body">
        {error && <div className="err"><AlertCircle size={16}/>{error}</div>}

        {loading && (
          <div style={{display:'flex',flexDirection:'column',gap:12}}>
            {[88,100,80].map((h,i) => <div key={i} className="skel" style={{height:h,borderRadius:12}}/>)}
          </div>
        )}

        {result && !loading && (
          <>
            <div className={`claim-card risk-${result.risk_tier}`}>
              <div className="claim-top">
                <div>
                  <div className="claim-type-label">Claim ID</div>
                  <div className="claim-id">{result.claim_id}</div>
                </div>
                <div className={`risk-badge rb-${result.risk_tier}`}>
                  {result.risk_tier==='high' ? <AlertTriangle size={13}/> :
                   result.risk_tier==='medium' ? <AlertCircle size={13}/> :
                   <CheckCircle size={13}/>}
                  {result.risk_tier.charAt(0).toUpperCase()+result.risk_tier.slice(1)} risk
                </div>
              </div>
              <div className="meta-row">
                {[
                  ['Type',   result.claim_type],
                  ['Amount', `£${result.claim_amount.toLocaleString('en-GB',{minimumFractionDigits:2,maximumFractionDigits:2})}`],
                  ['Status', result.status],
                  ['Region', result.region],
                ].map(([l,v]) => (
                  <div key={l}>
                    <div className="ml">{l}</div>
                    <div className="mv">{v}</div>
                  </div>
                ))}
              </div>
            </div>

            <div className="score-row">
              <div className="sc">
                <div className="sc-label">Fraud score</div>
                <div className="sc-num" style={{color:result.fraud_flag?'var(--red)':'var(--green)'}}>
                  {(result.fraud_score*100).toFixed(1)}%
                </div>
                <div className="sc-track">
                  <div className="sc-fill" style={{
                    width:`${result.fraud_score*100}%`,
                    background:result.fraud_flag?'var(--red)':'var(--green)'
                  }}/>
                </div>
                <div className={`sc-flag ${result.fraud_flag?'flag-bad':'flag-ok'}`}>
                  {result.fraud_flag ? <><AlertTriangle size={12}/> Flagged for review</> : <><CheckCircle size={12}/> No fraud indicators</>}
                </div>
              </div>

              <div className="sc">
                <div className="sc-label">Litigation score</div>
                <div className="sc-num" style={{color:result.litigation_flag?'var(--amber)':'var(--green)'}}>
                  {(result.litigation_score*100).toFixed(1)}%
                </div>
                <div className="sc-track">
                  <div className="sc-fill" style={{
                    width:`${result.litigation_score*100}%`,
                    background:result.litigation_flag?'var(--amber)':'var(--green)'
                  }}/>
                </div>
                <div className={`sc-flag ${result.litigation_flag?'flag-warn':'flag-ok'}`}>
                  {result.litigation_flag ? <><AlertCircle size={12}/> Litigation risk elevated</> : <><CheckCircle size={12}/> Low litigation risk</>}
                </div>
              </div>

              <div className="sc">
                <div className="sc-label">Resolution forecast</div>
                <div className="sc-num" style={{color:'var(--brand)'}}>
                  {result.resolution_days_forecast}
                  <span style={{fontSize:'0.95rem',fontWeight:500,color:'var(--text-3)'}}> days</span>
                </div>
                <div className="sc-track">
                  <div className="sc-fill" style={{
                    width:`${Math.min(result.resolution_days_forecast/5,100)}%`,
                    background:'var(--brand)'
                  }}/>
                </div>
                <div className={`sc-flag ${result.resolution_days_forecast>200?'flag-warn':'flag-ok'}`}>
                  <Clock size={12}/>
                  {result.resolution_days_forecast>200?'Complex — extended timeline':'Standard resolution expected'}
                </div>
              </div>
            </div>

            <div className="grid-2">
              <div className="card">
                <div className="card-head"><Activity size={15} className="card-icon"/><span className="card-title">Risk profile radar</span></div>
                <div className="card-body">
                  <ResponsiveContainer width="100%" height={210}>
                    <RadarChart data={radarData} cx="50%" cy="50%">
                      <PolarGrid stroke="#e4e7ec"/>
                      <PolarAngleAxis dataKey="s" tick={{fill:'#4b5563',fontSize:11}}/>
                      <Radar dataKey="v" stroke="#2563eb" fill="#2563eb" fillOpacity={0.1} strokeWidth={2} name="Score"/>
                      <Tooltip contentStyle={TT.contentStyle}/>
                    </RadarChart>
                  </ResponsiveContainer>
                </div>
              </div>

              <div className="policy-panel">
                <div className="policy-head"><FileText size={14}/> Relevant policy context</div>
                <div className="policy-body">{result.policy_context}</div>
              </div>
            </div>
          </>
        )}
      </div>
    </div>
  )
}

function Analytics() {
  const byType = [
    { t:'Liability', f:22, l:28 }, { t:'Travel', f:15, l:4 },
    { t:'Vehicle', f:12, l:8 }, { t:'Health', f:6, l:10 }, { t:'Property', f:8, l:6 },
  ]
  const resolution = [
    { s:'Litigated', v:289, c:'#dc2626' }, { s:'In Review', v:160, c:'#d97706' },
    { s:'Closed', v:181, c:'#059669' }, { s:'Open', v:100, c:'#2563eb' },
  ]
  const perf = [
    { l:'Fraud detection AUC-ROC',   v:'0.9143', c:'var(--green)',  i:Shield     },
    { l:'Litigation prediction AUC', v:'0.9399', c:'var(--brand)',  i:Activity   },
    { l:'Resolution forecast R²',    v:'0.9998', c:'var(--purple)', i:TrendingUp },
    { l:'SQL query accuracy',        v:'~92%',   c:'var(--amber)',  i:Zap        },
    { l:'Claims in database',        v:'10,000', c:'var(--text)',   i:Users      },
  ]

  return (
    <div className="page">
      <div className="page-head">
        <div className="page-title">Analytics & model performance</div>
        <div className="page-sub">Patterns, trends, and AI model metrics</div>
      </div>

      <div className="grid-2">
        <div className="card">
          <div className="card-head"><BarChart3 size={15} className="card-icon"/><span className="card-title">Fraud vs litigation rate by type</span></div>
          <div className="card-body">
            <ResponsiveContainer width="100%" height={210}>
              <BarChart data={byType} barSize={18}>
                <CartesianGrid strokeDasharray="3 3" stroke="#f0f2f5" vertical={false}/>
                <XAxis dataKey="t" tick={{fill:'#9ca3af',fontSize:11}} axisLine={false} tickLine={false}/>
                <YAxis tick={{fill:'#9ca3af',fontSize:10}} axisLine={false} tickLine={false} tickFormatter={v=>`${v}%`}/>
                <Tooltip {...TT} formatter={v=>[`${v}%`]}/>
                <Bar dataKey="f" fill="#dc2626" radius={[4,4,0,0]} name="Fraud rate"/>
                <Bar dataKey="l" fill="#2563eb" radius={[4,4,0,0]} name="Litigation rate"/>
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>

        <div className="card">
          <div className="card-head"><Clock size={15} className="card-icon"/><span className="card-title">Median resolution days by status</span></div>
          <div className="card-body">
            <ResponsiveContainer width="100%" height={210}>
              <BarChart data={resolution} barSize={36}>
                <CartesianGrid strokeDasharray="3 3" stroke="#f0f2f5" vertical={false}/>
                <XAxis dataKey="s" tick={{fill:'#9ca3af',fontSize:11}} axisLine={false} tickLine={false}/>
                <YAxis tick={{fill:'#9ca3af',fontSize:10}} axisLine={false} tickLine={false}/>
                <Tooltip {...TT} formatter={v=>[`${v} days`,'Median']}/>
                <Bar dataKey="v" radius={[5,5,0,0]} name="Days">
                  {resolution.map((d,i) => <Cell key={i} fill={d.c}/>)}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>
      </div>

      <div className="card">
        <div className="card-head"><Sparkles size={15} className="card-icon"/><span className="card-title">AI model performance</span></div>
        <div className="card-body">
          <div className="model-perf">
            {perf.map(({ l, v, c, i: Icon }) => (
              <div key={l} className="perf-row">
                <div className="perf-left"><Icon size={15} style={{color:c,flexShrink:0}}/>{l}</div>
                <div className="perf-val" style={{color:c}}>{v}</div>
              </div>
            ))}
          </div>
        </div>
      </div>
    </div>
  )
}

export default function App() {
  const [page,   setPage]   = useState('dashboard')
  const [online, setOnline] = useState(false)

  useEffect(() => {
    fetch(`${API}/health`, { signal: AbortSignal.timeout(3000) })
      .then(r => r.json())
      .then(d => setOnline(d.status === 'ok'))
      .catch(() => {})
  }, [])

  const pages = { dashboard:Dashboard, chat:Chat, claim:ClaimAnalyser, analytics:Analytics }
  const Page  = pages[page]

  return (
    <div className="app">
      <Sidebar page={page} setPage={setPage} online={online}/>
      <main className="main"><Page key={page}/></main>
    </div>
  )
}