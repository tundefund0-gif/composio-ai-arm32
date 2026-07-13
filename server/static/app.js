/* ═══════════════════════════════════════════════════════════════
   ZEN AGENT — Dashboard Client
   ═══════════════════════════════════════════════════════════════ */
const S={uid:'web-user',sid:null,mc:0,ws:null,busy:false,recon:0};
const $=s=>document.querySelector(s);
const $$=s=>document.querySelectorAll(s);

// DOM refs
const msgs=$('#msgs'),inp=$('#chat-inp'),send=$('#btn-send');
const sidEl=$('#sid'),uidEl=$('#uid'),mcEl=$('#msgc');
const sdot=$('#sdot'),stxt=$('#stxt');
const btnClear=$('#btn-clear'),btnNew=$('#btn-new'),menuToggle=$('#menu-toggle');
const welcome=$('#welcome');

/* ─── WebSocket ─────────────────────────────────────────────── */
function connect(){
  const p=location.protocol==='https:'?'wss:':'ws:';
  if(S.ws)S.ws.close();
  S.ws=new WebSocket(`${p}//${location.host}/ws/chat/${S.uid}`);
  S.ws.onopen=()=>{setStatus('on');S.recon=0};
  S.ws.onmessage=e=>{try{handle(JSON.parse(e.data))}catch(_){}};
  S.ws.onclose=()=>{
    setStatus('off');S.busy=false;rmTyping();
    if(S.recon<5){S.recon++;setTimeout(connect,1500*S.recon)}
  };
  S.ws.onerror=()=>setStatus('off');
}

function handle(d){
  switch(d.type){
    case'info':S.sid=d.session_id||'';sidEl.textContent=d.session_id||'—';uidEl.textContent=d.uid||S.uid;break;
    case'token':appendToken(d.content);break;
    case'reasoning':appendReasoning(d.content);break;
    case'done':finishStream(d.content);break;
    case'clear':clearUI();break;
    case'error':toast(d.message||'Error','err');finishStream(null);break;
  }
}

function wsSend(m){
  if(S.ws&&S.ws.readyState===WebSocket.OPEN){S.ws.send(JSON.stringify({message:m}));return true}
  return false
}

/* ─── Chat UI ──────────────────────────────────────────────── */
function addMsg(text,role){
  rmWelcome();
  const div=document.createElement('div');
  div.className=`msg ${role==='user'?'user':'assistant'}`;
  const ts=new Date().toLocaleTimeString([],{hour:'2-digit',minute:'2-digit'});
  div.innerHTML=`
    <div class="av">${role==='user'?'👤':'🤖'}</div>
    <div>
      <div class="bubble"></div>
      <div class="msg-time">${ts}</div>
    </div>`;
  const bubble=div.querySelector('.bubble');
  if(role==='user')bubble.textContent=text;
  else bubble.id='ast-msg';
  msgs.appendChild(div);
  scroll();scb();
  return bubble;
}

function addTyping(){
  rmTyping();
  const d=document.createElement('div');d.className='msg assistant';d.id='typing-msg';
  d.innerHTML=`
    <div class="av">🤖</div>
    <div><div class="bubble"><div class="typing-dots"><span></span><span></span><span></span></div></div></div>`;
  msgs.appendChild(d);scroll();
}

function appendToken(t){
  let el=document.querySelector('#ast-msg');
  if(!el){el=addMsg('','assistant');el.id='ast-msg'}
  rmTyping();
  el.textContent+=t;
  scroll();
  scb();
}

let _reasoningEl=null,_rBuf='';
function appendReasoning(t){
  _rBuf+=t;
  let el=document.querySelector('#ast-msg');
  if(!el){el=addMsg('','assistant');el.id='ast-msg'}
  rmTyping();
  if(!_reasoningEl||!document.contains(_reasoningEl)){
    const btn=document.createElement('div');btn.className='reasoning-btn';
    btn.textContent='🤔 Thinking...';btn.onclick=toggleReasoning;
    const box=document.createElement('div');box.className='reasoning-box';box.id='rbox';
    el.parentNode.appendChild(btn);btn.after(box);
    _reasoningEl=box;
  }
  _reasoningEl.textContent=_rBuf;
  scroll();
}

function toggleReasoning(){
  const box=document.querySelector('#rbox');
  if(!box)return;
  box.classList.toggle('open');
  const btn=box.previousElementSibling;
  if(btn&&btn.classList.contains('reasoning-btn'))
    btn.textContent=box.classList.contains('open')?'🤔 Hide thinking':'🤔 Show thinking…';
}

function finishStream(full){
  rmTyping();
  if(full!==null){
    const el=document.querySelector('#ast-msg');
    if(el&&el.textContent&&!full){} // keep it
  }
  S.busy=false;updBtn();updMC();
  $$('#ast-msg').forEach(e=>e.removeAttribute('id'));
  _reasoningEl=null;_rBuf='';
  scb();
}

function rmWelcome(){if(welcome&&welcome.parentNode)welcome.remove()}
function rmTyping(){const t=document.getElementById('typing-msg');if(t)t.remove()}
function clearUI(){msgs.innerHTML='';if(welcome)msgs.appendChild(welcome);updMC();S.mc=0}
function scroll(){requestAnimationFrame(()=>{msgs.scrollTop=msgs.scrollHeight})}

// Scroll anchor — auto scroll unless user scrolled up
let _userScrolled=false;
msgs.addEventListener('scroll',()=>{
  const d=msgs.scrollHeight-msgs.scrollTop-msgs.clientHeight;
  _userScrolled=d>60;
});
function scb(){if(!_userScrolled)scroll()}

/* ─── Markdown-like rendering ─────────────────────────────── */
function renderBubble(text,bubble){
  if(!text)return;
  // Simple markdown rendering
  let h=esc(text)
    .replace(/```(\w*)\n([\s\S]*?)```/g,'<pre><code class="lang-$1">$2</code></pre>')
    .replace(/`([^`]+)`/g,'<code>$1</code>')
    .replace(/\*\*\*(.+?)\*\*\*/g,'<strong><em>$1</em></strong>')
    .replace(/\*\*(.+?)\*\*/g,'<strong>$1</strong>')
    .replace(/\*(.+?)\*/g,'<em>$1</em>')
    .replace(/^### (.+)$/gm,'<h3>$1</h3>')
    .replace(/^## (.+)$/gm,'<h2>$1</h2>')
    .replace(/^# (.+)$/gm,'<h1>$1</h1>')
    .replace(/^- (.+)$/gm,'<li>$1</li>')
    .replace(/(<li>.*<\/li>\n?)+/g,'<ul>$&</ul>')
    .replace(/^\d+\. (.+)$/gm,'<li>$1</li>')
    .replace(/(<li>.*<\/li>\n?)+/g,'<ol>$&</ol>')
    .replace(/\n\n/g,'</p><p>')
    .replace(/\n/g,'<br>');
  bubble.innerHTML='<p>'+h+'</p>';
  
  // Add copy buttons to code blocks
  bubble.querySelectorAll('pre code').forEach(block=>{
    const pre=block.parentNode;
    if(pre.querySelector('.copy-btn'))return;
    const btn=document.createElement('button');
    btn.className='copy-btn';
    Object.assign(btn.style,{
      position:'absolute',top:'4px',right:'4px',
      background:'var(--border)',border:'none',color:'var(--text3)',
      padding:'2px 8px',borderRadius:'4px',fontSize:'10px',cursor:'pointer',
      fontFamily:'inherit',
    });
    btn.textContent='📋';
    btn.title='Copy code';
    btn.onclick=()=>{
      navigator.clipboard.writeText(block.textContent).then(()=>{
        btn.textContent='✅';setTimeout(()=>btn.textContent='📋',1500);
      });
    };
    pre.style.position='relative';
    pre.appendChild(btn);
  });
}

function esc(t){const d=document.createElement('div');d.textContent=t;return d.innerHTML}

/* ─── Send ────────────────────────────────────────────────── */
function sendMsg(){
  const t=inp.value.trim();
  if(!t||S.busy)return;
  addMsg(t,'user');inp.value='';inp.style.height='auto';
  addTyping();
  if(!wsSend(t))restChat(t);
}

async function restChat(m){
  S.busy=true;updBtn();
  try{
    const r=await fetch('/api/chat',{
      method:'POST',headers:{'Content-Type':'application/json'},
      body:JSON.stringify({message:m,user_id:S.uid,session_id:S.sid})
    });
    const d=await r.json();
    S.sid=d.session_id||S.sid;sidEl.textContent=S.sid||'—';
    rmTyping();
    const el=addMsg('','assistant');
    el.id='ast-msg';
    renderBubble(d.response||'[No response]',el);
    el.removeAttribute('id');
    finishStream(d.response);
  }catch(e){toast('Request failed','err');rmTyping();finishStream(null)}
}

/* ─── Events ──────────────────────────────────────────────── */
inp.addEventListener('input',()=>{
  inp.style.height='auto';inp.style.height=Math.min(inp.scrollHeight,80)+'px';updBtn()
});
inp.addEventListener('keydown',e=>{
  if(e.key==='Enter'&&!e.shiftKey){e.preventDefault();sendMsg()}
});
send.addEventListener('click',sendMsg);

btnClear.addEventListener('click',()=>{
  if(wsSend('/clear')){}else clearUI()
});
btnNew.addEventListener('click',()=>{
  S.sid=null;clearUI();
  if(S.ws)S.ws.close();setTimeout(connect,500);
  toast('New session started','ok')
});
menuToggle.addEventListener('click',()=>{
  document.getElementById('sidebar').classList.toggle('open')
});

// Chip suggestions
msgs.addEventListener('click',e=>{
  const c=e.target.closest('.chip');
  if(c){inp.value=c.dataset.p;inp.style.height='auto';inp.focus();updBtn()}
});

/* ─── Helpers ─────────────────────────────────────────────── */
function updBtn(){send.disabled=!inp.value.trim()||S.busy}
function updMC(){
  const c=$$('.msg').length;S.mc=c;
  mcEl.textContent=`${c} message${c!==1?'s':''}`;
}
function setStatus(s){
  sdot.className='dot '+s;
  stxt.textContent={on:'Connected',off:'Disconnected',busy:'Busy'}[s]||s;
}
function toast(m,t='info'){
  const c=document.getElementById('toasts');
  const d=document.createElement('div');d.className='toast '+t;d.textContent=m;
  c.appendChild(d);setTimeout(()=>d.remove(),3500)
}

/* ─── Init ────────────────────────────────────────────────── */
document.addEventListener('DOMContentLoaded',()=>{connect();updBtn();updMC()});
