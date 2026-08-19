function fixifyInitTheme(){
  const saved = localStorage.getItem('fixify_theme') || 'dark';
  document.documentElement.setAttribute('data-theme', saved);
  document.querySelectorAll('.theme-toggle-btn').forEach(btn=>{
    btn.textContent = saved === 'dark' ? '🌙 Dark' : '☀️ Light';
  });
}

function fixifyToggleTheme(){
  const current = document.documentElement.getAttribute('data-theme') || 'dark';
  const next = current === 'dark' ? 'light' : 'dark';
  document.documentElement.setAttribute('data-theme', next);
  localStorage.setItem('fixify_theme', next);
  document.querySelectorAll('.theme-toggle-btn').forEach(btn=>{
    btn.textContent = next === 'dark' ? '🌙 Dark' : '☀️ Light';
  });
  document.dispatchEvent(new CustomEvent('fixify-theme-changed', {detail:{theme: next}}));
}

function fixifyInitDrawer(){
  const drawer = document.getElementById('drawer');
  const overlay = document.getElementById('drawerOverlay');
  const toggleBtns = document.querySelectorAll('.hamburger');
  if(!drawer || !overlay) return;

  function open(){ drawer.classList.add('open'); overlay.classList.add('open'); }
  function close(){ drawer.classList.remove('open'); overlay.classList.remove('open'); }

  toggleBtns.forEach(btn=>btn.addEventListener('click', ()=>{
    drawer.classList.contains('open') ? close() : open();
  }));
  overlay.addEventListener('click', close);
  drawer.querySelectorAll('a').forEach(a=>a.addEventListener('click', close));
  document.addEventListener('keydown', e=>{ if(e.key === 'Escape') close(); });
}

function fixifyRegisterServiceWorker(){
  if('serviceWorker' in navigator){
    window.addEventListener('load', ()=>{
      navigator.serviceWorker.register('/sw.js').catch(()=>{});
    });
  }
}

document.addEventListener('DOMContentLoaded', ()=>{
  fixifyInitTheme();
  fixifyInitDrawer();
  fixifyRegisterServiceWorker();
  document.querySelectorAll('.theme-toggle-btn').forEach(btn=>{
    btn.addEventListener('click', fixifyToggleTheme);
  });
});
