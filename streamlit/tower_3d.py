"""Bidirectional 3D telecom-tower input component for the Predict tab."""

from __future__ import annotations

from collections.abc import Mapping

import streamlit as st


_HTML = """
<div class="telco3d" aria-label="Interactive 3D telecom tower customer profile">
  <canvas class="telco3d-canvas"></canvas>
  <div class="telco3d-pods" aria-label="Tower equipment sections"></div>
  <aside class="profile-zone zone-left" aria-label="Current profile, left"></aside>
  <aside class="profile-zone zone-right" aria-label="Current profile, right"></aside>
  <div class="telco3d-loading">Building 3D tower…</div>
</div>
"""


_CSS = """
.telco3d {
  position: relative; width: 100%; height: 570px; overflow: hidden;
  box-sizing:border-box;
  isolation: isolate; perspective: 1100px; background:transparent;
  --telco-amber:#F2B84B; --telco-amber-hot:#FFE39A; --telco-silver:#D9DEE1;
}
.telco3d-canvas { position:absolute; inset:0; width:100%; height:100%; display:block; z-index:0; }
.pod-button {
  position:absolute; z-index:6; min-width:124px; min-height:48px; transform:translate(-50%,-50%);
  padding:8px 9px; border:0; border-radius:0; cursor:pointer; color:var(--st-text-color);
  background:transparent; box-shadow:none; text-shadow:0 1px 3px color-mix(in srgb,var(--st-background-color) 80%,transparent);
  transition:opacity .2s, filter .2s, color .2s, text-shadow .2s;
}
.pod-button::before { content:""; position:absolute; top:50%; width:34px; height:1px;
  background:linear-gradient(90deg,color-mix(in srgb,var(--st-text-color) 48%,transparent),transparent); }
.pod-button::after { content:""; position:absolute; top:calc(50% - 3px); width:6px; height:6px;
  border-radius:50%; background:var(--telco-silver); box-shadow:0 0 8px rgba(217,222,225,.8); }
.pod-button.dock-left::before { right:-34px; }
.pod-button.dock-right::before { left:-34px; transform:scaleX(-1); }
.pod-button.dock-left::after { right:-3px; }
.pod-button.dock-right::after { left:-3px; }
.pod-button:hover { color:color-mix(in srgb,var(--st-text-color) 82%,white 18%);
  text-shadow:0 0 9px rgba(255,255,255,.95),0 1px 3px rgba(0,0,0,.22); }
.pod-button:focus:not(:focus-visible) { outline:none; }
.pod-button:focus-visible { outline:2px solid var(--telco-amber); outline-offset:4px; }
.pod-button.is-active {
  color:color-mix(in srgb,var(--st-text-color) 70%,var(--telco-amber) 30%);
  background:transparent; text-shadow:0 0 10px rgba(242,184,75,.85),0 1px 3px rgba(0,0,0,.2);
}
.pod-button.is-active::after { background:var(--telco-amber-hot);
  box-shadow:0 0 5px var(--telco-amber-hot),0 0 15px var(--telco-amber); }
.pod-icon { display:block; font-size:16px; margin-bottom:2px; }
.pod-name { display:block; font-size:12.5px; line-height:1.15; font-weight:760; white-space:nowrap; }
.pod-count { opacity:.68; font-size:10px; margin-left:5px; }
.profile-zone { position:absolute; z-index:9; top:24px; bottom:22px; width:29%; display:flex;
  flex-direction:column; justify-content:center; gap:15px; color:var(--st-text-color); pointer-events:none; }
.zone-left { left:20px; } .zone-right { right:20px; }
.profile-group { opacity:.96; transition:opacity .22s,filter .22s; animation:group-in .32s cubic-bezier(.2,.8,.2,1) both; }
.profile-group.is-muted { opacity:.52; filter:saturate(.55); }
.profile-group.is-editing { opacity:1; filter:none; }
.telco3d.has-active[data-active-side="left"] .zone-left .profile-group.is-muted,
.telco3d.has-active[data-active-side="right"] .zone-right .profile-group.is-muted { display:none; }
.group-title { display:flex; align-items:center; gap:8px; margin:0 2px 9px; color:color-mix(in srgb,var(--st-text-color) 68%,transparent);
  font-size:12px; font-weight:800; letter-spacing:.85px; text-transform:uppercase; }
.group-title::before { content:""; width:5px; height:5px; border:1px solid color-mix(in srgb,var(--telco-silver) 82%,transparent);
  transform:rotate(45deg); }
.profile-group.is-editing .group-title { color:var(--st-text-color); }
.profile-group.is-editing .group-title::before { border-color:var(--telco-amber-hot); background:var(--telco-amber);
  box-shadow:0 0 4px var(--telco-amber-hot),0 0 11px var(--telco-amber); }
.profile-chips,.edit-grid { display:grid; grid-template-columns:repeat(2,minmax(0,1fr)); gap:9px; }
.profile-group[data-section="Charges"] .profile-chips .summary-chip:first-child { grid-column:1/-1; }
.summary-chip { position:relative; min-height:49px; padding:8px 12px 8px 18px; display:flex; flex-direction:column;
  justify-content:center; border:1px solid color-mix(in srgb,var(--st-text-color) 13%,transparent); border-radius:13px;
  background:color-mix(in srgb,var(--st-background-color) 77%,transparent);
  box-shadow:inset 0 1px 0 rgba(255,255,255,.38),0 5px 15px rgba(31,45,56,.05); }
.summary-chip::before { content:""; position:absolute; left:8px; top:50%; width:5px; height:5px; border-radius:50%;
  transform:translateY(-50%); background:var(--telco-amber-hot); box-shadow:0 0 5px var(--telco-amber-hot),0 0 10px rgba(242,184,75,.55); }
.summary-label { overflow:hidden; text-overflow:ellipsis; white-space:nowrap; font-size:10.5px; line-height:1.15; opacity:.62; }
.summary-value { overflow:hidden; text-overflow:ellipsis; white-space:nowrap; font-size:14px; line-height:1.3; font-weight:750; }
.edit-field { min-width:0; padding:9px 10px 10px; border:1px solid color-mix(in srgb,var(--telco-amber) 28%,transparent);
  border-radius:13px; background:color-mix(in srgb,var(--st-background-color) 72%,transparent); pointer-events:auto;
  box-shadow:0 0 16px rgba(242,184,75,.06); animation:field-in .3s cubic-bezier(.2,.8,.2,1) both; }
.edit-field.is-wide { grid-column:1/-1; }
.edit-field.total-charge-card { transition:border-color .2s,background .2s,box-shadow .2s; }
.edit-field.total-charge-card.is-manual-active { border-color:var(--telco-amber);
  background:color-mix(in srgb,var(--telco-amber-hot) 18%,var(--st-background-color));
  box-shadow:0 0 6px rgba(255,227,154,.32),0 0 20px rgba(242,184,75,.22); }
.edit-field.total-charge-card.is-derived { border-color:color-mix(in srgb,var(--telco-silver) 54%,transparent); }
.field-head { display:flex; justify-content:space-between; align-items:center; gap:8px; margin-bottom:6px; }
.field-label { overflow:hidden; text-overflow:ellipsis; white-space:nowrap; font-size:12.5px; font-weight:700; }
.field-value { font-size:11.5px; font-weight:760; color:color-mix(in srgb,var(--st-text-color) 68%,var(--telco-amber) 32%); white-space:nowrap; }
.option-row { display:flex; flex-wrap:wrap; gap:7px; }
.option-button {
  position:relative; min-height:38px; flex:1 1 auto; border:1px solid color-mix(in srgb,var(--st-text-color) 14%,transparent);
  border-radius:11px; padding:7px 10px 7px 20px; cursor:pointer; font:inherit; font-size:13px;
  color:color-mix(in srgb,var(--st-text-color) 76%,transparent); background:color-mix(in srgb,var(--st-background-color) 78%,transparent);
  transition:color .18s,border-color .18s,text-shadow .18s,background .18s;
}
.option-button::before { content:""; position:absolute; left:9px; top:50%; width:6px; height:6px;
  border:1px solid color-mix(in srgb,var(--telco-silver) 80%,transparent); transform:translateY(-50%) rotate(45deg); }
.option-button:hover { color:var(--st-text-color); border-color:var(--telco-silver); }
.option-button[aria-pressed="true"] { color:color-mix(in srgb,var(--st-text-color) 68%,var(--telco-amber) 32%);
  border-color:var(--telco-amber); background:color-mix(in srgb,var(--telco-amber-hot) 25%,transparent);
  box-shadow:0 0 11px rgba(242,184,75,.16); text-shadow:0 0 8px rgba(242,184,75,.4); }
.option-button[aria-pressed="true"]::before { border-color:var(--telco-amber-hot); background:var(--telco-amber);
  box-shadow:0 0 4px var(--telco-amber-hot),0 0 10px var(--telco-amber); }
.field-select { width:100%; min-height:38px; border:1px solid color-mix(in srgb,var(--st-text-color) 16%,transparent);
  border-radius:11px; padding:7px 9px; color:var(--st-text-color); background:var(--st-background-color); font-size:12.5px; }
.field-number { width:100%; min-height:38px; box-sizing:border-box; border:1px solid color-mix(in srgb,var(--st-text-color) 16%,transparent);
  border-radius:11px; padding:7px 9px; color:var(--st-text-color); background:var(--st-background-color);
  font:inherit; font-size:13px; font-weight:720; }
.total-charge-card.is-manual-active .field-number { border-color:var(--telco-amber); outline:none; }
.total-charge-card.is-derived .field-number { opacity:.62; cursor:not-allowed; }
.field-range { width:100%; min-height:22px; accent-color:var(--telco-amber); cursor:pointer; }
@keyframes group-in { from { opacity:0; transform:translateY(8px); } to { transform:translateY(0); } }
@keyframes field-in { from { opacity:0; transform:translateY(7px) scale(.98); } to { opacity:1; transform:translateY(0) scale(1); } }
.telco3d-loading { position:absolute; inset:0; z-index:20; display:grid; place-items:center;
  color:color-mix(in srgb,var(--st-text-color) 60%,transparent); font-size:13px; }
.telco3d.is-ready .telco3d-loading { display:none; }
@media (max-width:900px) {
  .profile-zone { width:31%; gap:10px; }
  .zone-left { left:10px; } .zone-right { right:10px; }
  .profile-chips,.edit-grid { grid-template-columns:1fr; gap:5px; }
  .summary-chip { min-height:34px; }
}
@media (max-width:700px) {
  .telco3d { height:820px; }
  .pod-button { min-width:96px; min-height:42px; }
  .profile-zone { top:500px; bottom:12px; width:calc(50% - 18px); justify-content:flex-start; overflow-y:auto; }
  .zone-left { left:12px; } .zone-right { right:12px; }
}
@media (prefers-reduced-motion:reduce) {
  .profile-group,.edit-field { animation:none; }
}
"""


_JS = r"""
export default async function(component) {
  const { parentElement, data, setStateValue } = component;
  const root = parentElement.querySelector('.telco3d');
  const canvas = root.querySelector('.telco3d-canvas');
  const podsLayer = root.querySelector('.telco3d-pods');
  const leftZone = root.querySelector('.zone-left');
  const rightZone = root.querySelector('.zone-right');
  const renderId=`${Date.now()}-${Math.random()}`;
  root.dataset.renderId=renderId;
  let disposed = false;
  let hovered = false;
  const storageKey='telco-tower-profile-v1';
  const serverValue=data?.value || {};
  const serverState={
    profile:structuredClone(serverValue.profile || {}),
    changed:Array.isArray(serverValue.changed) ? [...serverValue.changed] : [],
    revision:Number(serverValue.revision || 0),
    overrideNonce:Number(serverValue.overrideNonce || 0),
  };
  let cachedState=null;
  try { cachedState=JSON.parse(sessionStorage.getItem(storageKey) || 'null'); } catch (_) {}
  const memoryState=root._towerProfileState || null;
  const cachedOverride=Number(cachedState?.overrideNonce || 0);
  const memoryOverride=Number(memoryState?.overrideNonce || 0);
  const forceServer=serverState.overrideNonce>Math.max(cachedOverride,memoryOverride);
  const candidates=[
    {source:'memory',priority:3,state:memoryState},
    {source:'cache',priority:2,state:cachedState},
    {source:'server',priority:1,state:serverState},
  ].filter(item=>item.state?.profile && typeof item.state.profile==='object');
  let chosen=forceServer
    ? {source:'server',priority:1,state:{...serverState,revision:Math.max(serverState.revision,Number(cachedState?.revision||0),Number(memoryState?.revision||0))+1}}
    : candidates.sort((a,b)=>(Number(b.state.revision||0)-Number(a.state.revision||0)) || (b.priority-a.priority))[0];
  if (!chosen) chosen={source:'server',state:serverState};
  let profile={...structuredClone(serverState.profile || {}),...structuredClone(chosen.state.profile || {})};
  let changed=new Set(Array.isArray(chosen.state.changed) ? chosen.state.changed : []);
  let revision=Number(chosen.state.revision || 0);
  let overrideNonce=Math.max(serverState.overrideNonce,Number(chosen.state.overrideNonce || 0));
  const hydrateServer=chosen.source!=='server' && revision>serverState.revision;
  const localActive=root.dataset.activeSection;
  let active = localActive==='__none__' ? null : (localActive || serverValue.active || null);
  const savedRotation=Number(root.dataset.towerRotation);
  let targetRotation = null;
  let last = performance.now();
  const sections = {
    Demographics: {label:'Demographics', icon:'◉', count:4, angle:0, y:2.38, side:'left', labelDy:38, fields:[
      {key:'in_gender',label:'Gender',options:['Female','Male']},
      {key:'in_senior',label:'Senior citizen',options:['No','Yes']},
      {key:'in_partner',label:'Partner',options:['No','Yes']},
      {key:'in_dependents',label:'Dependents',options:['No','Yes']}]},
    Charges: {label:'Charges',icon:'▣',count:3,angle:-1.35,y:1.05,side:'left',fields:[
      {key:'in_monthly',label:'Monthly charges',type:'range',min:18,max:120,step:.5,suffix:'/mo'},
      {key:'in_auto_total_mode',label:'Calculation mode',options:['Auto','Manual']},
      {key:'in_total_manual',label:'Total Charges (USD)',type:'number',min:0,step:10}]},
    Contract: {label:'Contract',icon:'▤',count:2,angle:1.25,y:1.38,side:'right',fields:[
      {key:'in_tenure',label:'Tenure',type:'range',min:0,max:72,step:1,suffix:' months'},
      {key:'in_contract',label:'Contract type',options:['Month-to-month','One year','Two year']}]},
    Billing: {label:'Billing',icon:'▥',count:2,angle:2.15,y:.35,side:'right',fields:[
      {key:'in_payment',label:'Payment method',type:'select',options:['Electronic check','Mailed check','Bank transfer (automatic)','Credit card (automatic)']},
      {key:'in_paperless',label:'Paperless billing',options:['No','Yes']}]},
    Connection: {label:'Core services',icon:'⌁',count:3,angle:-2.05,y:-.15,side:'left',fields:[
      {key:'in_phone',label:'Phone service',options:['Yes','No']},
      {key:'in_internet',label:'Internet service',options:['DSL','Fiber optic','No']},
      {key:'in_lines',label:'Multiple lines',options:['No','Yes']}]},
    Addons: {label:'Add-ons',icon:'⬡',count:6,angle:.72,y:-.62,side:'right',fields:[
      {key:'in_OnlineSecurity',label:'Online security',options:['No','Yes']},
      {key:'in_OnlineBackup',label:'Online backup',options:['No','Yes']},
      {key:'in_DeviceProtection',label:'Device protection',options:['No','Yes']},
      {key:'in_TechSupport',label:'Tech support',options:['No','Yes']},
      {key:'in_StreamingTV',label:'Streaming TV',options:['No','Yes']},
      {key:'in_StreamingMovies',label:'Streaming movies',options:['No','Yes']}]}
  };

  let THREE;
  try {
    THREE = await import('https://cdn.jsdelivr.net/npm/three@0.180.0/build/three.module.js');
  } catch (error) {
    root.querySelector('.telco3d-loading').textContent = '3D engine could not load. Check the network connection.';
    throw error;
  }
  // A state change can start a newer async renderer while an older Three.js
  // import is still resolving. Only the latest renderer may rebuild the host.
  if (root.dataset.renderId!==renderId) return () => {};
  podsLayer.replaceChildren();
  leftZone.replaceChildren();
  rightZone.replaceChildren();

  const renderer = new THREE.WebGLRenderer({canvas, antialias:true, alpha:true});
  renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
  renderer.outputColorSpace = THREE.SRGBColorSpace;
  renderer.shadowMap.enabled = true;
  renderer.shadowMap.type = THREE.PCFSoftShadowMap;
  const scene = new THREE.Scene();
  const camera = new THREE.PerspectiveCamera(38, 1, .1, 100);
  camera.position.set(0,.45,9.35);
  camera.lookAt(0,.25,0);
  scene.add(new THREE.HemisphereLight(0xf5f7f8,0x596166,2.25));
  const keyLight = new THREE.DirectionalLight(0xffffff,3.3);
  keyLight.position.set(4,7,6); keyLight.castShadow=true; scene.add(keyLight);
  const rimLight = new THREE.PointLight(0xf1f3f4,13,13); rimLight.position.set(-4,2,-2); scene.add(rimLight);

  const tower = new THREE.Group(); scene.add(tower);
  tower.rotation.x = -.04;
  tower.rotation.y = Number.isFinite(savedRotation)
    ? savedRotation
    : (active && sections[active] ? -sections[active].angle : 0);
  const steel = new THREE.MeshStandardMaterial({color:0xb8c0c4,metalness:.86,roughness:.3});
  const darkSteel = new THREE.MeshStandardMaterial({color:0x626b70,metalness:.9,roughness:.26});
  const braceSilver = new THREE.MeshStandardMaterial({color:0xd0d5d7,metalness:.82,roughness:.32});
  const antennaMat = new THREE.MeshStandardMaterial({color:0xe7e9e8,metalness:.34,roughness:.25});
  const podMat = new THREE.MeshStandardMaterial({color:0xc9ced0,metalness:.72,roughness:.27});
  const podHoverMat = new THREE.MeshStandardMaterial({color:0xf2f4f4,emissive:0xd9dee1,emissiveIntensity:.46,metalness:.68,roughness:.2});
  const podActiveMat = new THREE.MeshStandardMaterial({color:0xd5b15f,emissive:0xf2b84b,emissiveIntensity:.85,metalness:.55,roughness:.22});

  function beam(a,b,r,material,parent=tower) {
    const start = new THREE.Vector3(...a), end = new THREE.Vector3(...b);
    const direction = end.clone().sub(start); const length = direction.length();
    const mesh = new THREE.Mesh(new THREE.CylinderGeometry(r,r,length,7),material);
    mesh.position.copy(start).add(end).multiplyScalar(.5);
    mesh.quaternion.setFromUnitVectors(new THREE.Vector3(0,1,0),direction.normalize());
    mesh.castShadow=true; parent.add(mesh); return mesh;
  }
  const levels = 8;
  function corner(y,sx,sz) {
    const t=(y+2.05)/4.05;
    const halfWidth=1.42-(1.08*t);
    return [sx*halfWidth,y,sz*halfWidth];
  }
  for (const sx of [-1,1]) for (const sz of [-1,1]) beam(corner(-2.05,sx,sz),corner(2,sx,sz),.045,darkSteel);
  for (let i=0;i<levels;i++) {
    const y0=-2.05+i*(4.05/levels), y1=-2.05+(i+1)*(4.05/levels);
    for (const z of [-1,1]) {
      const a=corner(y0,-1,z), b=corner(y0,1,z), c=corner(y1,-1,z), d=corner(y1,1,z);
      beam(a,b,.022,steel);
      beam(a,d,.021,braceSilver); beam(b,c,.021,braceSilver);
    }
    for (const x of [-1,1]) {
      const a=corner(y0,x,-1), b=corner(y0,x,1), c=corner(y1,x,-1), d=corner(y1,x,1);
      beam(a,b,.019,steel);
      beam(a,d,.020,steel); beam(b,c,.020,steel);
    }
  }
  const baseCorners=[[-1.5,-2.06,-1.5],[1.5,-2.06,-1.5],[1.5,-2.06,1.5],[-1.5,-2.06,1.5]];
  for (let i=0;i<4;i++) beam(baseCorners[i],baseCorners[(i+1)%4],.07,darkSteel);
  beam([-.72,2.08,0],[.72,2.08,0],.055,darkSteel);
  for (const x of [-.48,0,.48]) {
    const antenna = new THREE.Mesh(new THREE.BoxGeometry(.19,.82,.13),antennaMat);
    antenna.position.set(x,2.48,.05); antenna.castShadow=true; tower.add(antenna);
    beam([x,2.08,0],[x,2.12,.05],.025,darkSteel);
  }
  const beacon = new THREE.Mesh(new THREE.SphereGeometry(.07,16,12),new THREE.MeshStandardMaterial({color:0xff4358,emissive:0xff1735,emissiveIntensity:2.4}));
  beacon.position.set(0,2.93,0); tower.add(beacon);

  const podObjects = {};
  for (const [name,cfg] of Object.entries(sections)) {
    const pod = new THREE.Group();
    const levelT=Math.max(0,Math.min(1,(cfg.y+2.05)/4.05));
    const radial=(1.42-(1.08*levelT))+.34;
    pod.position.set(Math.sin(cfg.angle)*radial,cfg.y,Math.cos(cfg.angle)*radial);
    pod.rotation.y=cfg.angle;
    const box = new THREE.Mesh(new THREE.BoxGeometry(.66,.32,.28),podMat); box.castShadow=true; pod.add(box);
    const rim = new THREE.Mesh(new THREE.BoxGeometry(.72,.37,.31),new THREE.MeshStandardMaterial({color:0xe3e6e7,wireframe:true,transparent:true,opacity:.76})); pod.add(rim);
    tower.add(pod); pod.userData={box,base:pod.position.clone(),cfg}; podObjects[name]=pod;
    const anchor=[Math.sin(cfg.angle)*(radial-.45),cfg.y,Math.cos(cfg.angle)*(radial-.45)];
    beam(anchor,[pod.position.x,pod.position.y,pod.position.z],.035,darkSteel);
    const button=document.createElement('button'); button.type='button'; button.className='pod-button'; button.dataset.section=name;
    button.setAttribute('aria-label',`${cfg.label}, ${cfg.count} fields`);
    button.innerHTML=`<span class="pod-icon">${cfg.icon}</span><span class="pod-name">${cfg.label}<span class="pod-count">${cfg.count}</span></span>`;
    button.addEventListener('mouseenter',()=>{ if(active!==name) box.material=podHoverMat; });
    button.addEventListener('mouseleave',()=>{ if(active!==name) box.material=podMat; });
    button.addEventListener('click',()=>activate(name)); podsLayer.appendChild(button); cfg.button=button;
  }

  const ground = new THREE.Mesh(new THREE.CircleGeometry(2.3,64),new THREE.ShadowMaterial({color:0x31383c,opacity:.14}));
  ground.rotation.x=-Math.PI/2; ground.position.y=-2.12; ground.receiveShadow=true; scene.add(ground);

  function displayValue(field,value) {
    if (field.key==='in_monthly') return `$${Number(value).toFixed(1)}${field.suffix}`;
    if (field.key==='in_total_manual') {
      const amount=profile.in_auto_total_mode==='Auto'
        ? Number(profile.in_tenure||0)*Number(profile.in_monthly||0)
        : Number(value||0);
      return `$${amount.toLocaleString(undefined,{minimumFractionDigits:2,maximumFractionDigits:2})}`;
    }
    if (field.type==='range') return `${value}${field.suffix||''}`;
    const short={'Bank transfer (automatic)':'Bank transfer','Credit card (automatic)':'Credit card'};
    return short[value] || value;
  }
  function saveLocalState() {
    const state={profile:structuredClone(profile),changed:[...changed],revision,overrideNonce};
    root._towerProfileState=state;
    try { sessionStorage.setItem(storageKey,JSON.stringify(state)); } catch (_) {}
  }
  function commit(key,value) {
    profile[key]=value;
    changed.add(key); revision+=1; saveLocalState();
    if (key==='in_auto_total_mode' || (profile.in_auto_total_mode==='Auto' && (key==='in_monthly' || key==='in_tenure'))) renderOverview();
    publishState();
  }
  function publishState() {
    setStateValue('value',{profile:structuredClone(profile),active,changed:[...changed],revision,overrideNonce});
  }
  function renderEditorField(field,index,animateFields) {
      const chip=document.createElement('div'); chip.className='edit-field'; chip.style.animationDelay=`${index*35}ms`;
      if (!animateFields) chip.style.animation='none';
      if (field.type==='range' || field.type==='select' || field.options?.length>2) chip.classList.add('is-wide');
      if (field.key==='in_total_manual') {
        chip.classList.add('total-charge-card');
        chip.classList.add(profile.in_auto_total_mode==='Manual' ? 'is-manual-active' : 'is-derived');
      }
      const head=document.createElement('div'); head.className='field-head';
      const label=document.createElement('span'); label.className='field-label'; label.textContent=field.label;
      const value=document.createElement('span'); value.className='field-value';
      value.textContent=field.key==='in_total_manual' && profile.in_auto_total_mode==='Auto'
        ? `⚡ ${displayValue(field,profile[field.key])}` : `${displayValue(field,profile[field.key])} ✓`;
      head.append(label,value); chip.append(head);
      if (field.type==='range') {
        const input=document.createElement('input'); input.type='range'; input.className='field-range';
        input.min=field.min; input.max=field.max; input.step=field.step; input.value=profile[field.key];
        input.setAttribute('aria-label',field.label);
        input.addEventListener('input',()=>{ value.textContent=`${displayValue(field,Number(input.value))} ✓`; });
        input.addEventListener('change',()=>commit(field.key,Number(input.value))); chip.append(input);
      } else if (field.type==='select') {
        const select=document.createElement('select'); select.className='field-select'; select.setAttribute('aria-label',field.label);
        field.options.forEach(option=>{ const el=document.createElement('option'); el.value=option; el.textContent=option; el.selected=option===profile[field.key]; select.append(el); });
        select.addEventListener('change',()=>{ value.textContent=`${displayValue(field,select.value)} ✓`; commit(field.key,select.value); }); chip.append(select);
      } else if (field.type==='number') {
        const input=document.createElement('input'); input.type='number'; input.className='field-number';
        input.min=field.min; input.step=field.step;
        input.value=profile.in_auto_total_mode==='Auto'
          ? Number(profile.in_tenure||0)*Number(profile.in_monthly||0)
          : Number(profile[field.key]||0);
        input.disabled=profile.in_auto_total_mode!=='Manual'; input.setAttribute('aria-label',field.label);
        let numberTimer; let lastCommitted=Number(profile[field.key]||0);
        const numberValue=()=>Math.max(0,Number(input.value)||0);
        const showNumber=()=>{ const next=numberValue();
          value.textContent=`$${next.toLocaleString(undefined,{minimumFractionDigits:2,maximumFractionDigits:2})} ✓`;
          return next; };
        const commitNumber=()=>{ const next=showNumber();
          if (next===lastCommitted) return;
          lastCommitted=next; commit(field.key,next); };
        input.addEventListener('input',()=>{ showNumber(); clearTimeout(numberTimer);
          numberTimer=setTimeout(commitNumber,400); });
        input.addEventListener('blur',()=>{ clearTimeout(numberTimer); commitNumber(); }); chip.append(input);
      } else {
        const row=document.createElement('div'); row.className='option-row';
        field.options.forEach(option=>{ const button=document.createElement('button'); button.type='button'; button.className='option-button';
          button.textContent=option; button.setAttribute('aria-pressed',String(option===profile[field.key]));
          button.addEventListener('click',()=>{ profile[field.key]=option; value.textContent=`${displayValue(field,option)} ✓`;
            row.querySelectorAll('button').forEach(el=>el.setAttribute('aria-pressed',String(el===button))); commit(field.key,option); }); row.append(button); });
        chip.append(row);
      }
      return chip;
  }
  function renderOverview() {
    leftZone.replaceChildren(); rightZone.replaceChildren();
    const activeToken=active || '__none__';
    const animateGroups=root.dataset.renderedActive!==activeToken;
    Object.entries(sections).forEach(([name,cfg],sectionIndex)=>{
      const group=document.createElement('section');
      group.className='profile-group'; group.dataset.section=name;
      group.style.animationDelay=`${sectionIndex*35}ms`;
      if (!animateGroups) group.style.animation='none';
      if (active && active!==name) group.classList.add('is-muted');
      if (active===name) group.classList.add('is-editing');
      const title=document.createElement('div'); title.className='group-title'; title.textContent=cfg.label;
      group.append(title);
      if (active===name) {
        const grid=document.createElement('div'); grid.className='edit-grid';
        cfg.fields.forEach((field,index)=>grid.append(renderEditorField(field,index,animateGroups)));
        group.append(grid);
      } else {
        const chips=document.createElement('div'); chips.className='profile-chips';
        cfg.fields.forEach(field=>{
          const chip=document.createElement('div'); chip.className='summary-chip';
          const label=document.createElement('span'); label.className='summary-label'; label.textContent=field.label;
          const value=document.createElement('span'); value.className='summary-value'; value.textContent=displayValue(field,profile[field.key]) || '—';
          chip.append(label,value); chips.append(chip);
        });
        group.append(chips);
      }
      (cfg.side==='left' ? leftZone : rightZone).append(group);
    });
    root.dataset.renderedActive=activeToken;
  }
  function activate(name,publish=true) {
    if (publish && active===name) { closeActive(); return; }
    active=name; root.dataset.activeSection=name;
    const cfg=sections[name]; targetRotation=-cfg.angle;
    root.dataset.activeSide=cfg.side;
    root.classList.add('has-active');
    Object.entries(sections).forEach(([key,item])=>item.button.classList.toggle('is-active',key===name));
    Object.entries(podObjects).forEach(([key,pod])=>{pod.userData.box.material=key===name?podActiveMat:podMat;});
    renderOverview();
    if (publish) publishState();
  }
  function closeActive() {
    active=null; root.dataset.activeSection='__none__';
    targetRotation=null; root.classList.remove('has-active'); delete root.dataset.activeSide;
    Object.values(sections).forEach(item=>item.button.classList.remove('is-active'));
    Object.values(podObjects).forEach(pod=>pod.userData.box.material=podMat);
    renderOverview();
    publishState();
  }
  root.addEventListener('mouseenter',()=>{hovered=true;root.classList.add('is-paused');});
  root.addEventListener('mouseleave',()=>{hovered=false;root.classList.remove('is-paused');});

  function resize() {
    const w=root.clientWidth,h=root.clientHeight; renderer.setSize(w,h,false); camera.aspect=w/h; camera.updateProjectionMatrix();
  }
  const resizeObserver=new ResizeObserver(resize); resizeObserver.observe(root); resize();
  root.classList.add('is-ready');
  saveLocalState();
  if (active && sections[active]) activate(active,false); else renderOverview();
  if (hydrateServer || revision>serverState.revision) publishState();

  function animate(now) {
    if (disposed) return; const dt=Math.min((now-last)/1000,.05); last=now;
    if (active && targetRotation!==null) {
      let delta=((targetRotation-tower.rotation.y+Math.PI)%(Math.PI*2))-Math.PI;
      tower.rotation.y += delta*Math.min(1,dt*4.2);
    } else if (!hovered && !window.matchMedia('(prefers-reduced-motion:reduce)').matches) {
      tower.rotation.y += dt*.27;
    }
    root.dataset.towerRotation=String(tower.rotation.y);
    for (const [name,pod] of Object.entries(podObjects)) {
      const selected=name===active, amount=selected ? .16 : 0;
      const cfg=pod.userData.cfg, radial=new THREE.Vector3(Math.sin(cfg.angle),0,Math.cos(cfg.angle));
      const target=pod.userData.base.clone().add(radial.multiplyScalar(amount)); pod.position.lerp(target,.12);
      const world=new THREE.Vector3(); pod.getWorldPosition(world); const projected=world.clone().project(camera);
      const podX=(projected.x*.5+.5)*root.clientWidth, y=(-projected.y*.5+.5)*root.clientHeight+(cfg.labelDy||0);
      const side=(Math.abs(podX-root.clientWidth/2)>28 ? (podX<root.clientWidth/2?'left':'right') : cfg.side);
      const x=podX+(side==='left'?-82:82);
      cfg.button.classList.toggle('dock-left',side==='left'); cfg.button.classList.toggle('dock-right',side==='right');
      cfg.button.style.left=`${x}px`; cfg.button.style.top=`${y}px`;
      const depth=Math.max(.72,Math.min(1,(1-projected.z)*2.6)); cfg.button.style.opacity=String(depth);
      cfg.button.style.filter=`brightness(${.72+depth*.28})`;
      if (active && !selected) cfg.button.style.opacity=String(depth*.68);
      cfg.button.style.zIndex=String(Math.round(5+depth*3));
    }
    const pulse=1+Math.sin(now*.005)*.35; beacon.material.emissiveIntensity=2.1*pulse;
    podActiveMat.emissiveIntensity=.68+pulse*.22;
    renderer.render(scene,camera); requestAnimationFrame(animate);
  }
  requestAnimationFrame(animate);
  return () => { disposed=true; resizeObserver.disconnect(); renderer.dispose(); scene.traverse(obj=>{obj.geometry?.dispose?.(); if(obj.material&&!Array.isArray(obj.material)) obj.material.dispose?.();}); };
}
"""


_tower_component = st.components.v2.component(
    "telco_tower_3d",
    html=_HTML,
    css=_CSS,
    js=_JS,
)


_PROFILE_KEYS = {
    "in_gender", "in_senior", "in_partner", "in_dependents",
    "in_monthly", "in_auto_total_mode", "in_total_manual", "in_tenure", "in_contract",
    "in_payment", "in_paperless", "in_phone", "in_internet", "in_lines",
    "in_OnlineSecurity", "in_OnlineBackup", "in_DeviceProtection",
    "in_TechSupport", "in_StreamingTV", "in_StreamingMovies",
}


def _apply_tower_value(value) -> None:
    """Copy one component value payload into native prediction session keys."""
    if not isinstance(value, Mapping):
        return
    incoming_revision = int(value.get("revision", 0) or 0)
    current_revision = int(st.session_state.get("tower_3d_revision", 0) or 0)
    incoming_override = int(value.get("overrideNonce", 0) or 0)
    current_override = int(st.session_state.get("tower_3d_override_nonce", 0) or 0)
    if incoming_revision < current_revision or incoming_override < current_override:
        return
    profile = value.get("profile", {})
    if not isinstance(profile, Mapping):
        return
    for key in _PROFILE_KEYS:
        if key in profile:
            st.session_state[key] = profile[key]
    if "active" in value:
        st.session_state["tower_3d_active"] = value["active"]
    if isinstance(value.get("changed"), list):
        st.session_state["tower_3d_changed"] = value["changed"]
    st.session_state["tower_3d_revision"] = incoming_revision
    st.session_state["tower_3d_override_nonce"] = max(
        current_override,
        incoming_override,
    )


def _sync_tower_state() -> None:
    """Callback path; direct result syncing below is retained as a fallback."""
    component_state = st.session_state.get("telco_tower_3d_state")
    value = getattr(component_state, "value", None)
    if value is None and isinstance(component_state, Mapping):
        value = component_state.get("value")
    _apply_tower_value(value)


def render_tower_3d(profile: dict) -> dict:
    """Mount the interactive scene and wire its values to Session State."""
    initial = {
        "profile": profile,
        "active": st.session_state.get("tower_3d_active"),
        "changed": st.session_state.get("tower_3d_changed", []),
        "revision": st.session_state.get("tower_3d_revision", 0),
        "overrideNonce": st.session_state.get("tower_3d_override_nonce", 0),
    }
    result = _tower_component(
        key="telco_tower_3d_state",
        data={"value": initial},
        default={"value": initial},
        height=570,
        on_value_change=_sync_tower_state,
    )
    # Component v2 returns its current state during the rerun triggered by a
    # field change. Applying it here also makes the new value available to the
    # billing summary and prediction call in this same script pass.
    _apply_tower_value(getattr(result, "value", None))
    return {key: st.session_state.get(key, profile.get(key)) for key in _PROFILE_KEYS}
