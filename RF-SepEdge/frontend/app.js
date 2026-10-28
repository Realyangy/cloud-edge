/* RF-SepEdge showcase frontend. No inference, no trained model running in browser.
   All displayed observations come from demo-results/manifest.json and linked files. */
(() => {
  'use strict';
  const $ = (id) => document.getElementById(id);
  const DEMO_REMOTE = 'https://raw.githubusercontent.com/Realyangy/cloud-edge/main/RF-SepEdge/demo-results/';
  const DEMO_LOCAL = new URL('../demo-results/', window.location.href).href;
  const INTERFERENCES = ['CommSignal2','CommSignal3','CommSignal5G1','EMISignal1'];
  const TRACKS = {
    mit: [
      ['混合输入', '查看目标通信信号与干扰信号叠加形成的复数 IQ。', '目标信号与干扰帧相加形成混合输入；WaveNet 推理时不会看到目标信号真值。'],
      ['WaveNet 分离', 'WaveNet 接收混合 IQ，并输出估计的目标复数波形。', '模型根据当前目标信号与干扰类型选择对应的官方预训练权重；这里回放的是已经完成的推理结果。'],
      ['目标波形恢复', '比较目标真值、混合输入和 WaveNet 恢复结果的波形与时频特征。', '目标 SOI 真值只用于结果对比；恢复波形来自 WaveNet 输出，不是把真值直接用于推理。'],
      ['效果评估', '使用官方解调流程，对比恢复前后的 BER 和波形 MSE。', 'BER 为错误比特数 / 总比特数；零观测误码不代表理论误码率严格为零。']
    ],
    spectrum: [
      ['宽带时频图', '将 262,144 点宽带复数 IQ 转换为二维时频图。', '本图来自 TorchSig 固定种子生成的真实实验数据流，时间和频率轴根据采样率换算。'],
      ['YOLO 检测定位', 'YOLO11n 定位不同信号出现的时间范围与频率范围。', '检测框由已保存的 YOLO 预测结果重现。点击框，可选中该目标。'],
      ['目标 IQ 提取', '根据检测框下变频、低通滤波、重采样，并进行复数 RMS 归一化。', '每个目标提取成 2 × 4096 的 I/Q 输入；显示的局部特征来自本次真实实验。'],
      ['XCiT 调制识别', 'XCiT V4 对提取 IQ 进行 57 类细分预测，并汇总为调制家族概率。', '预测家族通过对该家族各细分类概率求和得到，不一定等于细粒度 Top-1 所属家族。']
    ]
  };
  const state = {
    root: null, manifest: null, task: 'mit', mitStep: 0, mitView: 'waveforms', mitChart: 'ber',
    mitSample: null, mitRows: {}, playbackTimer: null, playbackTask: null, spectrumStep: 0, spectrumSample: 60,
    spectrumData: null, targetIndex: 0, spectrumInsight: 'compare',
    spectrumSummary: {}, trainingHistory: null, yoloSummary: null, lightboxUrl: ''
  };
  const cache = new Map();
  const fmt = (n, digits = 2) => Number.isFinite(Number(n)) ? Number(n).toFixed(digits) : '—';
  const pct = (n, digits = 2) => Number.isFinite(Number(n)) ? `${(Number(n)*100).toFixed(digits)}%` : '—';
  const escape = (x) => String(x ?? '').replace(/[&<>"']/g, a => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[a]));
  const asset = (path) => new URL(String(path).replace(/^\/+/, ''), state.root || DEMO_REMOTE).href;
  const rootAsset = (path, root) => new URL(String(path).replace(/^\/+/, ''), root).href;

  function showError(message) {
    $('error-banner').hidden = false;
    $('error-banner').textContent = message;
    $('data-status').textContent = '数据连接异常';
    document.querySelector('.status-line').classList.add('error');
  }
  async function fetchJSON(path) {
    if (cache.has(path)) return cache.get(path);
    const res = await fetch(asset(path), {cache: 'no-store'});
    if (!res.ok) throw new Error(`${res.status} ${path}`);
    const json = await res.json();
    cache.set(path, json);
    return json;
  }
  async function fetchCSV(path) {
    if (cache.has(path)) return cache.get(path);
    const res = await fetch(asset(path), {cache: 'no-store'});
    if (!res.ok) throw new Error(`${res.status} ${path}`);
    const rows = parseCSV(await res.text());
    cache.set(path, rows);
    return rows;
  }
  function parseCSV(text) {
    const lines = text.replace(/\r/g, '').trim().split('\n').filter(Boolean);
    if (lines.length < 2) return [];
    const keys = lines[0].split(',').map(x => x.trim());
    return lines.slice(1).map(line => {
      const values = line.split(',');
      return Object.fromEntries(keys.map((k, i) => [k, values[i]]));
    });
  }
  function imageLoad(element, placeholder, path) {
    const img = $(element), cover = $(placeholder);
    if (!img) return;
    if (cover) { cover.hidden = false; cover.textContent = '正在加载实验图像…'; }
    img.style.opacity = '0';
    let attemptedRemote = false;
    img.onload = () => { img.style.opacity = '1'; if (cover) cover.hidden = true; };
    img.onerror = () => {
      if (!attemptedRemote && state.root !== DEMO_REMOTE) {
        attemptedRemote = true;
        img.src = rootAsset(path, DEMO_REMOTE);
        return;
      }
      img.style.opacity = '0';
      if (cover) { cover.hidden = false; cover.textContent = '未找到图像素材，请确认 demo-results/ 已同步。'; }
    };
    img.src = asset(path);
  }
  function setPlainImage(element, path) {
    const img = $(element);
    img.onerror = () => { if (state.root !== DEMO_REMOTE && !img.dataset.fallback) {
      img.dataset.fallback = '1'; img.src = rootAsset(path, DEMO_REMOTE);
    }};
    img.removeAttribute('data-fallback');
    img.src = asset(path);
  }
  function activateButtons(parent, attribute, value) {
    $(parent).querySelectorAll(`[${attribute}]`).forEach(b => {
      const active = b.getAttribute(attribute) === String(value);
      b.classList.toggle('active', active);
      if (attribute === 'data-step') b.classList.toggle('visited', Number(b.getAttribute(attribute)) < Number(value));
      b.setAttribute('aria-pressed', String(active));
    });
  }
  function stopPlayback() {
    if (state.playbackTimer !== null) {clearInterval(state.playbackTimer);state.playbackTimer=null;}
    state.playbackTask=null;
    for(const name of ['mit','spectrum']){const b=$(name+'-auto');if(b){b.classList.remove('playing');b.textContent='▶ 自动回放';}}
  }
  function togglePlayback(task) {
    if(state.playbackTask===task){stopPlayback();return;}
    stopPlayback();state.playbackTask=task;const btn=$(task+'-auto');btn.textContent='Ⅱ 暂停回放';btn.classList.add('playing');
    const update=task==='mit'?setMITStep:setSpectrumStep;
    update(0);
    state.playbackTimer=setInterval(()=>{const idx=task==='mit'?state.mitStep:state.spectrumStep;if(idx>=3){stopPlayback();return;}update(idx+1);if(idx+1===3)stopPlayback();},2800);
  }
  function setTask(task) {
    stopPlayback();
    state.task = task;
    $('view-mit').hidden = task !== 'mit'; $('view-spectrum').hidden = task !== 'spectrum';
    for (const key of ['mit','spectrum']) {
      const b = $('tab-' + key); const active = key === task;
      b.classList.toggle('active',active); b.setAttribute('aria-selected',String(active));
    }
    window.location.hash = task === 'spectrum' ? '#spectrum' : '#mit';
  }
  function updateMITSelection() {
    stopPlayback();
    if (!state.manifest) return;
    const soi = $('mit-soi').value, interf = $('mit-interference').value, sinr = Number($('mit-sinr').value);
    state.mitSample = state.manifest.mit_wavenet.samples.find(s => s.soi_type === soi && s.interference === interf && Number(s.nominal_sinr_db) === sinr) || null;
    if (!state.mitSample) {
      $('mit-sample-id').textContent = '无匹配样本';
      return;
    }
    state.mitStep = 0; state.mitView = 'waveforms';
    renderMIT(); renderMITCharts();
  }
  function renderMIT() {
    const r = state.mitSample; if (!r) return;
    activateButtons('mit-steps', 'data-step', state.mitStep);
    activateButtons('mit-view-switch', 'data-view', state.mitView);
    $('mit-prev').disabled = state.mitStep === 0;
    $('mit-next').disabled = state.mitStep === 3;
    $('mit-next').textContent = state.mitStep === 3 ? '已到最后一步' : '下一步';
    $('mit-sample-id').textContent = `#${r.sample_index} · Seed 42`;
    const info = TRACKS.mit[state.mitStep];
    $('mit-stage-label').textContent = `0${state.mitStep + 1} · ${info[0]}`;
    $('mit-stage-text').textContent = info[1];
    $('mit-explainer-title').textContent = `当前阶段 · ${info[0]}`;
    $('mit-explainer-body').textContent = info[2];
    $('mit-visual-title').textContent = state.mitView === 'waveforms' ? '同一样本的 I/Q 波形对比' : '同一样本的时频特征对比';
    $('mit-visual-subtitle').textContent = '混合输入 · 目标真值 · WaveNet 恢复';
    const path = state.mitView === 'waveforms' ? r.waveforms : r.spectrograms;
    imageLoad('mit-main-image','mit-image-placeholder',path);
    state.mitCurrentImage = path;
    $('mit-ber-input').textContent = pct(r.input_ber);
    $('mit-ber-out').textContent = pct(r.wavenet_ber);
    const drop = Number(r.input_ber) > 0 ? (Number(r.input_ber) - Number(r.wavenet_ber)) / Number(r.input_ber) : 0;
    $('mit-ber-drop').textContent = `${drop >= 0 ? '↓ ' : '↑ '}${fmt(Math.abs(drop) * 100, 1)}%`;
    $('mit-drop-bar').style.width = `${Math.max(0,Math.min(100,drop * 100))}%`;
    $('mit-actual-sinr').textContent = `${fmt(r.actual_input_sinr_db)} dB`;
    $('mit-mse').textContent = `${fmt(r.wavenet_mse_db)} dB`;
    // From manifest MSE/BER; sample-level bit counts can differ by SOI.
    const nbits = r.soi_type === 'QPSK' ? 5120 : 57344;
    const wrong = Math.round(Number(r.wavenet_ber) * nbits);
    $('mit-bit-errors').textContent = `${wrong} / ${nbits.toLocaleString('en-US')}`;
  }
  function setMITStep(step) {
    state.mitStep = Math.max(0,Math.min(3,step));
    if (state.mitStep === 2) state.mitView = 'spectrograms';
    else if (state.mitStep === 0 || state.mitStep === 1) state.mitView = 'waveforms';
    renderMIT();
  }
  function renderMITCharts() {
    const soi = $('mit-soi').value, interf = $('mit-interference').value;
    const rows = (state.mitRows[interf] || []).filter(x => x.soi_type === soi).sort((a,b)=>Number(a.nominal_sinr_db)-Number(b.nominal_sinr_db));
    activateButtons('mit-chart-switch','data-chart',state.mitChart);
    $('mit-chart-title').textContent = state.mitChart === 'ber' ? 'BER 随 SINR 变化' : 'MSE 随 SINR 变化';
    if (!rows.length) { $('mit-line-chart').innerHTML = '<div class="empty-state">正在加载批量测试记录…</div>'; return; }
    const vals = rows.map(r => ({x:Number(r.nominal_sinr_db), a:Number(state.mitChart==='ber'?r.input_ber:r.mean_input_mse_db)*(state.mitChart==='ber'?100:1),b:Number(state.mitChart==='ber'?r.wavenet_ber:r.mean_wavenet_mse_db)*(state.mitChart==='ber'?100:1)}));
    $('mit-line-chart').innerHTML = renderLineChart(vals, state.mitChart === 'ber' ? '%' : 'dB', state.mitChart === 'ber' ? 'SINR (dB)' : 'SINR (dB)', state.mitChart === 'ber');
    const aggregates = INTERFERENCES.map(name=>{
      const rs=(state.mitRows[name]||[]).filter(x=>x.soi_type===soi);
      const bits=rs.reduce((s,x)=>s+Number(x.bits_evaluated||0),0);
      const errors=rs.reduce((s,x)=>s+Number(x.wavenet_bit_errors||0),0);
      return {name,percent:bits?errors/bits*100:null};
    });
    $('mit-compare-desc').textContent = `${soi==='QPSK'?'QPSK':'OFDM-QPSK'} · 4 类干扰 · 每类 70 条`;
    const max=Math.max(1,...aggregates.map(x=>x.percent||0));
    $('mit-interference-bars').innerHTML=aggregates.map(x=>`<div class="comparison-row"><span>${escape(x.name)}</span><div class="bar-track"><div class="bar-fill" style="width:${Math.max(0,(x.percent||0)/max*100)}%"></div></div><strong>${x.percent===null?'—':fmt(x.percent)+'%'}</strong></div>`).join('');
  }
  function renderLineChart(rows, suffix, xLabel, nonnegative) {
    if (!rows.length) return '<div class="empty-state">暂无曲线数据</div>';
    const W=760,H=300,left=68,right=30,top=26,bottom=55;
    const all=rows.flatMap(x=>[x.a,x.b]).filter(Number.isFinite);if(!all.length)return '<div class="empty-state">暂无可显示数值</div>';
    let min=Math.min(...all),max=Math.max(...all);
    if(nonnegative)min=0;
    if(min===max){min-=1;max+=1}
    else {let pad=(max-min)*.12;min-=nonnegative?0:pad;max+=pad}
    const minx=Math.min(...rows.map(x=>x.x)), maxx=Math.max(...rows.map(x=>x.x));
    const X=v=>left+(v-minx)/Math.max(1,maxx-minx)*(W-left-right);
    const Y=v=>top+(max-v)/(max-min)*(H-top-bottom);
    const path=k=>rows.map((r,i)=>`${i?'L':'M'}${X(r.x).toFixed(1)} ${Y(r[k]).toFixed(1)}`).join(' ');
    let s=`<svg viewBox="0 0 ${W} ${H}" role="img" aria-label="真实评估曲线">`;
    for(let i=0;i<=4;i++){const y=top+i*(H-top-bottom)/4;const v=max-i*(max-min)/4;s+=`<line x1="${left}" y1="${y}" x2="${W-right}" y2="${y}" stroke="#e3eaf4"/><text x="${left-11}" y="${y+5}" text-anchor="end" fill="#71839d" font-size="14">${fmt(v,nonnegative?0:1)}${suffix}</text>`}
    for(const r of rows)s+=`<text x="${X(r.x)}" y="${H-25}" fill="#71839d" font-size="14" text-anchor="middle">${r.x}</text>`;
    s+=`<text x="${(W+left-right)/2}" y="${H-4}" fill="#71839d" text-anchor="middle" font-size="14">${escape(xLabel)}</text>`;
    s+=`<path d="${path('a')}" fill="none" stroke="#91bbff" stroke-width="3.5" stroke-linejoin="round" stroke-linecap="round"/><path d="${path('b')}" fill="none" stroke="#2f61ed" stroke-width="3.5" stroke-linejoin="round" stroke-linecap="round"/>`;
    for(const r of rows)for(const [k,c] of [['a','#91bbff'],['b','#2f61ed']])s+=`<circle cx="${X(r.x)}" cy="${Y(r[k])}" r="5" fill="white" stroke="${c}" stroke-width="3"><title>SINR ${r.x} · ${k==='a'?'输入':'恢复'} ${fmt(r[k],3)}${suffix}</title></circle>`;
    s+='</svg>';return s;
  }

  async function selectSpectrumSample(n) {
    stopPlayback();
    if (!state.manifest) return;
    state.spectrumSample = Number(n);
    const rec=state.manifest.yolo_xcit_v4.samples.find(x=>Number(x.sample_index)===state.spectrumSample);
    if (!rec) return;
    state.spectrumRecord=rec;
    state.spectrumStep=0; state.targetIndex=0; state.spectrumData=null;
    $('spectrum-sample-count').textContent='正在加载…';
    $('spectrum-target-tbody').innerHTML='<tr><td colspan="7">正在加载目标记录…</td></tr>';
    try {
      state.spectrumData=await fetchJSON(rec.sample_json);
      state.targetIndex=state.spectrumData.predictions.findIndex(x=>x.ground_truth);
      if (state.targetIndex<0)state.targetIndex=0;
      renderSpectrum();
    } catch(e){showError('读取频谱样本失败：'+e.message)}
  }
  function renderSpectrum() {
    const d=state.spectrumData,r=state.spectrumRecord;if(!d||!r)return;
    const pred=d.predictions[state.targetIndex];if(!pred)return;
    activateButtons('spectrum-steps','data-step',state.spectrumStep);
    const mode=state.spectrumStep===0?'raw':state.spectrumStep===2?'features':'boxes';
    activateButtons('spectrum-image-mode','data-mode',mode);
    $('spectrum-prev').disabled=state.spectrumStep===0;
    $('spectrum-next').disabled=state.spectrumStep===3;
    $('spectrum-next').textContent=state.spectrumStep===3?'已到最后一步':'下一步';
    $('spectrum-sample-count').textContent=`${d.metrics.ground_truth_count} 个真实目标 · ${d.metrics.prediction_count} 个检测框`;
    const st=TRACKS.spectrum[state.spectrumStep];
    $('spectrum-stage-label').textContent=`0${state.spectrumStep+1} · ${st[0]}`;
    $('spectrum-stage-text').textContent=st[1];
    $('spectrum-explainer-title').textContent=`当前阶段 · ${st[0]}`;
    $('spectrum-explainer-body').textContent=st[2];
    $('freq-plot-shell').hidden=mode==='features';$('spectrum-feature-view').hidden=mode!=='features';
    if(mode==='features'){
      $('spectrum-visual-title').textContent='检测目标的局部 IQ 特征';
      $('spectrum-visual-subtitle').textContent='当前选中目标 #' + (state.targetIndex+1);
      const ft=(d.feature_assets||[]).find(f=>Number(f.prediction_index)===Number(pred.prediction_index)) || d.feature_assets[state.targetIndex];
      const rel=ft?.file ? r.sample_json.replace(/\/sample\.json$/,'/')+ft.file : r.local_features;
      setPlainImage('spectrum-feature-image',rel);
      state.spectrumCurrentImage=rel;
    }else{
      $('spectrum-visual-title').textContent=mode==='raw'?'原始宽带时频图':'YOLO11n 信号检测定位';
      $('spectrum-visual-subtitle').textContent='横轴：时间 (ms) · 纵轴：频率 (MHz)';
      imageLoad('spectrum-image','spectrum-image-placeholder',r.spectrogram);
      $('spectrum-boxes').innerHTML = mode==='boxes'?d.predictions.map((p,i)=>{
        const [x0,y0,x1,y1]=p.box_xyxy_normalized.map(v=>Math.min(1,Math.max(0,Number(v))));
        return `<button type="button" class="detect-box ${i===state.targetIndex?'active':''}" data-target="${i}" aria-label="选择检测目标 ${i+1}" style="left:${x0*100}%;top:${y0*100}%;width:${Math.max(.1,(x1-x0)*100)}%;height:${Math.max(.1,(y1-y0)*100)}%"><span class="detect-box-label">#${i+1}</span></button>`;
      }).join(''):'';
      $('spectrum-boxes').querySelectorAll('[data-target]').forEach(b=>b.addEventListener('click',()=>selectTarget(Number(b.dataset.target))));
      state.spectrumCurrentImage=mode==='boxes'?r.yolo_overlay:r.spectrogram;
    }
    // Keep target chooser and metrics synchronized with selected bounding box.
    $('target-buttons').innerHTML=d.predictions.map((p,i)=>`<button type="button" data-target="${i}" class="${i===state.targetIndex?'active':''}" aria-pressed="${i===state.targetIndex}">#${i+1}</button>`).join('');
    $('target-buttons').querySelectorAll('[data-target]').forEach(b=>b.addEventListener('click',()=>selectTarget(Number(b.dataset.target))));
    const detail=pred.with_rms||{};
    $('target-family').textContent=state.spectrumStep===3?(detail.family_top1||'—'):'未展示';
    $('target-fine').textContent=state.spectrumStep===3?(detail.fine_top1||'—'):'等待分类阶段';
    $('target-yolo-conf').textContent=state.spectrumStep>=1?pct(pred.confidence,1):'—';
    $('target-center').textContent=state.spectrumStep>=1?`${Number(pred.extraction?.predicted_center_offset_hz)/1e6>=0?'+':''}${fmt(Number(pred.extraction?.predicted_center_offset_hz)/1e6,2)} MHz`:'—';
    $('target-bandwidth').textContent=state.spectrumStep>=1?`${fmt(Number(pred.extraction?.predicted_bandwidth_hz)/1e6,2)} MHz`:'—';
    $('target-duration').textContent=state.spectrumStep>=1?`${fmt(Number(pred.extraction?.input_iq_count)/10000,2)} ms`:'—';
    $('target-gt').textContent=state.spectrumStep===3?(pred.ground_truth?.class||'未匹配真实目标'):'评估阶段显示';
    $('target-top5-wrap').hidden=state.spectrumStep!==3;
    $('target-top5').innerHTML=(detail.fine_top5||[]).map(x=>`<div class="top5-row"><span class="top5-label" title="${escape(x.class)}">${escape(x.class)}</span><div class="bar-track"><div class="bar-fill" style="width:${Math.max(0,Math.min(100,Number(x.probability)*100))}%"></div></div><strong>${pct(x.probability,1)}</strong></div>`).join('');
    renderTargetTable();
    $('spectrum-matched-stats').textContent=`TP ${d.metrics.TP} · FP ${d.metrics.FP} · FN ${d.metrics.FN}`;
  }
  function renderTargetTable(){
    const d=state.spectrumData;if(!d)return;
    $('spectrum-target-tbody').innerHTML=d.predictions.map((p,i)=>{
      const f=p.with_rms||{}, gt=p.ground_truth;
      const match=gt?.family===f.family_top1; const fine=gt?.class===f.fine_top1;
      const famclass=gt?(match?'correct':'wrong'):'dim';
      const fineclass=gt?(fine?'correct':'wrong'):'dim';
      const fc=Number(p.extraction?.predicted_center_offset_hz)/1e6;
      const bw=Number(p.extraction?.predicted_bandwidth_hz)/1e6;
      return `<tr data-target="${i}" class="${i===state.targetIndex?'selected':''}" tabindex="0" aria-label="选择目标 ${i+1}"><td><strong>#${i+1}</strong></td><td>${pct(p.confidence,1)}</td><td>${fc>=0?'+':''}${fmt(fc,2)} MHz</td><td>${fmt(bw,2)} MHz</td><td><span class="${famclass}">${escape(f.family_top1||'—')}</span></td><td><span class="${fineclass}">${escape(f.fine_top1||'—')}</span></td><td>${escape(gt?.class||'未匹配')}</td></tr>`;
    }).join('');
    $('spectrum-target-tbody').querySelectorAll('[data-target]').forEach(tr=>{
      tr.addEventListener('click',()=>selectTarget(Number(tr.dataset.target),true));
      tr.addEventListener('keydown',e=>{if(e.key==='Enter'||e.key===' '){e.preventDefault();selectTarget(Number(tr.dataset.target),true)}});
    });
  }
  function selectTarget(index,reveal=false){
    if (!state.spectrumData||!state.spectrumData.predictions[index])return;
    state.targetIndex=index;if(reveal)state.spectrumStep=3;renderSpectrum();
  }
  function setSpectrumStep(n) {state.spectrumStep=Math.max(0,Math.min(3,n));renderSpectrum()}
  function renderSpectrumMetrics(){
    const s=state.spectrumSummary['v4-final'];const y=state.yoloSummary;
    if (y){$('metric-map50').textContent=pct(y.mAP50);$('metric-map95').textContent=pct(y.mAP50_95)}
    if(s){$('metric-family').textContent=pct(s.classification.with_rms.family_accuracy_on_matched);$('metric-fine').textContent=pct(s.classification.with_rms.fine_accuracy_on_matched)}
  }
  function renderInsights() {
    activateButtons('spectrum-insights-switch','data-insight',state.spectrumInsight);
    const c=$('spectrum-insights-content');const a=state.spectrumSummary;
    if(!a['v4-final']||!a['v3']){c.innerHTML='<div class="empty-state">正在加载模型评测数据…</div>';return}
    const final=a['v4-final'].classification.with_rms;
    if(state.spectrumInsight==='compare'){
      $('spectrum-insights-title').textContent='XCiT V3 / V4 识别性能对比';
      $('spectrum-insights-description').textContent='同一批 100 张宽带图 · 381 个匹配目标';
      const v3=a['v3'].classification.with_rms;
      const metrics=[['家族准确率','family_accuracy_on_matched'],['57 类 Top-1','fine_accuracy_on_matched'],['端到端细类召回','fine_end_to_end_recall_on_all_gt']];
      c.innerHTML='<div class="model-bars">'+metrics.map(([name,key])=>`<div class="model-bar-row"><span>${escape(name)}</span><div class="model-bar-pair"><div><span style="width:${v3[key]*100}%"></span></div><div><span style="width:${final[key]*100}%"></span></div></div><strong>${pct(final[key])}</strong></div>`).join('')+'</div><div class="model-legend"><span><i class="key-dot"></i>V3</span><span><i class="key-dot v4"></i>V4 最终</span></div>';
    }else if(state.spectrumInsight==='family'){
      $('spectrum-insights-title').textContent='XCiT V4 的各调制家族表现';
      $('spectrum-insights-description').textContent='仅统计 YOLO 成功匹配且完成分类的目标';
      const rows=Object.entries(final.family_by_gt).map(([name,x])=>({name,count:x.matched_classified,fam:x.family_correct/x.matched_classified,fine:x.fine_correct/x.matched_classified}));
      c.innerHTML='<div class="family-performance">'+rows.map(x=>`<div><span>${escape(x.name)}</span><strong>${pct(x.fam)}</strong><small>细分类 ${pct(x.fine)} · ${x.count} 个匹配目标</small></div>`).join('')+'</div>';
    }else{
      $('spectrum-insights-title').textContent='XCiT V4 微调训练过程';
      $('spectrum-insights-description').textContent='独立验证集 7,950 个目标 · 10 个 Epoch';
      const hist=state.trainingHistory?.epochs||[];
      const rows=hist.map(x=>({x:Number(x.epoch),a:Number(x.train_fine_accuracy)*100,b:Number(x.validation.fine_accuracy)*100}));
      c.innerHTML='<div class="training-chart">'+renderLineChart(rows,'%','Epoch',true)+'</div><div class="model-legend"><span><i class="key-dot"></i>训练细类准确率</span><span><i class="key-dot v4"></i>验证细类准确率</span></div>';
    }
  }
  function openLightbox(path){if(!path)return;const im=$('lightbox-image');im.src=asset(path);im.onerror=()=>{if(state.root!==DEMO_REMOTE)im.src=rootAsset(path,DEMO_REMOTE)};$('lightbox').hidden=false;document.body.style.overflow='hidden'}
  function closeLightbox(){$('lightbox').hidden=true;document.body.style.overflow=''}
  function attachEvents(){
    $('tab-mit').addEventListener('click',()=>setTask('mit'));
    $('tab-spectrum').addEventListener('click',()=>setTask('spectrum'));
    for(const id of ['mit-soi','mit-interference','mit-sinr'])$(id).addEventListener('change',updateMITSelection);
    $('mit-steps').querySelectorAll('[data-step]').forEach(b=>b.addEventListener('click',()=>{stopPlayback();setMITStep(Number(b.dataset.step))}));
    $('mit-auto').addEventListener('click',()=>togglePlayback('mit'));
    $('mit-prev').addEventListener('click',()=>{stopPlayback();setMITStep(state.mitStep-1)});
    $('mit-next').addEventListener('click',()=>{stopPlayback();setMITStep(state.mitStep+1)});
    $('mit-view-switch').querySelectorAll('[data-view]').forEach(b=>b.addEventListener('click',()=>{state.mitView=b.dataset.view;renderMIT()}));
    $('mit-chart-switch').querySelectorAll('[data-chart]').forEach(b=>b.addEventListener('click',()=>{state.mitChart=b.dataset.chart;renderMITCharts()}));
    $('mit-zoom').addEventListener('click',()=>openLightbox(state.mitCurrentImage));
    $('spectrum-sample').addEventListener('change',e=>selectSpectrumSample(Number(e.target.value)));
    $('spectrum-steps').querySelectorAll('[data-step]').forEach(b=>b.addEventListener('click',()=>{stopPlayback();setSpectrumStep(Number(b.dataset.step))}));
    $('spectrum-auto').addEventListener('click',()=>togglePlayback('spectrum'));
    $('spectrum-prev').addEventListener('click',()=>{stopPlayback();setSpectrumStep(state.spectrumStep-1)});
    $('spectrum-next').addEventListener('click',()=>{stopPlayback();setSpectrumStep(state.spectrumStep+1)});
    $('spectrum-image-mode').querySelectorAll('[data-mode]').forEach(b=>b.addEventListener('click',()=>{stopPlayback();setSpectrumStep(({raw:0,boxes:1,features:2})[b.dataset.mode])}));
    $('spectrum-zoom').addEventListener('click',()=>openLightbox(state.spectrumCurrentImage));
    $('spectrum-insights-switch').querySelectorAll('[data-insight]').forEach(b=>b.addEventListener('click',()=>{state.spectrumInsight=b.dataset.insight;renderInsights()}));
    $('lightbox-close').addEventListener('click',closeLightbox);
    $('lightbox').addEventListener('click',e=>{if(e.target===$('lightbox'))closeLightbox()});
    window.addEventListener('keydown',e=>{if(e.key==='Escape')closeLightbox()});
  }
  async function init() {
    attachEvents();
    const opts = [DEMO_LOCAL,DEMO_REMOTE];
    for (const root of opts) {
      try {
        const res=await fetch(rootAsset('manifest.json',root),{cache:'no-store'});
        if(!res.ok)throw new Error('HTTP '+res.status);
        const json=await res.json();
        if(!Array.isArray(json.mit_wavenet?.samples)||!Array.isArray(json.yolo_xcit_v4?.samples))throw new Error('索引结构不完整');
        state.root=root;state.manifest=json;cache.set('manifest.json',json);
        break;
      }catch(e){/* Fall back to public GitHub raw data if local results are absent. */}
    }
    if(!state.manifest){
      showError('无法加载 demo-results/manifest.json。请检查网络连接，或将 frontend/ 与 demo-results/ 放在同一个 RF-SepEdge 目录，再从 RF-SepEdge 根目录启动 python -m http.server 8080。');
      $('mit-image-placeholder').textContent='暂无实验数据';$('spectrum-image-placeholder').textContent='暂无实验数据';
      return;
    }
    $('data-status').textContent=state.root===DEMO_LOCAL?'本地实验数据已就绪':'GitHub 实验数据已就绪';
    document.querySelector('.status-line').classList.add('ready');
    updateMITSelection();
    const jobs=[
      ...INTERFERENCES.map(async i=>{try{state.mitRows[i]=await fetchCSV(`signal-separation/${i.toLowerCase()}/by_sinr.csv`)}catch(e){console.error('CSV',i,e)}}),
      ...['v3','v4-final','v4-hier-w05'].map(async v=>{try{state.spectrumSummary[v]=await fetchJSON(`spectrum-analysis/model-selection/${v}/summary.json`)}catch(e){console.error('Model',v,e)}}),
      (async()=>{try{state.trainingHistory=await fetchJSON('spectrum-analysis/training-history/v4-final/history.json')}catch(e){console.error('History',e)}})(),
      (async()=>{try{state.yoloSummary=await fetchJSON('yolo-detection/seed42-50/summary.json')}catch(e){console.error('YOLO',e)}})(),
    ];
    await selectSpectrumSample(60);
    Promise.allSettled(jobs).then(()=>{renderMITCharts();renderSpectrumMetrics();renderInsights()});
    setTask(location.hash==='#spectrum'?'spectrum':'mit');
  }
  init().catch(e=>showError('前端初始化失败：'+e.message));
})();
