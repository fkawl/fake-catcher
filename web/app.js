const $ = id => document.getElementById(id);
let selected = null;
let busy = false;
function clearResults() {
  for (const id of ['input-image', 'fft-image', 'heatmap-image']) { $(id).hidden = true; $(id).removeAttribute('src'); }
  $('score').textContent = '—'; $('meter-fill').style.width = '0%'; $('timing').textContent = '분석 시간 —';
  $('badge').textContent = 'READY TO EXPLORE';
}
function selectFile(file) {
  if (busy || !file) return;
  clearResults(); $('error').textContent = ''; selected = null; $('analyze').disabled = true;
  if (file.size > 10 * 1024 * 1024) { $('error').textContent = '10MB 이하의 이미지를 선택하세요.'; return; }
  if (!['image/jpeg', 'image/png', 'image/webp'].includes(file.type)) { $('error').textContent = 'JPG, PNG, WEBP만 지원합니다.'; return; }
  selected = file; $('filename').textContent = `${file.name} · ${(file.size / 1024).toFixed(0)} KB`;
  $('analyze').disabled = false; $('verdict').textContent = '분석할 준비가 됐어요';
  $('explanation').textContent = '분석하기를 누르면 이미지의 특징을 확인합니다.';
}
$('file').addEventListener('change', e => selectFile(e.target.files[0]));
$('dropzone').addEventListener('keydown', e => { if (['Enter', ' '].includes(e.key)) {e.preventDefault(); if (!busy) $('file').click();} });
for (const type of ['dragover', 'dragleave', 'drop']) $('dropzone').addEventListener(type, e => {
  e.preventDefault(); $('dropzone').classList.toggle('dragover', type === 'dragover');
  if (type === 'drop') selectFile(e.dataTransfer.files[0]);
});
fetch('/api/status').then(r => {if (!r.ok) throw Error(); return r.json();}).then(data => $('model-status').textContent = data.message).catch(() => $('model-status').textContent = '서버에 연결하지 못했습니다. 새로고침해 주세요.');
$('analyze').addEventListener('click', async () => {
  if (!selected || busy) return;
  busy = true; $('analyze').disabled = true; $('file').disabled = true; $('analyze').textContent = '분석 중…'; $('error').textContent = ''; clearResults();
  try {
    const response = await fetch('/api/analyze', {method: 'POST', headers: {'Content-Type': selected.type}, body: selected});
    const data = await response.json();
    if (!response.ok) throw Error(data.detail || '분석에 실패했습니다.');
    for (const [id, key] of [['input-image', 'input'], ['fft-image', 'spectrum'], ['heatmap-image', 'heatmap']]) {
      if (data[key]) { $(id).src = data[key]; $(id).hidden = false; }
    }
    $('verdict').textContent = data.ready ? data.label : '주파수 미리보기 완료';
    $('badge').textContent = data.ready ? (data.variant === 'rgb' ? 'RGB BASELINE' : 'RGB + FFT') : 'PREVIEW ONLY';
    $('explanation').textContent = data.ready ? `예측 클래스의 RGB 판단 근거입니다. 위조 위치의 정답을 의미하지 않습니다.${data.heatmap_active ? '' : ' 뚜렷한 활성 영역이 없습니다.'}` : '학습된 모델이 없어 진위 점수와 히트맵은 표시하지 않습니다. 현재는 입력 이미지와 FFT를 확인할 수 있습니다.';
    if (data.ready && data.dataset_note?.includes('CIFAKE')) $('explanation').textContent += ' 저해상도 CIFAKE로 예비 학습한 모델이며 최신 AI 이미지의 정확도는 검증되지 않았습니다.';
    if (data.ready) { $('score').textContent = (data.score * 100).toFixed(1) + '%'; $('meter-fill').style.width = (data.score * 100) + '%'; $('timing').textContent = `판정 + 히트맵 ${data.elapsed_ms} ms · CPU`; }
    else $('timing').textContent = `원본 ${data.original_size[0]} × ${data.original_size[1]} · 모델 입력 256 × 256`;
  } catch (error) { $('error').textContent = error.message; $('verdict').textContent = '분석을 완료하지 못했어요'; }
  finally { busy = false; $('analyze').disabled = false; $('file').disabled = false; $('analyze').textContent = '이미지 분석하기'; }
});
