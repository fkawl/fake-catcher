import html
import json
from pathlib import Path


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--runs', type=Path, default=Path('runs/cifake_pilot'))
    parser.add_argument('--epochs', type=int, default=5)
    args = parser.parse_args()
    results, table, curves = {}, [], []
    for variant in ['rgb', 'rgb_fft']:
        folder = args.runs / variant
        result = json.loads((folder / f'evaluation_epoch_{args.epochs}' / 'test.json').read_text())
        history = json.loads((folder / 'history.json').read_text())
        results[variant] = {'test': result, 'history': history}
        cm = result['confusion_matrix_real_fake']
        table.append(f'<tr><th>{variant}</th><td>{result["accuracy"]:.2%}</td><td>{result["f1"]:.4f}</td><td>{result["auroc"]:.4f}</td><td>{cm[0][1]} / {sum(cm[0])}</td><td>{cm[1][0]} / {sum(cm[1])}</td></tr>')
        points = ' '.join(f'{40+i*500/max(len(history)-1,1)},{220-r["auroc"]*180}' for i, r in enumerate(history))
        color = '#007f72' if variant == 'rgb_fft' else '#6249b2'
        curves.append(f'<polyline fill="none" stroke="{color}" stroke-width="3" points="{points}"/>')
    page = '''<!doctype html><html lang="ko"><meta charset="utf-8"><title>Fake Catcher 예비 학습 보고서</title><style>body{font:17px/1.7 system-ui,sans-serif;max-width:1000px;margin:50px auto;padding:24px;color:#172738;background:#f3f7fa}h1{font-size:32px}table{width:100%;border-collapse:collapse;background:white}th,td{padding:14px;border-bottom:1px solid #ddd;text-align:left}.note{padding:20px;background:#fff3d4;border-radius:10px}svg{background:white;max-width:100%}</style><h1>Fake Catcher 예비 학습 결과</h1><p>CIFAKE 부분집합 · 학습 8,000장 / 검증 1,000장 / 테스트 2,000장 · real/fake 균형 · seed 42</p><p class="note">32×32 원본을 256×256으로 확대해 학습했습니다. 독립 테스트 표본이지만 생성기는 학습과 같은 SD 1.4입니다. 최신 생성기·고해상도·SNS 압축 환경의 성능은 검증하지 않았습니다. 이 수치를 일반적인 AI 이미지 탐지 정확도로 사용하지 마세요.</p><h2>분리된 테스트셋</h2><table><tr><th>모델</th><th>정확도</th><th>F1 (fake)</th><th>AUROC</th><th>실제→가짜 오탐</th><th>가짜→실제 누락</th></tr>'''
    page += ''.join(table) + '</table><h2>검증 AUROC 변화</h2><p>보라: RGB / 초록: RGB+FFT · 세로축 0–1, 가로축 epoch</p><svg viewBox="0 0 600 260" role="img" aria-label="모델별 검증 AUROC 학습 곡선"><path d="M40 40V220H560" fill="none" stroke="#444"/><text x="5" y="45">1.0</text><text x="5" y="220">0.0</text>' + ''.join(curves)
    page += ''.join(f'<text x="{40+i*500/max(args.epochs-1,1)}" y="245">{i+1}</text>' for i in range(args.epochs)) + '</svg><p>최고 검증 AUROC의 체크포인트를 평가했습니다. 분류 기준 0.5, 점수 보정 없음. 테스트셋은 학습 및 모델 선택에 사용하지 않았습니다. 각 구성 1회 실험이므로 FFT 효과의 확정적 결론은 낼 수 없습니다.</p><p>원자료 및 라이선스: <a href="https://github.com/jordan-bird/CIFAKE-Real-and-AI-Generated-Synthetic-Images">CIFAKE 저자 저장소</a>. Bird &amp; Lotfi (2024), Krizhevsky &amp; Hinton (2009).</p></html>'
    target = args.runs / 'report.html'
    target.write_text(page, encoding='utf-8')
    (args.runs / 'comparison.json').write_text(json.dumps(results, indent=2), encoding='utf-8')
    print(target)


if __name__ == '__main__':
    main()
