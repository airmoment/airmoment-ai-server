"""validate_explain.py

explain.py 검증 스크립트 — 네 가지 방법:
  1. 극단 케이스 단위 테스트
  2. SHAP 부호 vs 피처 값 부호 불일치율
  3. reasons 커버리지 (방향 검증 전후 비교)
  4. LLM 일관성 평가 (Claude API, 점수 1–5)
"""

import sys
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from predict import ConformalForecaster
from explain import (
    get_catboost_shap, get_lightgbm_shap,
    explain_forecast, _make_sentence, _TEMPLATE_MAP,
    FEATURE_TEMPLATES,
)
from inference import load_model, _horizon_index

# ---------------------------------------------------------------------------
# 모델 로드
# ---------------------------------------------------------------------------

print('Loading models...')
forecaster = ConformalForecaster(PROJECT_ROOT / 'airmoment_forecast.joblib')
model      = load_model(
    clf_path=PROJECT_ROOT / 'catboost_classifier_best.joblib',
    reg_path=PROJECT_ROOT / 'xgb_regressor_best.joblib',
)
clf = model['clf']
print('Done.\n')


# ---------------------------------------------------------------------------
# 공통 base feature
# ---------------------------------------------------------------------------

BASE = {
    'route_id': 'ICN-CDG', 'days_to_departure': 21,
    'searched_day_of_week': 'MON', 'outbound_month': 7,
    'outbound_day_of_week': 'FRI', 'is_weekend_search': 0,
    'is_peak_season': 1, 'is_holiday_near': 0, 'is_long_haul': 1,
    'offer_count': 8, 'nonstop_ratio': 0.25,
    'cheapest_nonstop_price': 890_000, 'cheapest_offer_has_layover': 1,
    'current_cheapest_price': 510_000,
    'curr_gap_to_typical_min': 75_000, 'curr_gap_to_typical_max': -230_000,
    'price_level': 'typical', 'hist_recent_std': 8_500,
    'hist_recent_slope': -1_200, 'curr_vs_hist_mean': 0.97,
    'lag_1_price': 522_000, 'price_change_1': -12_000,
    'rolling_std_3': 9_800, 'price_vs_rolling_mean_3': -6_000,
    'oil_price_usd': 78.5, 'oil_change_7d': -1.2, 'arr_fx_change_7d': 0.003,
}


# ===========================================================================
# 1. 극단 케이스 단위 테스트
# ===========================================================================

print('=' * 60)
print('1. 극단 케이스 단위 테스트')
print('=' * 60)

CASES = {
    'WAIT (유가↑ + 가격↑ + 성수기)': {
        **BASE,
        'oil_change_7d': 8.0,
        'price_change_1': 40_000,
        'hist_recent_slope': 3_000,
        'is_peak_season': 1,
        'days_to_departure': 45,   # D>30 → CatBoost SHAP
    },
    'BUY (유가↓ + 가격↓ + 비성수기)': {
        **BASE,
        'oil_change_7d': -8.0,
        'price_change_1': -40_000,
        'hist_recent_slope': -3_000,
        'is_peak_season': 0,
        'days_to_departure': 45,
    },
    'BUY (출발 임박 + 가격↓)': {
        **BASE,
        'days_to_departure': 3,    # D≤30 → LightGBM SHAP
        'price_change_1': -20_000,
        'hist_recent_slope': -1_500,
    },
    'WAIT (출발 여유 + 유가↑)': {
        **BASE,
        'days_to_departure': 60,
        'oil_change_7d': 10.0,
        'price_change_1': 25_000,
    },
}

for name, features in CASES.items():
    days = features['days_to_departure']
    try:
        wait_prob = float(clf.predict_proba(pd.DataFrame([features])
                          .reindex(columns=clf.num_cols_ + clf.cat_cols)
                          .fillna('__missing__' if False else np.nan))[:, 1][0])
    except Exception:
        wait_prob = float(clf.predict_proba(pd.DataFrame([features]))[:, 1][0])

    exp = explain_forecast(
        features, clf=clf, forecaster=forecaster,
        is_wait=(wait_prob >= 0.5), drop_amount=0, top_n=3,
    )

    print(f'\n[{name}]')
    print(f'  wait_prob={wait_prob:.3f}  direction={exp["direction"]}  days={days}  shap_from={"CatBoost" if days > 30 else "LightGBM"}')
    if exp['reasons']:
        for r in exp['reasons']:
            print(f'  • {r}')
    else:
        print('  • (reasons 없음)')


# ===========================================================================
# 2. SHAP 부호 vs 피처 값 부호 불일치율
# ===========================================================================

print('\n\n' + '=' * 60)
print('2. SHAP 부호 vs 피처 값 부호 불일치율')
print('=' * 60)

VALIDATE_FEATS = [
    feat for feat, *_, validate in FEATURE_TEMPLATES if validate
]

# 샘플 데이터 로드 (최대 500행)
DATA_PATH = PROJECT_ROOT / 'data_0603' / 'processed' / 'quantile_dataset_v2.csv'
if not DATA_PATH.exists():
    DATA_PATH = PROJECT_ROOT / 'data_0526' / 'processed' / 'quantile_dataset_v2.csv'

df_sample = pd.read_csv(DATA_PATH).dropna(subset=['current_cheapest_price']).head(500)
print(f'샘플 {len(df_sample)}행 사용\n')

mismatch_stats = []
for feat in VALIDATE_FEATS:
    if feat not in df_sample.columns:
        continue

    feat_vals  = df_sample[feat].dropna()
    valid_idx  = feat_vals.index

    # SHAP 계산 (days_to_departure 기준으로 source 분기)
    shap_vals = []
    for idx in valid_idx:
        row  = df_sample.loc[idx].to_dict()
        days = int(row.get('days_to_departure', 999))
        try:
            if days > 30:
                sm = get_catboost_shap(clf, row)
            else:
                sm = get_lightgbm_shap(forecaster, row, days)
            shap_vals.append(sm.get(feat, np.nan))
        except Exception:
            shap_vals.append(np.nan)

    shap_arr = np.array(shap_vals)
    feat_arr = feat_vals.to_numpy(float)

    valid    = ~np.isnan(shap_arr) & ~np.isnan(feat_arr)
    n        = valid.sum()
    if n == 0:
        continue

    mismatch = ((feat_arr[valid] > 0) != (shap_arr[valid] > 0)).sum()
    pct      = mismatch / n * 100
    mismatch_stats.append((feat, n, mismatch, pct))

print(f'  {"피처":<28} {"샘플":>6} {"불일치":>6} {"불일치율":>8}')
print(f'  {"-"*28} {"-"*6} {"-"*6} {"-"*8}')
for feat, n, mis, pct in sorted(mismatch_stats, key=lambda x: -x[3]):
    flag = ' ⚠' if pct > 30 else ''
    print(f'  {feat:<28} {n:>6} {mis:>6} {pct:>7.1f}%{flag}')


# ===========================================================================
# 3. reasons 커버리지 (방향 검증 전후 비교)
# ===========================================================================

print('\n\n' + '=' * 60)
print('3. reasons 커버리지 (방향 검증 전후 비교)')
print('=' * 60)

from explain import _make_sentence as make_with_validate

def make_without_validate(feat, shap_val, feat_value, min_shap):
    """방향 검증 없이 SHAP 부호만으로 문장 생성."""
    if abs(shap_val) < min_shap:
        return None
    template = _TEMPLATE_MAP.get(feat)
    if template is None:
        return None
    pos_tmpl, neg_tmpl, fmt_fn, _ = template
    tmpl = pos_tmpl if shap_val > 0 else neg_tmpl
    if tmpl is None:
        return None
    if '{v}' in tmpl and fmt_fn is not None:
        return tmpl.replace('{v}', fmt_fn(feat_value))
    return tmpl

n_before, n_after = [], []
n_zero_before, n_zero_after = 0, 0

sample_rows = df_sample.head(200)
for _, row in sample_rows.iterrows():
    feat_dict = row.to_dict()
    days      = int(feat_dict.get('days_to_departure', 999))
    min_shap  = 0.01 if days > 30 else 500

    try:
        if days > 30:
            shap_map = get_catboost_shap(clf, feat_dict)
        else:
            shap_map = get_lightgbm_shap(forecaster, feat_dict, days)
    except Exception:
        continue

    ranked = sorted(shap_map.items(), key=lambda x: abs(x[1]), reverse=True)

    cnt_before, cnt_after = 0, 0
    for feat, shap_val in ranked:
        raw = feat_dict.get(feat, np.nan)
        try:
            fv = float(raw)
        except (TypeError, ValueError):
            fv = np.nan
        if make_without_validate(feat, shap_val, fv, min_shap):
            cnt_before += 1
        if make_with_validate(feat, shap_val, fv, min_shap):
            cnt_after += 1

    n_before.append(min(cnt_before, 3))
    n_after.append(min(cnt_after, 3))
    if cnt_before == 0: n_zero_before += 1
    if cnt_after  == 0: n_zero_after  += 1

n = len(n_before)
print(f'\n  샘플 {n}행 기준')
print(f'  {"":25} {"검증 전":>8} {"검증 후":>8}')
print(f'  {"-"*25} {"-"*8} {"-"*8}')
print(f'  {"평균 reasons 수":25} {np.mean(n_before):>8.2f} {np.mean(n_after):>8.2f}')
print(f'  {"reasons=0인 비율":25} {n_zero_before/n*100:>7.1f}% {n_zero_after/n*100:>7.1f}%')
print(f'  {"reasons>=3인 비율":25} {sum(x>=3 for x in n_before)/n*100:>7.1f}% {sum(x>=3 for x in n_after)/n*100:>7.1f}%')


# ===========================================================================
# 4. LLM 일관성 평가 — 배치 프롬프트 출력 (ChatGPT/Claude에 붙여넣기용)
# ===========================================================================

print('\n\n' + '=' * 60)
print('4. LLM 일관성 평가 — 배치 프롬프트')
print('=' * 60)

N_EVAL = 50
cases: list[dict] = []

for _, row in df_sample.head(N_EVAL).iterrows():
    feat_dict = row.to_dict()
    try:
        wait_prob = float(clf.predict_proba(pd.DataFrame([feat_dict]))[:, 1][0])
    except Exception:
        continue
    decision = 'WAIT' if wait_prob >= 0.5 else 'BUY'
    exp = explain_forecast(
        feat_dict, clf=clf, forecaster=forecaster,
        is_wait=(wait_prob >= 0.5), drop_amount=0, top_n=3,
    )
    cases.append({'decision': decision, 'reasons': exp['reasons']})

lines = [
    '아래는 항공권 가격 예측 서비스의 출력 결과입니다.',
    '각 케이스에 대해 사용자 입장에서 결정과 이유가 얼마나 일관성 있고 납득 가능한지 1–5점으로 평가하세요.',
    '',
    '5점: 결정과 이유가 완벽히 일관되고 납득 가능',
    '3점: 이유 일부만 결정과 연결됨',
    '1점: 이유가 결정과 모순되거나 이해 불가',
    '',
    '반드시 아래 형식으로만 답하세요. 번호 순서대로, 한 줄에 하나씩:',
    '케이스1: [점수] [한 줄 설명]',
    '케이스2: [점수] [한 줄 설명]',
    '...',
    '',
    '---',
]

for i, c in enumerate(cases, 1):
    label = '지금 구매하세요 (BUY)' if c['decision'] == 'BUY' else '기다리세요 (WAIT)'
    body  = '\n'.join(f'  - {r}' for r in c['reasons']) if c['reasons'] else '  - (이유 없음)'
    lines.append(f'케이스{i}:')
    lines.append(f'  결정: {label}')
    lines.append(f'  이유:')
    lines.append(body)
    lines.append('')

batch_prompt = '\n'.join(lines)

print(f'\n  {len(cases)}개 케이스 생성 완료.')
print('  아래 전체를 복사해서 ChatGPT / Claude 창에 붙여넣으세요.\n')
print('─' * 60)
print(batch_prompt)
print('─' * 60)
