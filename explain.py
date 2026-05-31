"""explain.py

SHAP 기반 모델 판단 근거 생성.

각 feature의 SHAP 값(예측값을 얼마나 올리거나 내렸는지)을 계산하고
영향도 상위 feature들을 한국어 문장으로 변환한다.

사용법:
    from explain import explain_forecast
    result = explain_forecast(features, bundle, horizon=1)

    result = {
        'q50': 567377,
        'direction': 'up',           # 현재가 대비 예측 방향
        'reasons': [
            '최근 가격이 오르는 추세입니다',
            '현재 가격이 과거 평균보다 낮아 상승 여지가 있습니다',
            '출발일이 21일 남아 구매 압력이 높아지고 있습니다',
        ]
    }
"""

from __future__ import annotations

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Feature → 한국어 문장 매핑
# ---------------------------------------------------------------------------
# 각 항목: (feature명, 양수 SHAP 문장, 음수 SHAP 문장, 값 포맷 함수)
# 포맷 함수: feature 값을 받아 문장에 삽입할 문자열 반환 (None이면 값 미사용)

def _days(v):   return f'{int(v)}일'
def _krw(v):    return f'₩{int(v):,}'
def _pct(v):    return f'{v*100:.0f}%'
def _month(v):
    months = {1:'1월',2:'2월',3:'3월',4:'4월',5:'5월',6:'6월',
              7:'7월',8:'8월',9:'9월',10:'10월',11:'11월',12:'12월'}
    return months.get(int(v), f'{int(v)}월')

FEATURE_TEMPLATES: list[tuple[str, str, str, object]] = [
    # (feature명, SHAP>0 문장, SHAP<0 문장, 값 포맷 함수)

    ('days_to_departure',
     '출발까지 {v} 남아 가격 상승 압력이 높아지고 있습니다',
     '출발까지 {v} 남아 있어 가격이 낮은 편입니다',
     _days),

    ('current_cheapest_price',
     '현재가({v})가 높아 이후에도 높게 유지될 가능성이 있습니다',
     '현재가({v})가 낮아 이후에도 낮게 유지될 가능성이 있습니다',
     _krw),

    ('lag_1_price',
     '직전 검색 대비 가격이 올랐습니다',
     '직전 검색 대비 가격이 내렸습니다',
     None),

    ('price_change_1',
     '최근 가격이 오르는 추세입니다',
     '최근 가격이 내리는 추세입니다',
     None),

    ('hist_recent_slope',
     '최근 며칠간 가격이 지속적으로 상승하고 있습니다',
     '최근 며칠간 가격이 지속적으로 하락하고 있습니다',
     None),

    ('curr_vs_hist_mean',
     '현재 가격이 과거 평균보다 높아 추가 상승 여지는 제한적입니다',
     '현재 가격이 과거 평균보다 낮아 상승 여지가 있습니다',
     None),

    ('curr_gap_to_typical_min',
     '현재 가격이 통상 최저가보다 높습니다',
     '현재 가격이 통상 최저가보다 낮은 수준입니다',
     None),

    ('curr_gap_to_typical_max',
     '현재 가격이 통상 최고가에 근접해 있습니다',
     '현재 가격이 통상 최고가보다 많이 낮습니다',
     None),

    ('cheapest_nonstop_price',
     '직항 최저가가 높아 전반적인 가격대가 높습니다',
     '직항 최저가가 낮아 전반적인 가격대가 낮습니다',
     None),

    ('nonstop_ratio',
     '직항 비율이 높아 가격대가 높습니다',
     '직항 비율이 낮아 경유편 위주로 가격이 낮습니다',
     None),

    ('rolling_std_3',
     '최근 가격 변동성이 커 불확실성이 높습니다',
     '최근 가격이 안정적으로 유지되고 있습니다',
     None),

    ('price_vs_rolling_mean_3',
     '현재 가격이 최근 평균보다 높습니다',
     '현재 가격이 최근 평균보다 낮습니다',
     None),

    ('is_peak_season',
     '성수기(7·8·12월)라 가격이 높은 편입니다',
     None,   # 비성수기는 굳이 언급 안 해도 됨
     None),

    ('is_holiday_near',
     '한국 공휴일 전후라 수요가 높아 가격이 오릅니다',
     None,
     None),

    ('is_long_haul',
     '장거리 노선이라 기본 가격대가 높습니다',
     None,
     None),

    ('oil_price_usd',
     '국제 유가가 높아 항공권 가격 상승 요인이 있습니다',
     '국제 유가가 낮아 항공권 가격 하락 요인이 있습니다',
     None),

    ('oil_change_7d',
     '최근 7일간 유가가 상승해 가격 상승 압력이 있습니다',
     '최근 7일간 유가가 하락해 가격 하락 압력이 있습니다',
     None),

    ('arr_fx_change_7d',
     '도착국 통화 강세로 원화 환산 가격이 오를 수 있습니다',
     '도착국 통화 약세로 원화 환산 가격이 내릴 수 있습니다',
     None),

    ('route_id_enc',
     '이 노선은 가격이 높게 형성되는 경향이 있습니다',
     '이 노선은 가격이 낮게 형성되는 경향이 있습니다',
     None),

    ('outbound_month',
     '{v} 출발은 가격이 높은 시기입니다',
     '{v} 출발은 가격이 낮은 시기입니다',
     _month),
]

# feature명 → (양수문장, 음수문장, 포맷함수) 빠른 조회용 dict
_TEMPLATE_MAP = {
    feat: (pos, neg, fmt)
    for feat, pos, neg, fmt in FEATURE_TEMPLATES
}


# ---------------------------------------------------------------------------
# SHAP 계산
# ---------------------------------------------------------------------------

def get_shap_values(
    model,
    X: np.ndarray,
    feat_cols: list[str],
) -> dict[str, float]:
    """
    LightGBM pred_contrib=True 로 SHAP 값 추출.
    반환: {feature명: shap_value}  (bias 항 제외)
    마지막 열이 bias(expected value)이므로 제거.
    """
    contrib = model.predict(X, pred_contrib=True)   # shape: (n_samples, n_features+1)
    shap    = contrib[0, :-1]                        # 샘플 1개, bias 제외
    return dict(zip(feat_cols, shap))


# ---------------------------------------------------------------------------
# 문장 생성
# ---------------------------------------------------------------------------

def _make_sentence(
    feat: str,
    shap_val: float,
    feat_value: float,
    min_shap: float = 500,   # 이 이하면 무시 (영향 미미)
) -> str | None:
    """SHAP 값과 feature 값을 받아 한국어 문장 반환. 영향 미미하면 None."""
    if abs(shap_val) < min_shap:
        return None

    template = _TEMPLATE_MAP.get(feat)
    if template is None:
        return None

    pos_tmpl, neg_tmpl, fmt_fn = template
    tmpl = pos_tmpl if shap_val > 0 else neg_tmpl
    if tmpl is None:
        return None

    if '{v}' in tmpl and fmt_fn is not None:
        return tmpl.replace('{v}', fmt_fn(feat_value))
    return tmpl


# ---------------------------------------------------------------------------
# 메인 함수
# ---------------------------------------------------------------------------

def explain_forecast(
    features: dict,
    bundle: dict,
    horizon: int = 1,
    top_n: int = 3,
    current_price: int | None = None,
) -> dict:
    """
    특정 horizon 모델의 예측 근거를 한국어 문장으로 반환.

    Parameters
    ----------
    features   : predict.py에 넘기는 것과 같은 feature dict
    bundle     : joblib.load('airmoment_model.joblib')
    horizon    : 1 | 3 | 7 | 14
    top_n      : 반환할 근거 문장 수 (기본 3개)
    current_price : 현재가 (없으면 features에서 추출)

    Returns
    -------
    {
        'horizon_days': 1,
        'q50': 567377,
        'direction': 'up' | 'down' | 'flat',
        'direction_amount': 57377,   # 현재가 대비 예측 중앙값 차이
        'reasons': ['...', '...', '...']
    }
    """
    from predict import NUMERIC_COLS   # 순환참조 방지용 lazy import

    model     = bundle['models'][horizon]
    feat_cols = bundle['feature_cols'][horizon]
    encoders  = bundle['encoders']

    # ── Feature 인코딩 (predict.py와 동일 로직) ───────────────────────────
    row = {col: features.get(col, np.nan) for col in feat_cols
           if not col.endswith('_enc')}
    for cat, mapping in encoders.items():
        enc_col = f'{cat}_enc'
        if enc_col not in feat_cols:
            continue
        raw = str(features.get(cat, '__missing__') or '__missing__')
        row[enc_col] = float(mapping.get(raw, mapping.get('__missing__', 0)))

    df_row = pd.DataFrame([row])[feat_cols]
    X      = df_row.astype(float).to_numpy()

    # ── SHAP 계산 ─────────────────────────────────────────────────────────
    shap_map = get_shap_values(model, X, feat_cols)

    # ── q50 예측값 ────────────────────────────────────────────────────────
    q50 = int(round(float(model.predict(X)[0])))
    cur = current_price or int(features.get('current_cheapest_price', q50))
    diff = q50 - cur
    if   diff >  cur * 0.01:  direction = 'up'
    elif diff < -cur * 0.01:  direction = 'down'
    else:                     direction = 'flat'

    # ── SHAP 절댓값 기준 정렬 → 상위 feature 문장 생성 ───────────────────
    ranked = sorted(shap_map.items(), key=lambda x: abs(x[1]), reverse=True)

    reasons: list[str] = []
    for feat, shap_val in ranked:
        if len(reasons) >= top_n:
            break
        feat_value = float(df_row[feat].iloc[0]) if feat in df_row.columns else np.nan
        sentence   = _make_sentence(feat, shap_val, feat_value)
        if sentence:
            reasons.append(sentence)

    return {
        'horizon_days':     horizon,
        'q50':              q50,
        'direction':        direction,
        'direction_amount': diff,
        'reasons':          reasons,
    }


def explain_all_horizons(
    features: dict,
    bundle: dict,
    top_n: int = 3,
) -> list[dict]:
    """모든 horizon에 대해 explain_forecast 실행."""
    cur = int(features.get('current_cheapest_price', 0))
    return [
        explain_forecast(features, bundle, horizon=h, top_n=top_n, current_price=cur)
        for h in [1, 3, 7, 14]
    ]


# ---------------------------------------------------------------------------
# 스모크 테스트
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    import json, joblib

    MODEL_PATH = 'data_0526/models/airmoment_model.joblib'
    bundle     = joblib.load(MODEL_PATH)

    features = {
        'route_id':                   'ICN-CDG',
        'days_to_departure':          21,
        'searched_day_of_week':       'MON',
        'outbound_month':             7,
        'outbound_day_of_week':       'FRI',
        'is_weekend_search':          0,
        'is_peak_season':             1,
        'is_holiday_near':            0,
        'is_long_haul':               1,
        'offer_count':                8,
        'nonstop_ratio':              0.25,
        'cheapest_nonstop_price':     890_000,
        'cheapest_offer_has_layover': 1,
        'current_cheapest_price':     510_000,
        'curr_gap_to_typical_min':    75_000,
        'curr_gap_to_typical_max':   -230_000,
        'price_level':               'typical',
        'hist_recent_std':            8_500,
        'hist_recent_slope':         -1_200,
        'curr_vs_hist_mean':          0.97,
        'lag_1_price':                522_000,
        'price_change_1':            -12_000,
        'rolling_std_3':              9_800,
        'price_vs_rolling_mean_3':   -6_000,
        'oil_price_usd':              82.5,
        'oil_change_7d':              1.2,
        'arr_fx_change_7d':          -0.3,
    }

    results = explain_all_horizons(features, bundle, top_n=3)

    cur = features['current_cheapest_price']
    print(f'\n현재가: ₩{cur:,}\n')
    for r in results:
        arrow = '↑' if r['direction'] == 'up' else ('↓' if r['direction'] == 'down' else '→')
        print(f'+{r["horizon_days"]:>2d}일  {arrow}  q50=₩{r["q50"]:,}  '
              f'(현재가 대비 {r["direction_amount"]:+,}원)')
        for reason in r['reasons']:
            print(f'   • {reason}')
        print()
