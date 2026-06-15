"""explain.py

SHAP 기반 모델 판단 근거 생성.

각 feature의 SHAP 값(예측값을 얼마나 올리거나 내렸는지)을 계산하고
영향도 상위 feature들을 한국어 문장으로 변환한다.

사용법:
    from explain import explain_forecast
    from inference import load_model, predict_flight_decision
    from predict import load_forecaster

    model      = load_model()
    forecaster = load_forecaster('airmoment_forecast.joblib')
    decision   = predict_flight_decision(features, model, forecaster=forecaster)

    result = explain_forecast(
        features,
        clf=model['clf'],
        forecaster=forecaster,
        is_wait=(decision['decision'] == 'WAIT'),
        drop_amount=decision['predicted_drop_amount'],
        top_n=3,
    )

    result = {
        'direction':        'down',      # 'down' | 'up'
        'direction_amount': 45000,       # 절감/추가부담 예상액 (KRW)
        'reasons': [
            '현재 가격이 과거 평균보다 높아 하락 여지가 있습니다',
            '최근 가격이 올라 현재가가 높은 상태입니다',
            '국제 유가가 높아 현재 항공권 가격이 비싼 편입니다',
        ]
    }
"""

from __future__ import annotations

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Feature → 한국어 문장 매핑
# ---------------------------------------------------------------------------
# 각 항목: (feature명, SHAP>0 문장, SHAP<0 문장, 값 포맷 함수, 방향 검증 여부)
# 포맷 함수  : feature 값을 받아 문장에 삽입할 문자열 반환 (None이면 값 미사용)
# 방향 검증 : True면 feature 값의 부호와 SHAP 값의 부호가 일치할 때만 문장을 사용

def _days(v):   return f'{int(v)}일'
def _krw(v):    return f'₩{int(v):,}'
def _pct(v):    return f'{v*100:.0f}%'
def _month(v):
    months = {1:'1월',2:'2월',3:'3월',4:'4월',5:'5월',6:'6월',
              7:'7월',8:'8월',9:'9월',10:'10월',11:'11월',12:'12월'}
    return months.get(int(v), f'{int(v)}월')

FEATURE_TEMPLATES: list[tuple] = [
    # (feature명, SHAP>0 문장, SHAP<0 문장, 값 포맷 함수, 방향 검증)

    ('days_to_departure',
     '출발까지 {v} 남아 가격 상승 압력이 높아지고 있습니다',
     '출발까지 {v} 남아 있어 가격이 낮은 편입니다',
     _days, False),

    ('current_cheapest_price',
     '현재가({v})가 높아 이후에도 높게 유지될 가능성이 있습니다',
     '현재가({v})가 낮아 이후에도 낮게 유지될 가능성이 있습니다',
     _krw, False),

    ('lag_1_price',
     '직전 검색 대비 가격이 올랐습니다',
     '직전 검색 대비 가격이 내렸습니다',
     None, False),

    ('price_change_1',
     '최근 가격이 올라 현재가가 높은 상태입니다',
     '최근 가격이 하락해 현재가가 저렴한 상태입니다',
     None, True),

    ('hist_recent_slope',
     '최근 며칠간 가격이 지속적으로 상승하고 있습니다',
     '최근 며칠간 가격이 지속적으로 하락하고 있습니다',
     None, True),

    ('curr_vs_hist_mean',
     '현재 가격이 과거 평균보다 높아 하락 여지가 있습니다',
     '현재 가격이 과거 평균보다 낮은 수준입니다',
     None, False),

    ('curr_gap_to_typical_min',
     '현재 가격이 통상 최저가보다 높습니다',
     '현재 가격이 통상 최저가보다 낮은 수준입니다',
     None, True),

    ('curr_gap_to_typical_max',
     '현재 가격이 통상 최고가에 근접해 있습니다',
     '현재 가격이 통상 최고가보다 많이 낮습니다',
     None, True),

    ('cheapest_nonstop_price',
     '직항 최저가가 높아 전반적인 가격대가 높습니다',
     '직항 최저가가 낮아 전반적인 가격대가 낮습니다',
     None, False),

    ('nonstop_ratio',
     '직항 비율이 높아 가격대가 높습니다',
     '직항 비율이 낮아 경유편 위주로 가격이 낮습니다',
     None, False),

    ('rolling_std_3',
     '최근 가격 변동이 심해 조금 더 지켜보는 것이 유리합니다',
     '최근 가격이 안정적으로 유지되고 있습니다',
     None, False),

    ('price_vs_rolling_mean_3',
     '현재 가격이 최근 평균보다 높습니다',
     '현재 가격이 최근 평균보다 낮습니다',
     None, True),

    ('is_peak_season',
     '성수기(7·8·12월)라 가격이 높은 편입니다',
     None,   # 비성수기는 굳이 언급 안 해도 됨
     None, True),

    ('is_holiday_near',
     '한국 공휴일 전후라 수요가 높아 가격이 오릅니다',
     None,
     None, True),

    ('is_long_haul',
     '장거리 노선이라 기본 가격대가 높습니다',
     None,
     None, False),

    ('oil_price_usd',
     '국제 유가가 높아 현재 항공권 가격이 비싼 편입니다',
     '국제 유가가 낮아 항공권 가격 하락 요인이 있습니다',
     None, False),

    ('oil_change_7d',
     '최근 유가 상승으로 현재 항공권 가격이 높아진 상태입니다',
     '최근 7일간 유가가 하락해 가격 하락 압력이 있습니다',
     None, True),

    ('arr_fx_change_7d',
     '도착국 통화 강세로 원화 환산 가격이 오를 수 있습니다',
     '도착국 통화 약세로 원화 환산 가격이 내릴 수 있습니다',
     None, True),

    ('route_id_enc',
     '이 노선은 가격이 높게 형성되는 경향이 있습니다',
     '이 노선은 가격이 낮게 형성되는 경향이 있습니다',
     None, False),

    ('outbound_month',
     '{v} 출발은 가격이 높은 시기입니다',
     '{v} 출발은 가격이 낮은 시기입니다',
     _month, False),
]

# feature명 → (양수문장, 음수문장, 포맷함수, 방향검증) 빠른 조회용 dict
_TEMPLATE_MAP = {
    feat: (pos, neg, fmt, validate)
    for feat, pos, neg, fmt, validate in FEATURE_TEMPLATES
}

# LightGBM 전용 override (D≤30, 가격 자체에 대한 SHAP)
# pos: shap>0 (가격 상승 기여 → BUY 이유), neg: shap<=0 (가격 하락 기여 → WAIT 이유)
# None이면 _TEMPLATE_MAP의 기본 문장을 그대로 사용
_LGB_OVERRIDES: dict[str, tuple[str | None, str | None]] = {
    'rolling_std_3': (
        '최근 가격 변동이 커 더 오르기 전에 구매하는 것이 유리할 수 있습니다',
        None,
    ),
    'curr_vs_hist_mean': (
        '현재 가격이 과거 평균보다 높아 더 오르기 전에 구매가 유리합니다',
        None,
    ),
}


# ---------------------------------------------------------------------------
# SHAP 계산
# ---------------------------------------------------------------------------

# forecast['q10'] 인덱스(horizon_index) → 실제 horizon 일수
_INDEX_TO_HORIZON = {1: 1, 2: 3, 3: 7, 4: 14}


def get_catboost_shap(clf, features: dict) -> dict[str, float]:
    """
    CatBoost ShapValues로 BUY/WAIT 분류 결정에 대한 feature 기여도 추출.
    반환: {feature명: shap_value}  (bias 항 제외, log-odds 단위)
    """
    from catboost import Pool

    input_df = pd.DataFrame([features])
    for col in clf.num_cols_ + clf.cat_cols:
        if col not in input_df.columns:
            input_df[col] = np.nan

    X_prep      = clf._prepare(input_df)
    cat_indices = [X_prep.columns.tolist().index(c)
                   for c in clf.cat_cols if c in X_prep.columns]
    shap_values = clf.model_.get_feature_importance(
        Pool(X_prep, cat_features=cat_indices),
        type='ShapValues',
    )
    shap = shap_values[0, :-1]   # 마지막 열 = bias
    return dict(zip(X_prep.columns.tolist(), shap))


def get_lightgbm_shap(forecaster, features: dict, days_to_departure: int) -> dict[str, float]:
    """
    LightGBM pred_contrib=True 로 가격 예측에 대한 feature 기여도 추출.
    days_to_departure를 /predict와 동일한 horizon_index로 매핑해 모델 선택.
    반환: {feature명: shap_value}  (bias 항 제외, KRW 단위)
    """
    from inference import horizon_index

    horizon   = _INDEX_TO_HORIZON[horizon_index(days_to_departure)]
    model     = forecaster.models[horizon]
    feat_cols = forecaster.feature_cols[horizon]
    X         = forecaster._encode(features, feat_cols)
    contrib   = model.predict(X, pred_contrib=True)
    shap      = contrib[0, :-1]
    return dict(zip(feat_cols, shap))


# ---------------------------------------------------------------------------
# 문장 생성
# ---------------------------------------------------------------------------

def _make_sentence(
    feat: str,
    shap_val: float,
    feat_value: float,
    min_shap: float = 0.01,
    use_lgb: bool = False,
) -> str | None:
    """SHAP 값과 feature 값을 받아 한국어 문장 반환. 영향 미미하면 None."""
    if abs(shap_val) < min_shap:
        return None

    template = _TEMPLATE_MAP.get(feat)
    if template is None:
        return None

    pos_tmpl, neg_tmpl, fmt_fn, validate = template

    if use_lgb and feat in _LGB_OVERRIDES:
        lgb_pos, lgb_neg = _LGB_OVERRIDES[feat]
        if shap_val > 0 and lgb_pos is not None:
            pos_tmpl = lgb_pos
        elif shap_val <= 0 and lgb_neg is not None:
            neg_tmpl = lgb_neg

    # 방향 검증: feature 값의 부호와 SHAP 값의 부호가 다르면 문장이 의미상 맞지 않으므로 스킵
    if validate and not np.isnan(feat_value):
        if (feat_value > 0) != (shap_val > 0):
            return None

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
    clf,
    forecaster,
    is_wait: bool,
    drop_amount: float,
    top_n: int = 3,
) -> dict:
    """
    days_to_departure 기준으로 SHAP 출처를 분기해 BUY/WAIT 판단 근거를 반환한다.

      D > 30 : CatBoost 분류기가 1차 조건 → CatBoost SHAP  (log-odds 단위)
      D ≤ 30 : conformal(LightGBM)이 1차 조건 → LightGBM SHAP (KRW 단위)

    Parameters
    ----------
    features    : feature dict (predict.py / inference.py에 넘기는 것과 동일)
    clf         : NativeCatBoostClassifier (model['clf'])
    forecaster  : ConformalForecaster
    is_wait     : 실제 결정이 WAIT이면 True (predict_flight_decision 결과 기준)
    drop_amount : 절감/추가부담 예상액 (KRW, predicted_drop_amount)
    top_n       : 반환할 근거 문장 수

    Returns
    -------
    {
        'direction':        'down' | 'up',
        'direction_amount': 45000,
        'reasons':          ['...', '...', '...']
    }
    """
    days_raw = features.get('days_to_departure')
    days = int(days_raw) if days_raw is not None else 999

    if days > 30:
        shap_map = get_catboost_shap(clf, features)
        min_shap = 0.01   # log-odds 단위
    else:
        shap_map = get_lightgbm_shap(forecaster, features, days)
        min_shap = 0.005  # log-ratio 단위 (≈0.5% 가격 기여 이상만 표시)

    # WAIT → 가격 하락 예상 → direction='down', BUY → direction='up'
    direction = 'down' if is_wait else 'up'

    # direction과 일치하는 부호의 SHAP만 사용해 이유와 결정이 항상 align되게 함.
    #   CatBoost (D>30): shap>0 = WAIT 쪽 기여
    #   LightGBM (D≤30): shap>0 = 가격 상승 기여(BUY 이유)
    if days > 30:
        want_positive = (direction == 'down')
    else:
        want_positive = (direction == 'up')

    # D≤30(LightGBM)에서 days_to_departure는 가격 예측 기여 설명으로 부적절해 제외
    exclude = {'days_to_departure'} if days <= 30 else set()

    ranked = sorted(
        [(f, s) for f, s in shap_map.items()
         if (s > 0) == want_positive and f not in exclude],
        key=lambda x: abs(x[1]),
        reverse=True,
    )

    reasons: list[str] = []
    for feat, shap_val in ranked:
        if len(reasons) >= top_n:
            break
        feat_value = pd.to_numeric(features.get(feat, np.nan), errors='coerce')
        sentence   = _make_sentence(feat, shap_val, feat_value, min_shap, use_lgb=(days <= 30))
        if sentence:
            reasons.append(sentence)

    return {
        'direction':        direction,
        'direction_amount': int(round(drop_amount)),
        'reasons':          reasons,
    }


# ---------------------------------------------------------------------------
# 스모크 테스트
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    from inference import load_model, predict_flight_decision
    from predict import load_forecaster

    model      = load_model()
    forecaster = load_forecaster('airmoment_forecast.joblib')

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

    decision = predict_flight_decision(features, model, forecaster=forecaster)
    result = explain_forecast(
        features,
        clf=model['clf'],
        forecaster=forecaster,
        is_wait=(decision['decision'] == 'WAIT'),
        drop_amount=decision['predicted_drop_amount'],
        top_n=3,
    )

    cur   = features['current_cheapest_price']
    arrow = '↓' if result['direction'] == 'down' else '↑'
    label = '절감 예상' if result['direction'] == 'down' else '추가 부담 예상'
    print(f'\n현재가: ₩{cur:,}')
    print(f'{arrow} {label}: ₩{result["direction_amount"]:,}\n')
    for reason in result['reasons']:
        print(f'   • {reason}')
