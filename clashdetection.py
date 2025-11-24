# clashdetection.py
# Streamlit 기반 Clash 우선순위 + Gemini 결과보고서 + 챗봇

import pandas as pd
import streamlit as st
import google.generativeai as genai

st.set_page_config(
    page_title="AI Clash Agent (CI Ranking)",
    page_icon="🧱",
    layout="wide"
)

st.title("🧱 AI Clash Agent (CI Ranking + Gemini Report)")

st.markdown(
    """
업로드한 Clash CSV/XLSX를 기반으로 **간섭 중요도(CI)**를 계산하고,
- 우선 수정해야 할 간섭 순위(Rank)를 산출합니다.  
- Top 10 + 판정불가 항목을 **Gemini 결과보고서**로 요약합니다.  
- 결과 관련 질문을 할 수 있습니다.
"""
)

st.markdown(
    """
### 🔎 CI 계산 공식 및 의미

이 웹앱은 Bitaraf et al. (Buildings, 2024)의 **개선된 BIM 기반 간섭 우선순위 산정 방법**을 참고하여  
아래와 같은 CI(Clash Importance) 공식을 사용합니다.

> CI = P × WS × WMEP × N × R × U

- P : Clash 결과에서 가져온 간섭 깊이(침투량)  
- WS : 기둥, 보, 기초, 전단벽, 슬래브 등 구조 요소 가중치  
- WMEP : 덕트, 설비, 배관, 전기설비 등 MEP 요소 가중치  
- N : 동일 MEP 요소가 발생시키는 간섭 개수  
- R : 층별 간섭 밀도 비율 (해당 층 간섭 수 / 최다 층 간섭 수)  
- U : 용도 계수 (현재 1.0로 고정)

논문의 기본 공식을 그대로 사용하되,  
N·R·U 세 변수는 업로드한 Clash 테이블만으로 계산할 수 있도록
독자적으로 단순화·재구성한 버전이라는 점을 참고해 주세요.
"""
)

# ----------------------------------------

def detect_mep_type(s: str) -> str:
    t = str(s).lower()
    if "airterminal" in t: return "AirTerminal"
    if "ductsegment" in t or "duct segment" in t: return "DuctSegment"
    if "pipe" in t: return "PipeSegment"
    if "cabletray" in t or "cable_tray" in t: return "CableTray"
    return "OtherMEP"

def detect_struct_type(s: str) -> str:
    t = str(s).lower()
    if "column" in t or "ifccolumn" in t: return "Column"
    if "beam" in t or "ifcbeam" in t: return "Beam"
    if "slab" in t or "roof" in t or "ifcslab" in t: return "Slab"
    if "wall" in t or "ifcwall" in t: return "Wall"
    if "pile" in t or "ifcpile" in t: return "Pile"
    return "OtherStruct"

def ws_from_struct(st_type: str) -> float:
    if st_type in ["Column","Beam"]: return 0.321
    if st_type == "Pile": return 0.188
    if st_type == "Wall": return 0.125
    if st_type == "Slab": return 0.045
    return 0.045

def w_mep_from_type(mep_type: str) -> float:
    if mep_type == "DuctSegment": return 0.54
    if mep_type == "AirTerminal": return 0.28
    if mep_type == "PipeSegment": return 0.12
    return 0.06

# ----------------------------------------

def compute_ci(df: pd.DataFrame, u_use: float = 1.0, p_min_threshold: float = 0.0) -> pd.DataFrame:
    df = df.copy()

    col_clash_name = "간섭 이름"
    col_distance = "거리"
    col_mep_id = "항목 ID 1"
    col_mep_floor = "도면층"
    col_mep_type_raw = "항목 유형1"
    col_st_id = "항목 ID 2"
    col_st_floor = "도면층.1"
    col_st_type_raw = "항목 유형2"

    df[col_distance] = pd.to_numeric(df[col_distance], errors="coerce").fillna(0.0)
    df["P"] = df[col_distance].abs()
    df["MEP_Type"] = df[col_mep_type_raw].fillna(df[col_mep_id]).apply(detect_mep_type)
    df["ST_Type"] = df[col_st_type_raw].fillna(df[col_st_id]).apply(detect_struct_type)

    df["WS"] = df["ST_Type"].apply(ws_from_struct)
    df["WMEP"] = df["MEP_Type"].apply(w_mep_from_type)

    df["N"] = df.groupby(col_mep_id)[col_clash_name].transform("count").astype(float)

    floor_counts = df.groupby(col_mep_floor)[col_clash_name].transform("count").astype(float)
    max_floor_count = floor_counts.max() if floor_counts.max() > 0 else 1.0
    df["R"] = floor_counts / max_floor_count

    df["U"] = float(u_use)

    df["CI"] = df["P"] * df["WS"] * df["WMEP"] * df["N"] * df["R"] * df["U"]

    df["판정결과"] = "판정가능"

    # ----- 🔥 여기 수정됨 🔥 -----
    mask_unknown = (df["MEP_Type"] == "OtherMEP") | (df["ST_Type"] == "OtherStruct")
    df.loc[mask_unknown, "판정결과"] = "판정불가"
    # ------------------------------------

    df = df.sort_values("CI", ascending=False).reset_index(drop=True)
    df["CI_rank"] = df["CI"].rank(method="min", ascending=False).astype(int)

    return df

