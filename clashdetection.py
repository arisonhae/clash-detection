# clashdetection.py
# Streamlit 기반 Clash 우선순위 + Gemini 결과보고서 + 챗봇

import os
import io
import pandas as pd
import streamlit as st
import google.generativeai as genai

# ======================================
# 0. 기본 설정
# ======================================

st.set_page_config(
    page_title="AI Clash Agent (CI Ranking)",
    page_icon="🧱",
    layout="wide"
)

st.title("🧱 AI Clash Agent (CI Ranking + Gemini Report)")

st.markdown("""
업로드한 Clash CSV/XLSX를 기반으로 **간섭 중요도(CI)**를 계산하고,
- 우선 수정해야 할 간섭 순위(Rank)를 산출합니다.  
- Top 10 + 판정불가 항목을 **Gemini 결과보고서**로 요약합니다.  
- 아래 챗봇에서 결과 관련 질문도 할 수 있습니다.
""")


# ======================================
# 1. 타입 판별 함수 (MEP / 구조)
# ======================================

def detect_mep_type(s: str) -> str:
    """MEP 항목 ID/유형 문자열에서 MEP 타입 분류"""
    t = str(s)
    t_low = t.lower()

    if "airterminal" in t_low:
        return "AirTerminal"
    if "ductsegment" in t_low or "duct segment" in t_low:
        return "DuctSegment"
    if "pipe" in t_low:
        return "PipeSegment"
    if "cabletray" in t_low or "cable_tray" in t_low:
        return "CableTray"

    # 그 외는 판정불가 대상으로 처리
    return "OtherMEP"


def detect_struct_type(s: str) -> str:
    """구조 항목 ID/유형 문자열에서 구조 타입 분류"""
    t = str(s)
    t_low = t.lower()

    if "column" in t_low or "ifccolumn" in t_low:
        return "Column"
    if "beam" in t_low or "ifcbeam" in t_low:
        return "Beam"
    if "slab" in t_low or "roof" in t_low or "ifcslab" in t_low:
        return "Slab"
    if "wall" in t_low or "ifcwall" in t_low:
        return "Wall"
    if "pile" in t_low or "ifcpile" in t_low:
        return "Pile"

    # 그 외는 판정불가 대상으로 처리
    return "OtherStruct"


# ======================================
# 2. 가중치 함수
# ======================================

def ws_from_struct(st_type: str) -> float:
    """
    구조 요소 가중치 (WS) - BWM 기반 값 예시
    Column / Beam = 0.321
    Pile(Foundation) = 0.188
    Wall(Shearwall/Brace) = 0.125
    Slab/Roof = 0.045
    기타는 보수적으로 Slab 수준
    """
    if st_type == "Column":
        return 0.321
    if st_type == "Beam":
        return 0.321
    if st_type == "Pile":
        return 0.188
    if st_type == "Wall":
        return 0.125
    if st_type == "Slab":
        return 0.045
    # 기타 구조
    return 0.045


def w_mep_from_type(mep_type: str) -> float:
    """
    MEP 요소 가중치 (WMEP) - 예시 값
    Duct > AirTerminal > Pipe > Others
    """
    if mep_type == "DuctSegment":
        return 0.54
    if mep_type == "AirTerminal":
        return 0.28
    if mep_type == "PipeSegment":
        return 0.12
    # CableTray, 기타 등
    return 0.06


# ======================================
# 3. CI 계산 함수
# ======================================

def compute_ci(
    df: pd.DataFrame,
    u_use: float = 1.0,
    p_min_threshold: float = 0.0
) -> pd.DataFrame:
    """
    CI = P × WS × WMEP × N × R × U
    - P: 간섭 깊이(거리 절대값)
    - WS: 구조 요소 가중치
    - WMEP: 설비 요소 가중치
    - N: 동일 MEP 부재가 발생시키는 간섭 개수
    - R: 층별 간섭 밀도 (해당 층 간섭 수 / 최다 층 간섭 수)
    - U: 용도 계수 (현재 1.0 고정)

    입력 데이터는 다음 형식을 가정한다.
    - 간섭 이름
    - 거리
    - 항목 ID 1 (MEP)
    - 도면층 (MEP 층)
    - 항목 유형1 (MEP 타입)
    - 항목 ID 2 (ST)
    - 도면층.1 (ST 층)
    - 항목 유형2 (ST 타입)
    """
    df = df.copy()

    # 1) 실제 파일 컬럼 이름 (데이터 정리본.xlsx 형식 고정)
    col_clash_name   = "간섭 이름"
    col_distance     = "거리"
    col_mep_id       = "항목 ID 1"
    col_mep_floor    = "도면층"
    col_mep_type_raw = "항목 유형1"
    col_st_id        = "항목 ID 2"
    col_st_floor     = "도면층.1"
    col_st_type_raw  = "항목 유형2"

    # 필수 컬럼 체크
    required_cols = [
        col_clash_name, col_distance,
        col_mep_id, col_mep_floor, col_mep_type_raw,
        col_st_id, col_st_floor, col_st_type_raw
    ]
    for c in required_cols:
        if c not in df.columns:
            raise KeyError(f"필수 컬럼이 없습니다: {c}")

    # 2) 거리 숫자형 변환 + P (간섭 깊이)
    df[col_distance] = pd.to_numeric(df[col_distance], errors="coerce").fillna(0.0)
    df["P"] = df[col_distance].abs()

    # 3) 타입 분류 (MEP / ST)
    df["MEP_Type"] = df[col_mep_type_raw].fillna(df[col_mep_id]).apply(detect_mep_type)
    df["ST_Type"] = df[col_st_type_raw].fillna(df[col_st_id]).apply(detect_struct_type)

    # 4) 가중치 계산
    df["WS"] = df["ST_Type"].apply(ws_from_struct)
    df["WMEP"] = df["MEP_Type"].apply(w_mep_from_type)

    # 5) N: 동일 MEP ID가 만드는 간섭 개수
    df["N"] = (
        df.groupby(col_mep_id)[col_clash_name]
        .transform("count")
        .astype(float)
    )

    # 6) R: 층별 간섭 수 비율 (해당 층 간섭 / 최다 층 간섭)
    floor_counts = (
        df.groupby(col_mep_floor)[col_clash_name]
        .transform("count")
        .astype(float)
    )
    max_floor_count = floor_counts.max() if floor_counts.max() > 0 else 1.0
    df["R"] = floor_counts / max_floor_count

    # 7) U: 용도 계수 (지금은 1.0)
    df["U"] = float(u_use)

    # 8) CI 원값
    df["CI_raw"] = df["P"] * df["WS"] * df["WMEP"] * df["N"] * df["R"] * df["U"]

    # 9) P 최소 기준 (원하면 작은 간섭 제거)
    if p_min_threshold > 0:
        df["CI"] = df["CI_raw"]
        df.loc[df["P"] < p_min_threshold, "CI"] = 0.0
    else:
        df["CI"] = df["CI_raw"]

    # 10) 판정불가 여부 (타입을 제대로 분류 못한 경우)
    df["판정결과"] = "판정가능"
    mask_unknown = (df["MEP_Type"] == "OtherMEP") | (df["ST_Type"] == "OtherStruct")
    df.loc[mask_unknown, "판정결과"] = "판정불가"

    # 11) 보고서/표시에 쓸 alias 컬럼 (사람이 보기 좋은 이름)
    df["MEP 항목 ID"]   = df[col_mep_id]
    df["MEP 도면층"]     = df[col_mep_floor]
    df["MEP 항목 유형"]  = df[col_mep_type_raw]
    df["ST 항목 ID"]    = df[col_st_id]
    df["ST 도면층"]      = df[col_st_floor]
    df["ST 항목 유형"]   = df[col_st_type_raw]

    # 12) 정렬 + Rank
    df = df.sort_values("CI", ascending=False).reset_index(drop=True)
    df["CI_rank"] = df["CI"].rank(method="min", ascending=False).astype(int)

    return df


# ======================================
# 4. Gemini 설정 함수
# ======================================

def init_gemini():
    api_key = st.secrets["google"]["api_key"]
    if not api_key:
        st.warning("⚠️ Gemini API 키가 설정되지 않았습니다. secrets.toml을 확인해주세요.")
        return None

    # 키 앞부분 확인용 (잘 읽히는지 체크)
    st.sidebar.markdown(f"🔑 Gemini key prefix: `{api_key[:6]}***`")

    genai.configure(api_key=api_key)

    # ----- 연결 테스트 -----
    try:
        test_model = genai.GenerativeModel("gemini-pro")  # 🔁 여기만 변경
        _ = test_model.generate_content("테스트입니다. 한 줄만 답해줘.")
        st.sidebar.success("✅ Gemini 연결 테스트 성공")
    except Exception as e:
        st.sidebar.error(f"❌ Gemini 테스트 실패: {e}")
        return None

    # 실제 사용할 모델도 동일하게
    return genai.GenerativeModel("gemini-pro")  # 🔁 여기도 변경
# ======================================
# 5. Gemini 결과보고서 생성
# ======================================

def generate_report_gemini(model, df_ci: pd.DataFrame) -> str:
    """
    Top 10 + 판정불가 항목을 기반으로 결과보고서 생성
    """
    if df_ci is None or df_ci.empty:
        return "데이터가 없어 보고서를 생성할 수 없습니다."

    top10 = df_ci.head(10).copy()

    # 보고서에 넘길 최소 컬럼만 정리
    cols_for_report = [
        "CI_rank", "간섭 이름",
        "MEP 항목 ID", "ST 항목 ID",
        "MEP_Type", "ST_Type",
        "판정결과", "P", "WS", "WMEP", "N", "R", "CI"
    ]
    cols_for_report = [c for c in cols_for_report if c in top10.columns]
    top10_small = top10[cols_for_report]

    # 판정불가 항목만 따로 추출
    unknown_rows = df_ci[df_ci["판정결과"] == "판정불가"][cols_for_report]

    top10_md = top10_small.to_markdown(index=False)
    unknown_md = unknown_rows.to_markdown(index=False) if not unknown_rows.empty else "없음"

    prompt = f"""
너는 건설/BIM 간섭 검토를 돕는 엔지니어야.
아래 표는 Clash 간섭 우선순위 평가 결과이며, CI가 클수록 먼저 처리해야 하는 간섭이다.

[Top 10 Clash 목록]
{top10_md}

[판정불가(타입 분류 실패) Clash 목록]
{unknown_md}

다음 조건에 따라 **한국어** 보고서를 작성해줘.

1. '요약' 섹션에서 Top 10의 전반적 특징을 2~3문장으로 설명.
2. '우선 조치 대상' 섹션에서 상위 3~5개 간섭을 간단히 설명하되,
   - 각 간섭의 MEP/구조 타입,
   - 왜 중요한지(간섭 깊이, 반복 발생 여부, 층 밀도 등)을 설명.
3. '판정불가 항목' 섹션에서 위 표의 판정불가 간섭이 있다면
   - 몇 건인지,
   - 추가 모델 정보(예: MEP/구조 타입 정보 보완)가 필요하다는 점을 명시적으로 언급.
4. 최대한 실무 엔지니어가 이해하기 쉬운 표현으로 작성하고, 너무 과장된 표현은 피한다.
"""

    response = model.generate_content(prompt)
    return response.text


# ======================================
# 6. Gemini 챗봇
# ======================================

def init_chat_state():
    if "chat_history" not in st.session_state:
        st.session_state["chat_history"] = []  # list of {"role": "user"/"assistant", "content": str}


def chat_with_gemini(model, user_msg: str, df_ci: pd.DataFrame | None):
    """
    간단한 Q&A 챗봇.
    - df_ci가 있으면, 상위 일부 데이터를 context로 같이 넘김.
    """
    # context로 보낼 요약 (너무 길면 줄이기)
    context = ""
    if df_ci is not None and not df_ci.empty:
        top5 = df_ci.head(5).copy()
        cols = ["CI_rank", "간섭 이름", "MEP 항목 ID", "ST 항목 ID", "판정결과", "CI"]
        cols = [c for c in cols if c in top5.columns]
        context = top5[cols].to_markdown(index=False)

    history_text = ""
    for h in st.session_state["chat_history"][-6:]:  # 최근 몇 개만
        role = "사용자" if h["role"] == "user" else "AI"
        history_text += f"{role}: {h['content']}\n"

    prompt = f"""
너는 BIM Clash 분석 결과를 설명해주는 한국어 도우미야.

[최근 대화]
{history_text}

[현재 Clash 우선순위 상위 일부 요약]
{context}

위 상황을 참고해서, 사용자의 마지막 질문에 답변해줘.
답변은 최대한 구체적으로, 하지만 과하게 길지 않게 써줘.
사용자가 CI, Rank, 판정불가 의미를 물으면 각각 간단히 정의해줘.
"""

    full_prompt = prompt + f"\n\n[사용자 질문]\n{user_msg}"

    response = model.generate_content(full_prompt)
    return response.text


# ======================================
# 7. 메인 UI
# ======================================

st.sidebar.header("📂 입력 데이터 업로드")

uploaded_file = st.sidebar.file_uploader(
    "Clash 결과 CSV/XLSX 파일을 업로드하세요",
    type=["csv", "xlsx"]
)

p_min_threshold = st.sidebar.number_input(
    "P 최소 간섭 깊이 기준 (선택, 0이면 사용 안 함)",
    min_value=0.0,
    max_value=1000.0,
    value=0.0,
    step=1.0
)

st.sidebar.markdown("---")
st.sidebar.markdown("📌 파일 형식 예시: `간섭 이름, 거리, 간섭 지점, 항목 ID 1, 도면층, 항목 유형1, 항목 ID 2, 도면층.1, 항목 유형2`")


df_ci = None

# ---------- 파일 처리 & CI 계산 ----------
if uploaded_file is not None:
    st.subheader("1️⃣ 업로드 데이터 미리보기")

    try:
        if uploaded_file.name.lower().endswith(".csv"):
            df_raw = pd.read_csv(uploaded_file, encoding="utf-8-sig")
        else:
            df_raw = pd.read_excel(uploaded_file)  # openpyxl이 requirements에 들어 있어야 함
    except Exception as e:
        st.error(f"파일을 읽는 중 오류가 발생했습니다: {e}")
        df_raw = None

    if df_raw is not None:
        st.dataframe(df_raw.head(20), use_container_width=True)

        st.subheader("2️⃣ CI 계산 및 Rank 산출")

        try:
            df_ci = compute_ci(df_raw, u_use=1.0, p_min_threshold=p_min_threshold)
            st.success("✅ CI 계산 및 Rank 산출이 완료되었습니다.")

            # 상위 20개 표시
            st.markdown("**상위 20개 간섭 (CI 기준 내림차순)**")
            show_cols = [
                "CI_rank", "간섭 이름",
                "MEP 항목 ID", "ST 항목 ID",
                "MEP_Type", "ST_Type",
                "판정결과", "P", "WS", "WMEP", "N", "R", "CI"
            ]
            show_cols = [c for c in show_cols if c in df_ci.columns]
            st.dataframe(df_ci[show_cols].head(20), use_container_width=True)

            # 다운로드 버튼 (CSV)
            st.markdown("#### 📥 결과 파일 다운로드")
            out_csv = df_ci.to_csv(index=False, encoding="utf-8-sig")
            st.download_button(
                label="🔽 CI 결과 CSV 다운로드",
                data=out_csv,
                file_name="ci_result_with_rank.csv",
                mime="text/csv",
            )

        except KeyError as e:
            st.error(f"CI 계산에 필요한 컬럼이 없습니다: {e}")
        except Exception as e:
            st.error(f"CI 계산 중 오류가 발생했습니다: {e}")

else:
    st.info("좌측에서 Clash CSV/XLSX 파일을 업로드하면 CI 계산을 시작할 수 있습니다.")


# ---------- Gemini 모델 초기화 ----------
model = init_gemini()

# ---------- 결과보고서 ----------
st.markdown("---")
st.subheader("3️⃣ Gemini 결과보고서 생성 (Top 10 + 판정불가 포함)")

if model is None:
    st.warning("Gemini 모델이 초기화되지 않았습니다. API 키 설정을 먼저 해주세요.")
else:
    if df_ci is None or df_ci.empty:
        st.info("먼저 CSV/XLSX를 업로드하고 CI를 계산해야 결과보고서를 생성할 수 있습니다.")
    else:
        if st.button("📄 Gemini로 결과보고서 생성"):
            with st.spinner("Gemini가 보고서를 작성하는 중입니다..."):
                report_text = generate_report_gemini(model, df_ci)
            st.markdown("#### 📄 결과보고서 (AI 생성)")
            st.write(report_text)


# ---------- 챗봇 ----------
st.markdown("---")
st.subheader("4️⃣ Gemini 챗봇 (결과 관련 질문)")

init_chat_state()

if model is None:
    st.warning("Gemini 모델이 초기화되지 않았습니다. API 키 설정을 먼저 해주세요.")
else:
    # 기존 대화 출력
    for h in st.session_state["chat_history"]:
        if h["role"] == "user":
            st.markdown(f"**👤 사용자:** {h['content']}")
        else:
            st.markdown(f"**🤖 AI:** {h['content']}")

    user_input = st.text_input("질문을 입력하세요. (예: CI가 뭐야?, 판정불가는 어떤 의미야?)")

    if user_input:
        st.session_state["chat_history"].append({"role": "user", "content": user_input})
        with st.spinner("AI가 답변을 작성 중입니다..."):
            answer = chat_with_gemini(model, user_input, df_ci)
        st.session_state["chat_history"].append({"role": "assistant", "content": answer})
        st.experimental_rerun()

