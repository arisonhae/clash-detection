# clashdetection.py
# Streamlit 기반 Clash 우선순위 + Gemini 결과보고서 + 챗봇

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
  - WS, WMEP 값의 구조는 논문에서 제시한 BWM(Best–Worst Method) 기반 가중치 체계를 따릅니다.
- N : 동일 MEP 요소가 발생시키는 간섭 개수  
- R : 층별 간섭 밀도 비율(해당 층 간섭 수 / 최다 층 간섭 수)  
- U : 용도 계수 (현재는 1.0으로 고정)

논문의 기본 공식을 그대로 사용하되,  
N · R · U 세 변수의 구체적인 정의와 계산 방식은 이 웹앱에서 업로드한 Clash 테이블만으로 계산할 수 있도록  
독자적으로 단순화·재구성한 버전이라는 점을 함께 참고해 주세요.
"""
)

# ======================================
# 1. 타입 판별 함수 (MEP / 구조)
# ======================================

def detect_mep_type(s: str) -> str:
    t = str(s).lower()
    if "airterminal" in t:
        return "AirTerminal"
    if "ductsegment" in t or "duct segment" in t:
        return "DuctSegment"
    if "pipe" in t:
        return "PipeSegment"
    if "cabletray" in t or "cable_tray" in t:
        return "CableTray"
    return "OtherMEP"


def detect_struct_type(s: str) -> str:
    t = str(s).lower()
    if "column" in t or "ifccolumn" in t:
        return "Column"
    if "beam" in t or "ifcbeam" in t:
        return "Beam"
    if "slab" in t or "roof" in t or "ifcslab" in t:
        return "Slab"
    if "wall" in t or "ifcwall" in t:
        return "Wall"
    if "pile" in t or "ifcpile" in t:
        return "Pile"
    return "OtherStruct"

# ======================================
# 2. 가중치 함수
# ======================================

def ws_from_struct(st_type: str) -> float:
    if st_type == "Column": return 0.321
    if st_type == "Beam": return 0.321
    if st_type == "Pile": return 0.188
    if st_type == "Wall": return 0.125
    if st_type == "Slab": return 0.045
    return 0.045


def w_mep_from_type(mep_type: str) -> float:
    if mep_type == "DuctSegment": return 0.54
    if mep_type == "AirTerminal": return 0.28
    if mep_type == "PipeSegment": return 0.12
    return 0.06

# ======================================
# 3. CI 계산 함수
# ======================================

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

    required_cols = [
        col_clash_name, col_distance, col_mep_id, col_mep_floor,
        col_mep_type_raw, col_st_id, col_st_floor, col_st_type_raw
    ]

    for c in required_cols:
        if c not in df.columns:
            raise KeyError(f"필수 컬럼이 없습니다: {c}")

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

    df["CI_raw"] = df["P"] * df["WS"] * df["WMEP"] * df["N"] * df["R"] * df["U"]

    if p_min_threshold > 0:
        df["CI"] = df["CI_raw"]
        df.loc[df["P"] < p_min_threshold, "CI"] = 0.0
    else:
        df["CI"] = df["CI_raw"]

    df["판정결과"] = "판정가능"
    mask_unknown = (df["MEP_Type"] == "OtherMEP") | (df["ST_Type"] == "OtherStruct"]
    df.loc[mask_unknown, "판정결과"] = "판정불가"

    df = df.sort_values("CI", ascending=False).reset_index(drop=True)
    df["CI_rank"] = df["CI"].rank(method="min", ascending=False).astype(int)

    return df

# ======================================
# Gemini 설정
# ======================================

PREFERRED_MODELS = [
    "gemini-2.5-flash",
    "gemini-1.5-flash",
    "gemini-1.5-pro",
]

def init_gemini():
    api_key = st.secrets.get("GEMINI_API_KEY")
    if not api_key:
        st.sidebar.error("⚠️ GEMINI_API_KEY가 설정되지 않았습니다.")
        return None

    st.sidebar.markdown(f"🔑 Gemini key prefix: `{api_key[:6]}***`")
    st.sidebar.markdown(f"📦 google-generativeai 버전: `{genai.__version__}`")

    genai.configure(api_key=api_key)

    available_names = []
    try:
        models = list(genai.list_models())
        for m in models:
            methods = getattr(m, "supported_generation_methods", [])
            if "generateContent" in methods:
                available_names.append(m.name)
    except Exception:
        pass

    candidate_names = []
    if available_names:
        for pref in PREFERRED_MODELS:
            for an in available_names:
                if an.endswith(pref):
                    candidate_names.append(an.replace("models/", ""))
                    break

    if not candidate_names:
        candidate_names = PREFERRED_MODELS[:]

    last_error = None
    for name in candidate_names:
        try:
            model = genai.GenerativeModel(name)
            _ = model.generate_content("테스트입니다.")
            st.sidebar.success(f"✅ Gemini 연결 성공 (사용 모델: `{name}`)")
            st.session_state["gemini_model_name"] = name
            return model
        except Exception as e:
            last_error = e

    st.sidebar.error(f"❌ Gemini 모델 초기화 실패: {last_error}")
    return None

# ======================================
# Gemini 결과보고서 생성
# ======================================

def generate_report_gemini(model, df_ci: pd.DataFrame) -> str:
    if df_ci is None or df_ci.empty:
        return "데이터가 없어 보고서를 생성할 수 없습니다."

    top10 = df_ci.head(10).copy()
    cols = [
        "CI_rank", "간섭 이름", "MEP 항목 ID", "ST 항목 ID",
        "MEP_Type", "ST_Type", "판정결과", "P", "WS", "WMEP", "N", "R", "CI"
    ]
    cols = [c for c in cols if c in top10.columns]
    top10_small = top10[cols]
    
    unknown_rows = df_ci[df_ci["판정결과"] == "판정불가"][cols]

    top10_md = top10_small.to_markdown(index=False)
    unknown_md = unknown_rows.to_markdown(index=False) if not unknown_rows.empty else "없음"

    prompt = f"""
너는 건설/BIM 간섭 검토를 돕는 엔지니어야.

[Top 10]
{top10_md}

[판정불가]
{unknown_md}

Top 10 특징 요약,
우선 조치 대상 3~5개,
판정불가 항목 안내까지 포함하여 한글 보고서 작성
"""
    response = model.generate_content(prompt)
    return response.text

# ======================================
# Gemini 챗봇
# ======================================

def init_chat_state():
    if "chat_history" not in st.session_state:
        st.session_state["chat_history"] = []


def chat_with_gemini(model, user_msg: str, df_ci: pd.DataFrame | None):
    context = ""
    if df_ci is not None and not df_ci.empty:
        top5 = df_ci.head(5).copy()
        cols = ["CI_rank","간섭 이름","MEP 항목 ID","ST 항목 ID","판정결과","CI"]
        cols = [c for c in cols if c in  top5.columns]
        context = top5[cols].to_markdown(index=False)

    history_text = ""
    for h in st.session_state["chat_history"][-6:]:
        role = "사용자" if h["role"] == "user" else "AI"
        history_text += f"{role}: {h['content']}\n"

    prompt = f"""
[최근 대화]
{history_text}

[Clash 우선순위 요약]
{context}

[사용자 질문]
{user_msg}
"""

    response = model.generate_content(prompt)
    return response.text

# ======================================
# UI
# ======================================

if "report_text" not in st.session_state:
    st.session_state["report_text"] = None

st.sidebar.header("📂 입력 데이터 업로드")
uploaded_file = st.sidebar.file_uploader("Clash 결과 CSV/XLSX 파일을 업로드하세요", type=["csv","xlsx"])

p_min_threshold = st.sidebar.number_input(
    "P 최소 간섭 깊이 기준 (선택, 0이면 사용 안 함)",
    min_value=0.0, max_value=1000.0, value=0.0, step=1.0,
)

st.sidebar.markdown("---")
st.sidebar.markdown("📌 파일 형식 예시: `간섭 이름, 거리, 항목 ID 1, 도면층, 항목 유형1...`")

df_ci = None

if uploaded_file is not None:
    st.subheader("📁 업로드 데이터 미리보기")

    try:
        if uploaded_file.name.lower().endswith(".csv"):
            df_raw = pd.read_csv(uploaded_file, encoding="utf-8-sig")
        else:
            df_raw = pd.read_excel(uploaded_file)
    except Exception as e:
        st.error(f"파일을 읽는 중 오류 발생: {e}")
        df_raw = None

    if df_raw is not None:
        st.dataframe(df_raw.head(20), use_container_width=True)

        st.subheader("🧮 CI 계산 및 Rank 산출")

        try:
            df_ci = compute_ci(df_raw, u_use=1.0, p_min_threshold=p_min_threshold)
            st.success("✅ CI 계산 및 Rank 산출 완료")

            st.session_state["report_text"] = None

            st.markdown("**상위 20개 간섭 (CI 기준 내림차순)**")
            show_cols = [
                "CI_rank","간섭 이름","MEP 항목 ID","ST 항목 ID","MEP_Type",
                "ST_Type","판정결과","P","WS","WMEP","N","R","CI",
            ]
            show_cols = [c for c in show_cols if c in df_ci.columns]
            st.dataframe(df_ci[show_cols].head(20), use_container_width=True)

            st.markdown("#### 📥 결과 파일 다운로드")
            out_csv = df_ci.to_csv(index=False, encoding="utf-8-sig")

            st.download_button(
                label="🔽 CI 결과 CSV 다운로드",
                data=out_csv,
                file_name="ci_result_with_rank.csv",
                mime="text/csv",
            )

        except Exception as e:
            st.error(f"CI 계산 중 오류 발생: {e}")

else:
    st.info("좌측에서 CSV/XLSX 파일을 업로드하면 CI 계산을 시작할 수 있습니다.")

model = init_gemini()

st.markdown("---")
st.markdown("### 🤖 Gemini 결과보고서")

if model is None:
    st.warning("Gemini 모델이 초기화되지 않았습니다.")
else:
    if df_ci is None or df_ci.empty:
        st.info("먼저 CSV/XLSX 업로드 후 CI 계산이 필요합니다.")
    else:
        if st.session_state["report_text"] is None:
            with st.spinner("Gemini가 결과보고서를 작성하는 중입니다..."):
                st.session_state["report_text"] = generate_report_gemini(model, df_ci)

        st.markdown("#### 📄 결과보고서 (AI 자동 생성)")
        st.write(st.session_state["report_text"])

st.markdown("---")
st.subheader("💬 Gemini 챗봇")

init_chat_state()

if model is None:
    st.warning("Gemini 모델이 초기화되지 않았습니다.")
else:
    for h in st.session_state["chat_history"]:
        role = "user" if h["role"] == "user" else "assistant"
        with st.chat_message(role):
            st.markdown(h["content"])

    user_input = st.chat_input("CI, Rank, 판정불가 의미나 결과 해석 등을 물어보세요.")
    if user_input:
        st.session_state["chat_history"].append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        with st.chat_message("assistant"):
            with st.spinner("AI가 답변 작성 중입니다..."):
                answer = chat_with_gemini(model, user_input, df_ci)
                st.markdown(answer)

        st.session_state["chat_history"].append(
            {"role": "assistant", "content": answer}
        )
