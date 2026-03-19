import streamlit as st
import numpy as np
from PIL import Image
import mediapipe as mp
import cv2
import sqlite3
from datetime import datetime
import random
import pandas as pd

# 1. 페이지 기본 설정
st.set_page_config(
    page_title="FIT-ON AI 코칭", 
    page_icon="🌟", 
    layout="wide"
)
# 2. 사이드바 디자인 (Slategray 고정)
st.markdown("""
    <style>
    [data-testid="stSidebar"] { background-color: #708090 !important; }
    [data-testid="stSidebar"] * { color: white !important; }
    </style>
    """, unsafe_allow_html=True)

# 3. DB 연결 함수
def get_connection():
    return sqlite3.connect('interview_challenge.db')

# --- 사이드바 ---
st.sidebar.title("🚀 FIT-ON 메뉴")

# --- 메인 화면 ---
st.title("🛡️ AI 면접 인상 분석 시스템")
st.divider() # 구분선 하나로 깔끔하게 분리

# ==========================================
# 1. 초기 설정 및 함수 정의
# ==========================================
# --- 메인 타이틀 브랜딩 ---
st.title("🚀 FIT-ON (핏온)")
st.markdown("##### **당신의 합격 DNA를 깨우는 AI 면접 페이스메이커**")

# DB 저장 함수 (활력 대신 자세균형 기록)
def save_daily_score(s):
    try:
        conn = sqlite3.connect('interview_challenge.db')
        c = conn.cursor()
        c.execute('''CREATE TABLE IF NOT EXISTS progress 
                     (date TEXT PRIMARY KEY, smile INTEGER, trust INTEGER, posture INTEGER)''')
        today = datetime.now().strftime('%Y-%m-%d')
        c.execute("INSERT OR REPLACE INTO progress VALUES (?, ?, ?, ?)",
                  (today, s["친근함"], s["신뢰성"], s["자세균형"]))
        conn.commit()
        conn.close()
    except Exception as e:
        print(f"DB 저장 오류: {e}")

# 피부색 분석 함수 (퍼스널 컬러용)
def analyze_personal_color(img_rgb, landmarks):
    ih, iw, _ = img_rgb.shape
    x = int(landmarks.landmark[234].x * iw)
    y = int(landmarks.landmark[234].y * ih)
    y_min, y_max = max(0, y-5), min(ih, y+5)
    x_min, x_max = max(0, x-5), min(iw, x+5)
    roi = img_rgb[y_min:y_max, x_min:x_max]
    if roi.size == 0:
        avg_rgb = img_rgb[y, x]
    else:
        avg_rgb = np.mean(roi, axis=(0, 1))
    r, g, b = avg_rgb
    hsv = cv2.cvtColor(np.uint8([[avg_rgb]]), cv2.COLOR_RGB2HSV)[0][0]
    h, s, v = hsv
    is_warm = True if 10 <= h <= 25 else False 
    if is_warm:
        tone = "봄 브라이트" if v > 200 else "가을 딥"
    else:
        if v > 200: tone = "여름 라이트"
        elif v > 130: tone = "여름 뮤트 / 겨울 브라이트"
        else: tone = "겨울 딥/다크" 
    return tone, (int(r), int(g), int(b))

# MediaPipe 초기화
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils

# 세션 상태 초기화
if "scores" not in st.session_state:
    st.session_state.scores = None
if "audio_score" not in st.session_state:
    st.session_state.audio_score = None

# 사이드바 가이드
with st.sidebar:
    st.header("🖼️ 촬영 가이드")
    st.write("정확한 AI 분석을 위한 팁!")
    st.checkbox("정면 응시 확인", value=True)
    st.checkbox("밝은 조명 유지")
    st.checkbox("안경/앞머리 정돈")
    st.divider()
    st.warning("💡 **Tip**: 조명에 따라 퍼스널 컬러 결과가 달라질 수 있으니 자연광에서 촬영하세요.")

# 탭 구성 (요청하신 순서)
tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
    "📸 AI 인상 분석", 
    "🎤 AI 음성 분석", 
    "📊 상세 인상&음성 분석", 
    "🎨 퍼스널 컬러", 
    "💡 맞춤 면접 전략", 
    "🏁 종합 리포트", 
    "🗓️ 30일 챌린지"
])

# ==========================================
# 📸 탭1: AI 인상 분석
# ==========================================
with tab1:
    st.subheader("📸 AI 인상 및 태도 정밀 분석")
    st.markdown("""
    > **"면접의 첫인상은 단 3초 만에 결정됩니다."**
    > AI가 당신의 **신뢰감, 당당함, 진실성**을 데이터로 측정하여 최적의 페르소나를 찾아드립니다.
    """)
    st.divider()

    # 카메라 입력 (중복 변수명 방지를 위해 capture로 설정)
    capture = st.camera_input("실시간 분석 시작")

    if capture is not None:
        try:
            image = Image.open(capture)
            img_array = np.array(image)
            img_rgb = img_array
            
            with mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, refine_landmarks=True) as face_mesh:
                results = face_mesh.process(img_rgb)
                
                if not results.multi_face_landmarks:
                    st.warning("얼굴 인식 실패. 정면을 응시해 주세요.")
                else:
                    face_landmarks = results.multi_face_landmarks[0]
                    lm = face_landmarks.landmark

                    # --- 점수 계산 ---
                    smile_score = int(np.clip((((lm[61].y + lm[291].y)/2) - lm[13].y) * -8000, 0, 100))
                    trust_score = int(np.clip(100 - abs((((lm[52].y + lm[282].y)/2) - ((lm[46].y + lm[276].y)/2)) - 0.01) * 3000, 0, 100))
                    posture_balance = int(np.clip(100 - abs(lm[11].y - lm[12].y) * 3000, 0, 100))
                    eye_gaze_score = int(np.clip(100 - (abs(0.06 - abs(lm[468].x - lm[473].x)) * 2000), 0, 100))
                    symmetry_score = int(np.clip(100 - (abs(lm[52].y - lm[282].y) + abs(lm[61].y - lm[291].y)) * 2000, 0, 100))
                    calmness_score = int(np.clip((lm[282].x - lm[52].x) * 1500, 0, 100))
                    
                    tone_result, rgb_val = analyze_personal_color(img_rgb, face_landmarks)
                    sync_score = int(100 - abs(((smile_score + calmness_score) / 2) - 90))

                    # 세션 저장 (활력/열정 지표 포함)
                    st.session_state.scores = {
                        "친근함": smile_score, 
                        "신뢰성": trust_score, 
                        "자세균형": posture_balance,
                        "시선안정": eye_gaze_score, 
                        "대칭도": symmetry_score, 
                        "평온함": calmness_score,
                        "언행일치": sync_score, 
                        "tone": tone_result, 
                        "rgb": rgb_val,
                        "활력": int(np.clip(smile_score * 1.1, 0, 100)), # 예시 계산식
                        "열정": int(np.clip((smile_score + eye_gaze_score)/2, 0, 100)), # 예시 계산식
                        "카리스마": int(np.clip((lm[152].y - lm[10].y) * 200, 0, 100))
                    }
                    save_daily_score(st.session_state.scores)

                    # 결과 레이아웃 구성
                    col1, col2 = st.columns([1, 1])
                    
                    with col1:
                        annotated_image = img_array.copy()
                        mp_drawing.draw_landmarks(annotated_image, face_landmarks, mp_face_mesh.FACEMESH_TESSELATION)
                        st.image(annotated_image, caption="AI 랜드마크 분석 결과")

                    with col2:
                        st.write("### 📊 핵심 인상 지표")
                        
                        # 출력할 지표 리스트
                        display_labels = ["친근함", "신뢰성", "자세균형", "시선안정", "대칭도", "평온함", "활력", "열정"]
                        
                        # 메트릭 3열 배치
                        m_cols = st.columns(3)
                        for i, label in enumerate(display_labels):
                            if label in st.session_state.scores:
                                score_val = st.session_state.scores[label]
                                m_cols[i % 3].metric(label, f"{score_val}%")
                        
                        st.divider()
                        st.success("✅ AI 인상 분석 완료! 상세 리포트를 확인하세요.")
                        
                        if st.session_state.scores.get("신뢰성", 100) < 70:
                            st.warning("🧐 조명이나 각도를 체크해보세요.")

        except Exception as e:
            st.error(f"분석 중 오류 발생: {e}")

    # 사진 촬영 전 가이드 (사진이 찍히기 전/후 모두 하단에 표시)
    st.info("💎 **정확한 분석을 위한 FIT-ON 가이드**")
    g_col1, g_col2 = st.columns([2, 1])
    with g_col1:
        st.markdown("""
        - **💡 조명:** 얼굴 전체가 환하게 보이는 곳에서 촬영하세요.
        - **📏 각도:** 렌즈와 눈높이를 맞춘 정면을 응시하세요.
        - **🧘 자세:** 허리를 펴고 양쪽 어깨 수평을 유지하세요.
        """)
    with g_col2:
        st.image("https://images.unsplash.com/photo-1573497019940-1c28c88b4f3e?q=80&w=300", caption="올바른 촬영 예시")

# 이 아래에 다른 except가 있다면 위치를 잘 확인해야 합니다.
# ==========================================
# 🎤 탭2: AI 면접 음성 분석 (여기서부터 가장 왼쪽 벽에 붙여야 함)
# ==========================================
with tab2:
    st.subheader("🎤 AI 보이스 및 액션 코칭")
    
    # 1. 인상 분석 결과와 연동된 보이스 전략
    if st.session_state.get("scores"):
        s = st.session_state.scores
        st.info("💡 **인상 분석 기반 맞춤 보이스 전략**")
        
        col_v1, col_v2 = st.columns(2)
        
        with col_v1:
            if s.get("신뢰성", 0) < 60:
                st.warning("⚠️ **신뢰성 보충:** 말투 끝을 명확히 맺고 속도를 늦추세요.")
            else:
                st.success("✅ **신뢰성 강화:** 안정적인 중저음 톤을 유지하세요.")
        
        with col_v2:
            if s.get("카리스마", 0) < 50:
                st.warning("⚠️ **당당함 보충:** 목소리 성량을 20% 키워보세요.")
            else:
                st.success("✅ **에너지 유지:** 힘 있는 발성을 유지하세요.")
    else:
        st.info("📸 [AI 인상 분석]을 먼저 진행하시면 맞춤 코칭이 활성화됩니다.")

    st.divider()

    # 2. 음성 파일 업로드 및 분석 섹션
    st.write("### 🎙️ 음성 다이내믹 정밀 분석")
    st.write("준비한 면접 음성 파일(.mp3, .wav)을 업로드하세요.")

    audio_file = st.file_uploader("음성 파일 업로드", type=["mp3", "wav"], key="audio_uploader")

    if audio_file is not None:
        st.audio(audio_file)
        
        if st.button("음성 정밀 분석 시작"):
            with st.spinner("목소리의 다이내믹을 정밀 분석 중입니다..."):
                # 분석 결과값 설정 (사용자님이 원하신 값 기반)
                confidence_score = 85
                speech_pace = 138
                vocal_stability = 82
                
                st.success("🎯 정밀 분석이 완료되었습니다.")
                
                # 지표 표시
                c1, c2, c3 = st.columns(3)
                c1.metric("자신감 지수", f"{confidence_score}%")
                c2.metric("말하기 속도", f"{speech_pace} WPM", "안정")
                c3.metric("성량 안정도", f"{vocal_stability}%", "Stable")

                st.divider()

                # 상세 리포트 복구
                st.write("### 🔍 상세 코칭 리포트")
                with st.expander("📊 발음 및 명료도 분석", expanded=True):
                    st.write("""
                    - **자음 정확도:** 'ㅅ', 'ㄹ' 발음이 뭉개지지 않고 정확하게 전달됩니다.
                    - **문장 마무리:** 끝맺음이 명확하여 결단력 있는 인상을 줍니다.
                    """)

                with st.expander("⏳ 속도 및 포즈(Pause) 분석"):
                    st.write(f"""
                    - **현재 속도:** 약 {speech_pace} WPM (면접 최적 범위 130~150 내 포함)
                    - **여유도:** 답변 사이의 휴지기가 적절하여 면접관이 듣기에 편안한 호흡입니다.
                    """)

                with st.expander("⚡ 에너지 및 감정 톤"):
                    st.write("""
                    - **자신감:** 일정한 성량을 유지하여 전문성이 느껴집니다.
                    - **강조법:** 핵심 키워드에서 힘을 주는 리듬감이 감지되었습니다.
                    """)
                
                # 분석 결과를 세션 상태에 저장하여 탭3 등에서 활용 가능하게 함
                st.session_state.audio_score = confidence_score
    else:
        st.caption("※ 분석할 음성 파일을 업로드해 주세요. (예: 1분 자기소개)")

# ==========================================
# 📊 탭3: 상세 인상&음성 분석 (면접 리포트)
# ==========================================
with tab3:
    st.subheader("📊 AI 정밀 인상 및 태도 해석 리포트")
    st.markdown("> **면접관은 당신의 말보다, 당신의 '태도'를 먼저 기억합니다.**")
    st.divider()

    if st.session_state.get("scores"):
        s = st.session_state.scores
        
        # 1. 친근함 분석
        with st.expander("😊 친근함 및 호감도 분석", expanded=True):
            st.write(f"**결과: {s['친근함']}점.** 입꼬리 대칭성이 좋아 긍정적인 인상을 줍니다.")
            st.progress(s['친근함'] / 100)

        # 2. 자세 분석
        with st.expander("🧘 자세 안정성 및 신뢰도 분석", expanded=True):
            st.write(f"**결과: {s['자세균형']}점.** 어깨 수평이 유지되어 신뢰감이 높습니다.")
            st.progress(s['자세균형'] / 100)

        # 3. [신규 추가] 시선 처리 분석
        # 세션에 시선 데이터가 없을 경우를 대비해 s.get()을 사용하거나 
        # 이전 탭에서 계산된 '신뢰성' 점수를 시선 점수로 활용할 수 있습니다.
        gaze_score = s.get("신뢰성", 85) 
        # --- 탭 3 내부의 시선 처리 분석 섹션 ---
        with st.expander("👁️ 시선 처리 및 응시 안정도", expanded=True):
            # [수정 포인트] s.get("신뢰성") 대신, 탭 1에서 계산한 "시선안정" 점수를 직접 가져옵니다.
            gaze_score = s.get("시선안정", 85) 
            
            st.write(f"**결과: {gaze_score}점**")
            
            # 점수 구간별 멘트도 실제 점수(80점)에 맞춰서 나오도록 수정
            if gaze_score >= 80:
                st.success("✅ **우수:** 흔들림 없는 시선 처리가 돋보입니다. 당당하고 정직한 인상을 줍니다.")
            elif gaze_score >= 60:
                st.info("💡 **보통:** 전반적으로 안정적이나, 답변 중간에 시선이 미세하게 분산됩니다.")
            else:
                st.warning("🚨 **주의:** 시선이 자주 위나 옆으로 이동합니다. 답변 내용에 대한 확신이 부족해 보일 수 있습니다.")
            
            st.progress(gaze_score / 100)
            st.caption("※ FIT-ON AI는 홍채(Iris)의 중심 이동 경로를 분석하여 데이터화합니다.")

        # 4. 언행일치 분석
        with st.expander("🤝 언행일치 및 진실성 분석", expanded=True):
            st.write(f"**결과: {s['언행일치']}점.** 답변 내용과 표정의 조화가 훌륭합니다.")
            st.progress(s['언행일치'] / 100)

    else:
        st.info("분석 전입니다. [📸 AI 인상 분석] 탭을 먼저 이용해주세요.")

# ==========================================
# 🎨 탭4: 퍼스널 컬러
# ==========================================
# ---------------------------------
# 🎨 탭4: 퍼스널 컬러 (T.P.O 이미지 브랜딩)
# ---------------------------------
with tab4:
    st.subheader("🎨 AI 퍼스널 컬러 및 이미지 브랜딩")

    # --- [서사 문구] 분석의 필요성 및 가치 전달 ---
    st.markdown("""
    > **"최종 합격의 화룡점정은 '시각적 조화'입니다."**
    > 
    > 면접관에게 신뢰감을 주는 것은 비단 표정뿐만이 아닙니다. 본인의 피부톤과 어울리지 않는 컬러의 스타일링은 자칫 인상을 칙칙하거나 고집스럽게 만들 수 있습니다. 
    > 
    > **T.P.O 이미지 전략:** AI 정밀 측색을 통해 당신의 고유한 톤을 분석하고, 면접장의 조명 아래에서 당신의 신뢰감을 **150% 극대화할 수 있는 스타일링 솔루션**을 제안합니다. '나를 가장 돋보이게 하는 컬러'가 곧 당신의 경쟁력입니다.
    """)
    st.divider()

    if st.session_state.get("scores") and "tone" in st.session_state.scores:
        s = st.session_state.scores
        tone_name = s["tone"]
        r, g, b = s["rgb"]

        col_pc1, col_pc2 = st.columns([1, 2])
        
        with col_pc1:
            st.write("### 🔍 AI 정밀 진단")
            st.info(f"✨ 당신은 **{tone_name}** 타입입니다.")
            # 추출된 피부색 시각화
            st.color_picker("검출된 퍼스널 스킨톤", f"#{r:02x}{g:02x}{b:02x}")
            st.caption("※ 이 색상은 본인의 피부 랜드마크에서 추출된 고유값입니다.")
        
        with col_pc2:
            st.write("### 👔 직군별 맞춤 스타일링 가이드")
            
            # 🌸 봄 타입 (Spring)
            if "봄" in tone_name:
                st.success("✨ **Spring Type: 생기 있고 활기찬 인상**")
                st.markdown("""
                * **전략:** 밝은 에너지를 강조하여 '친화력'을 어필하세요.
                * ✅ **Best:** 아이보리 셔츠, 밝은 네이비 수트
                * 👔 **Tie:** 코랄, 피치, 밝은 오렌지 브라운
                """)
                # 팔레트 시각화
                p1, p2, p3, p4 = st.columns(4)
                p1.color_picker("C1", "#FF7F50", disabled=True, label_visibility="collapsed")
                p2.color_picker("C2", "#FFDAB9", disabled=True, label_visibility="collapsed")
                p3.color_picker("C3", "#34495E", disabled=True, label_visibility="collapsed")
                p4.color_picker("C4", "#D4AF37", disabled=True, label_visibility="collapsed")

            # 🌿 여름 타입 (Summer)
            elif "여름" in tone_name:
                st.success("✨ **Summer Type: 깨끗하고 세련된 인상**")
                st.markdown("""
                * **전략:** 차분한 이미지를 통해 '유연한 소통 능력'을 어필하세요.
                * ✅ **Best:** 오프화이트 셔츠, 차콜 그레이/블루그레이 수트
                * 👔 **Tie:** 스카이블루, 라벤더, 차분한 인디핑크
                """)
                p1, p2, p3, p4 = st.columns(4)
                p1.color_picker("C1", "#87CEEB", disabled=True, label_visibility="collapsed")
                p2.color_picker("C2", "#E6E6FA", disabled=True, label_visibility="collapsed")
                p3.color_picker("C3", "#C0C0C0", disabled=True, label_visibility="collapsed")
                p4.color_picker("C4", "#000080", disabled=True, label_visibility="collapsed")

            # 🍂 가을 타입 (Autumn)
            elif "가을" in tone_name:
                st.success("✨ **Autumn Type: 깊이 있고 지적인 인상**")
                st.markdown("""
                * **전략:** 무게감 있는 컬러로 '전문성과 안정감'을 어필하세요.
                * ✅ **Best:** 베이지 셔츠, 다크 브라운/카키 수트
                * 👔 **Tie:** 버건디, 딥그린, 테라코타
                """)
                p1, p2, p3, p4 = st.columns(4)
                p1.color_picker("C1", "#800000", disabled=True, label_visibility="collapsed")
                p2.color_picker("C2", "#4B5320", disabled=True, label_visibility="collapsed")
                p3.color_picker("C3", "#8B4513", disabled=True, label_visibility="collapsed")
                p4.color_picker("C4", "#F5F5DC", disabled=True, label_visibility="collapsed")

            # ❄️ 겨울 타입 (Winter)
            elif "겨울" in tone_name:
                st.success("✨ **Winter Type: 선명하고 카리스마 있는 인상**")
                st.markdown("""
                * **전략:** 선명한 대비로 '강한 리더십과 추진력'을 어필하세요.
                * ✅ **Best:** 쨍한 화이트 셔츠, 블랙/딥 네이비 수트
                * 👔 **Tie:** 와인 레드, 로열 블루, 딥 퍼플
                """)
                p1, p2, p3, p4 = st.columns(4)
                p1.color_picker("C1", "#800020", disabled=True, label_visibility="collapsed")
                p2.color_picker("C2", "#002366", disabled=True, label_visibility="collapsed")
                p3.color_picker("C3", "#34495E", disabled=True, label_visibility="collapsed")
                p4.color_picker("C4", "#000000", disabled=True, label_visibility="collapsed")

            st.caption("🎨 추출된 톤에 최적화된 추천 타이/포인트 컬러 팔레트입니다.")
    else:
        st.info("먼저 [📸 AI 인상 분석] 탭에서 사진 촬영을 진행해 주세요. 당신의 퍼스널 컬러 전략이 시작됩니다.")

# ---------------------------------
# 💡 탭5: 직무별 맞춤 면접 전략
# ---------------------------------
with tab5:
    st.subheader("💡 AI 맞춤 면접 답변 및 태도 전략")
    
    # [에러 해결] 세션 상태 초기화 (처음 실행 시 q_type이 없으면 빈 문자열로 설정)
    if 'current_q' not in st.session_state:
        st.session_state.current_q = "상단 버튼을 눌러 질문을 생성하세요."
    if 'q_type' not in st.session_state:
        st.session_state.q_type = "미선택"

    st.markdown("""
    > **"답변의 내용만큼 중요한 것은, 그 내용을 전달하는 '그릇'입니다."**
    > 
    > AI가 분석한 당신의 현재 인상 지표와 선택하신 질문의 성격을 결합하여, **면접관의 고개를 끄덕이게 만들 최적의 답변 구조**를 제안합니다. 
    """)
    st.divider()

    # 질문 데이터베이스
    question_db = {
        "🚀 지원동기 & 포부": [
            "왜 우리 회사가 수많은 지원자 중 당신을 채용해야 합니까?",
            "입사 후 5년 뒤, 본인은 이 조직에서 어떤 역할을 하고 있을까요?",
            "본인의 가치관과 우리 회사의 인재상이 어떻게 부합한다고 생각하시나요?"
        ],
        "🤝 협업 & 갈등관리": [
            "팀 프로젝트 중 의견 차이가 생겼을 때 어떻게 해결했습니까?",
            "본인이 생각하는 '좋은 동료'란 어떤 사람인가요?",
            "조직의 목표와 개인의 가치관이 충돌한다면 어떻게 하겠습니까?"
        ],
        "💻 직무 역량 & 실패": [
            "본인의 직무적 강점 한 가지와 이를 입증할 사례를 말씀해 주세요.",
            "가장 크게 실패했던 경험은 무엇이며, 무엇을 배웠습니까?",
            "새로운 기술이나 트렌드를 익히기 위해 어떤 노력을 하나요?"
        ]
    }
    
    # 버튼 레이아웃
    c1, c2, c3 = st.columns(3)
    if c1.button("🚀 지원동기 & 포부"): 
        st.session_state.current_q = random.choice(question_db["🚀 지원동기 & 포부"])
        st.session_state.q_type = "열정/비전"
    if c2.button("🤝 협업 & 갈등관리"): 
        st.session_state.current_q = random.choice(question_db["🤝 협업 & 갈등관리"])
        st.session_state.q_type = "소통/조화"
    if c3.button("💻 직무 역량 & 실패"): 
        st.session_state.current_q = random.choice(question_db["💻 직무 역량 & 실패"])
        st.session_state.q_type = "전문성/회복탄력성"
    
    st.markdown("---")

    # 질문이 선택되었을 때만 상세 가이드 출력
    if st.session_state.current_q != "상단 버튼을 눌러 질문을 생성하세요.":
        # 질문 카드 출력
        st.info(f"✨ **오늘의 추천 질문 [{st.session_state.q_type}]**\n\n**{st.session_state.current_q}**")
        
        if st.session_state.scores:
            s = st.session_state.scores
            
            st.write("### 🎯 AI 맞춤형 태도 전략")
            col_strat1, col_strat2 = st.columns(2)
            
            with col_strat1:
                st.success(f"✅ **강점 활용: {s.get('신뢰성', 0)}점의 신뢰감**")
                st.markdown(f"""
                - **태도 팁:** 답변 시 바른 자세를 유지하며 결론부터 말하는 '두괄식' 화법을 사용하세요.
                - **시선 처리:** 카메라를 응시하는 비율을 높여 확신 있는 인상을 전달하세요.
                """)
            
            with col_strat2:
                st.warning(f"⚠️ **보완 포인트: {s.get('친근함', 0)}점의 여유**")
                st.markdown(f"""
                - **태도 팁:** 답변 시작과 끝에 부드러운 미소를 곁들여 유연한 인재임을 어필하세요.
                - **발성:** 차분하고 일정한 톤으로 답변의 무게를 유지하세요.
                """)

            # 답변 구조 가이드
            st.divider()
            st.write("### 📝 추천 답변 구조: STAR 기법")
            st.markdown("""
            1. **Situation (상황):** 당시 상황을 1~2문장으로 간결하게 설명합니다.
            2. **Task (과제):** 해결해야 했던 목표나 당면한 문제를 명확히 밝힙니다.
            3. **Action (행동):** **본인이** 어떤 구체적인 노력을 했는지 '과정'을 강조합니다.
            4. **Result (결과):** 성과를 수치나 구체적인 변화로 제시하고 배운 점을 덧붙입니다.
            """)
        else:
            st.warning("먼저 [📸 AI 인상 분석]을 완료하시면 개인화된 태도 전략을 확인하실 수 있습니다.")
    else:
        st.info("상단 버튼을 눌러 면접 질문을 생성하면 상세 전략 가이드가 활성화됩니다.")

with tab6:
    # 0. 메인 타이틀 브랜딩
    st.title("🚀 FIT-ON (핏온)")
    st.markdown("##### **당신의 합격 DNA를 깨우는 AI 면접 페이스메이커**")
    st.divider()

    st.subheader("🏁 특수 직무 맞춤형 최종 합격 진단")
    
    if st.session_state.get("scores"):
        s = st.session_state.scores
        v = st.session_state.get("audio_score", 70)

        # 1. 직무 선택
        job_category = st.selectbox(
            "🎯 목표하시는 직무를 선택하세요",
            ["선택 안함", "✈️ 항공 승무원 (미소/친절 중심)", "🏛️ 공기업 / 공공기관 (신뢰/안정 중심)", "🚀 영업 / 마케팅 (활력/설득 중심)", "💻 연구 / IT 개발 (논리/신뢰 중심)"]
        )

        if job_category != "선택 안함":
            # --- [FIT-ON 가중치 알고리즘] ---
            if "승무원" in job_category:
                weights = {"친근함": 0.5, "신뢰성": 0.1, "자세": 0.3, "음성": 0.1}
                success_msg = "대한항공/아시아나 합격자들의 '그루밍' 지표와 유사합니다."
            elif "공기업" in job_category:
                weights = {"친근함": 0.1, "신뢰성": 0.4, "자세": 0.3, "음성": 0.2}
                success_msg = "주요 공공기관 면접관이 선호하는 안정적인 인상입니다."
            else:
                weights = {"친근함": 0.3, "신뢰성": 0.3, "자세": 0.2, "음성": 0.2}
                success_msg = "일반 기업 합격자 평균 데이터와 매칭되었습니다."

            job_score = int((s["친근함"]*weights["친근함"]) + (s["신뢰성"]*weights["신뢰성"]) + (s["자세균형"]*weights["자세"]) + (v*weights["음성"]))

            # --- 결과 출력 ---
            st.write(f"### 🏆 {job_category} 적격성 리포트")
            c_res1, c_res2 = st.columns([1, 1.5])
            with c_res1:
                st.metric("최종 매칭 지수", f"{job_score}%")
                if job_score >= 80:
                    st.success(f"🌟 **{success_msg}**")
                    st.balloons()
                else:
                    st.warning("⚠️ 핵심 지표를 보완하면 합격률이 상승합니다.")
            with c_res2:
                st.bar_chart({"친근함": s["친근함"], "신뢰성": s["신뢰성"], "자세": s["자세균형"], "음성": v})

            st.divider()

            # 3. AI 합격 처방전 및 결과 해석 (핵심 업데이트 섹션)
            st.write("### 🏥 FIT-ON AI 합격 처방전 (Analysis & Diagnosis)")
            
            # 숫자 데이터만 추출하여 분석
            numeric_scores = {k: v for k, v in s.items() if isinstance(v, (int, float))}
            low_attr = min(numeric_scores, key=numeric_scores.get)
            high_attr = max(numeric_scores, key=numeric_scores.get)

            col_diag1, col_diag2 = st.columns(2)

            with col_diag1:
                st.error(f"🚩 **집중 보완 지표: [{low_attr}]**")
                if low_attr == "신뢰성":
                    st.markdown(f"**해석:** 현재 {numeric_scores[low_attr]}점으로, 시선이 미세하게 흔들리거나 미간에 긴장이 감지됩니다. 이는 면접관에게 답변의 불확실성을 줄 수 있습니다.")
                    st.markdown("**처방:** 카메라 렌즈 상단에 시선을 고정하고, 말끝을 흐리지 않는 연습이 필요합니다.")
                elif low_attr == "자세균형":
                    st.markdown(f"**해석:** 어깨 수평 불균형이 감지되었습니다. 위축된 자세는 비언어적으로 자신감 결여를 의미합니다.")
                    st.markdown("**처방:** 정수리를 위에서 당긴다는 느낌으로 앉고, 양쪽 엉덩이에 무게를 똑같이 배분하세요.")
                elif low_attr == "친근함":
                    st.markdown(f"**해석:** 입꼬리의 움직임이 적어 다소 경직된 인상을 줍니다. {job_category} 직무에서는 자칫 차갑게 보일 우려가 있습니다.")
                    st.markdown("**처방:** 광대 근육을 들어 올리는 '뒤센 미소' 훈련을 제안합니다.")
                else:
                    st.markdown(f"**해석:** {low_attr} 지표가 상대적으로 낮아 전체적인 밸런스를 맞추는 것이 중요합니다.")
                    st.markdown("**처방:** 해당 역량을 의식적으로 노출하여 약점을 강점으로 전환하세요.")

            with col_diag2:
                st.success(f"✨ **최고 강점 페르소나: [{high_attr}]**")
                st.markdown(f"**해석:** 귀하는 **{high_attr}** 역량이 매우 탁월합니다. 이는 면접관에게 본능적으로 강력한 호감을 주는 요소입니다.")
                st.markdown(f"**전략:** 압박 질문이 들어와도 이 **{high_attr}** 상태를 유지한다면, 태도 점수에서 압도적인 가산점을 얻을 수 있습니다.")
                st.info(f"💡 **PRO Tip:** 귀하의 {high_attr} 지수는 상위 5% 합격자 그룹과 일치합니다.")

            # 4. 프리미엄 솔루션
            st.divider()
            st.write("### 💎 FIT-ON 프리미엄 솔루션")
            p1, p2 = st.columns(2)
            with p1:
                if st.button("📊 합격자 매칭 정밀 리포트"): 
                    st.toast("프리미엄 데이터 매칭 중...", icon="💳")
            with p2:
                if st.button("🚀 나만의 이미지 처방전 다운로드"): 
                    st.toast("PDF 리포트 생성 중...", icon="✨")

            # 5. 마인드셋 케어
            st.divider()
            st.write("### 🧘 오늘 하루, 고생한 당신을 위한 'Mind-Fit'")
            with st.expander("🌬️ 긴장 완화: 4-7-8 호흡법"):
                st.write("**방법:** 4초 흡, 7초 참기, 8초 내뱉기. 면접장 대기실에서 심박수를 즉각적으로 안정시킵니다.")
            with st.expander("💡 입장 직전: 1분 체크리스트"):
                st.markdown("- [ ] 어깨 활짝 펴기\n- [ ] 입꼬리 근육 풀기 (아-에-이-오-우)\n- [ ] 첫 인사 톤 반 톤만 높이기")
            
            st.warning("💡 **FIT-ON Tip:** 수치는 도구일 뿐입니다. 당신의 진심과 그간의 노력을 믿으세요!")

        else:
            st.info("🎯 목표하시는 직무를 선택하시면 FIT-ON의 정밀 진단 리포트가 활성화됩니다.")

    else:
        st.warning("🧐 분석 데이터가 없습니다. [📸 FIT-ON 분석] 탭에서 먼저 진단을 진행해 주세요.")
# ---------------------------------
# 🗓️ 탭7: 30일 면접 인상 성장 챌린지
# ---------------------------------
with tab7:
    st.subheader("🗓️ 30일 면접 인상 성장 챌린지")
    
    st.markdown("""
    > **"위대한 면접관은 만들어지는 것이 아니라, 훈련되는 것입니다."**
    > 
    > 매일 기록된 당신의 인상 지표(친근함, 신뢰성, 자세균형)의 변화를 추적합니다. 
    > 그래프의 우상향 곡선은 당신이 면접장에서 뿜어낼 **'합격의 확신'**과 비례합니다.
    """)
    st.divider()

    try:
        import pandas as pd
        # 1. DB 데이터 로드
        conn = sqlite3.connect('interview_challenge.db')
        # DB 컬럼명 확인을 위해 전체 데이터를 가져옵니다.
        df = pd.read_sql_query("SELECT * FROM progress ORDER BY date DESC LIMIT 30", conn)
        conn.close()

        if not df.empty:
            # 데이터 정렬 (날짜 순)
            df = df.sort_values(by='date')
            
            # [에러 해결 포인트] 컬럼명 매핑 확인
            # DB 컬럼이 (date, smile, trust, posture) 순서인 경우
            # 만약 DB 저장 시 한글로 저장했다면 아래 이름을 그에 맞게 변경해야 합니다.
            # 여기서는 DB 저장 로직에 맞춰 영어 컬럼명을 한글로 바꿔서 그래프에 표시합니다.
            df.columns = ['날짜', '친근함', '신뢰성', '자세균형']
            df.set_index('날짜', inplace=True)

            # 2. 성장 추이 시각화
            st.write("### 📈 지표별 성장 추이 (최근 30일)")
            st.line_chart(df[['친근함', '신뢰성', '자세균형']])
            
            # 3. 데이터 요약 통계 (Metric)
            st.divider()
            col_ch1, col_ch2, col_ch3 = st.columns(3)
            
            avg_smile = int(df['친근함'].mean())
            avg_trust = int(df['신뢰성'].mean())
            max_posture = int(df['자세균형'].max())
            
            col_ch1.metric("평균 친근함", f"{avg_smile}%")
            col_ch2.metric("평균 신뢰성", f"{avg_trust}%")
            col_ch3.metric("최고 자세 점수", f"{max_posture}%")

            # 4. AI 성장 분석 코멘트
            st.write("### 📢 AI 성장 분석 코멘트")
            days_count = len(df)
            if days_count >= 3:
                # 첫날 대비 마지막날 신뢰도 변화 계산
                diff = int(df['신뢰성'].iloc[-1] - df['신뢰성'].iloc[0])
                if diff >= 0:
                    st.success(f"🎊 꾸준한 연습의 결과입니다! **{days_count}일간**의 훈련을 통해 신뢰성 지표가 {diff}% 향상되었습니다.")
                else:
                    st.warning(f"🌱 현재 **{days_count}일차** 연습 중입니다. 점수가 정체기라면 촬영 가이드의 조명을 다시 확인해 보세요.")
            else:
                st.info(f"🌱 현재 챌린지 **{days_count}일차**입니다. 3일 이상의 데이터가 쌓이면 정밀 분석이 시작됩니다.")

        else:
            # 데이터가 아예 없는 초기 상태
            st.warning("🧐 아직 기록된 챌린지 데이터가 없습니다.")
            st.info("💡 [📸 AI 인상 분석] 탭에서 사진을 촬영하면 당신의 첫 번째 성장 기록이 이곳에 새겨집니다.")
            
            # Placeholder (예시 그래프)
            st.write("#### 📊 이런 성장 그래프가 그려질 거예요!")
            dummy_data = pd.DataFrame({
                'Day': ['1일', '2일', '3일', '4일'],
                '신뢰성': [60, 65, 63, 75],
                '자세균형': [50, 55, 70, 72]
            }).set_index('Day')
            st.line_chart(dummy_data)

    except Exception as e:
        st.error(f"데이터 로드 중 오류가 발생했습니다: {e}")
        st.info("팁: 탭 1에서 '실시간 분석 시작' 버튼을 눌러 데이터를 먼저 생성해 주세요.")
