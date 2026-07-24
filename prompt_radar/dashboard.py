"""
Интерактивный дашборд «Промпт-радар» на Streamlit.
Редакционный корпоративный дизайн.
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from prompt_radar.engine import build_analytics
from prompt_radar.llm_client import LLMClient


# ── Парсинг файлов ─────────────────────────────────────────────────────

def _find_column(df, candidates):
    for c in candidates:
        if c in df.columns:
            return c
        for col in df.columns:
            if col.lower().strip() == c.lower():
                return col
    return None


def parse_uploaded_file(uploaded_file) -> pd.DataFrame:
    name = uploaded_file.name.lower()
    content = uploaded_file.read()

    if name.endswith(".json"):
        data = json.loads(content)
        if isinstance(data, list):
            df = pd.DataFrame(data)
        elif isinstance(data, dict):
            for key in ["logs", "requests", "data", "records", "entries"]:
                if key in data and isinstance(data[key], list):
                    df = pd.DataFrame(data[key])
                    break
            else:
                df = pd.DataFrame([data])
        else:
            raise ValueError("Неизвестная структура JSON")
    elif name.endswith(".csv"):
        df = pd.read_csv(pd.io.common.BytesIO(content))
    elif name.endswith((".tsv", ".txt")):
        df = pd.read_csv(pd.io.common.BytesIO(content), sep="\t")
    else:
        raise ValueError(f"Неподдерживаемый формат: {name}")

    text_col = _find_column(df, [
        "request_text", "text", "query", "prompt", "message",
        "request", "input", "user_message", "запрос", "текст", "промпт",
    ])
    if text_col is None:
        str_cols = df.select_dtypes(include=["object"]).columns
        if len(str_cols) > 0:
            text_col = max(str_cols, key=lambda c: df[c].astype(str).str.len().mean())
        else:
            raise ValueError(f"Не найдена колонка с текстом. Колонки: {list(df.columns)}")

    rename_map = {}
    if text_col != "request_text":
        rename_map[text_col] = "request_text"

    for candidates, target in [
        (["timestamp", "created_at", "date", "time", "datetime", "дата"], "timestamp"),
        (["user_id", "userId", "user", "author", "client_id", "пользователь"], "user_id"),
        (["token_count", "tokens", "num_tokens", "токены"], "token_count"),
        (["response_time_sec", "response_time", "latency"], "response_time_sec"),
        (["satisfaction_score", "satisfaction", "rating", "score", "оценка"], "satisfaction_score"),
        (["true_category", "category", "label", "категория"], "true_category"),
    ]:
        col = _find_column(df, candidates)
        if col and col != target:
            rename_map[col] = target

    df = df.rename(columns=rename_map)
    df = df.dropna(subset=["request_text"])
    df["request_text"] = df["request_text"].astype(str).str.strip()
    df = df[df["request_text"].str.len() > 10]

    if "id" not in df.columns:
        df["id"] = range(1, len(df) + 1)
    if "timestamp" not in df.columns:
        df["timestamp"] = pd.Timestamp.now()
    if "user_id" not in df.columns:
        df["user_id"] = "unknown"
    if "token_count" not in df.columns:
        df["token_count"] = (df["request_text"].str.len() / 4).astype(int)
    if "response_time_sec" not in df.columns:
        df["response_time_sec"] = 0.0

    return df


# ── Конфигурация ───────────────────────────────────────────────────────

st.set_page_config(
    page_title="Промпт-радар",
    page_icon="radar",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Палитра ────────────────────────────────────────────────────────────

P = {
    "purple": "#4A00B4",
    "purple_light": "#7B2FE0",
    "purple_dark": "#3D00A0",
    "purple_bg": "#F3EDFF",
    "white": "#FFFFFF",
    "bg": "#FAFAF8",
    "text": "#1A1A1A",
    "text_secondary": "#666666",
    "border": "#E0E0E0",
    "success": "#166534",
    "warning": "#92400E",
    "danger": "#991B1B",
    "success_bg": "#F0FDF4",
    "warning_bg": "#FEFCE8",
    "danger_bg": "#FEF2F2",
}

CHART_COLORS = [
    "#4A00B4", "#7B2FE0", "#9B59F0", "#B47FFF",
    "#5C00CC", "#3D00A0", "#2D0080", "#8B5CF6",
    "#A78BFA", "#C4B5FD", "#DDD6FE", "#EDE9FE",
]


# ── Стили ──────────────────────────────────────────────────────────────

st.markdown(f"""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@400;600;700;800;900&family=Inter:wght@400;500;600&display=swap');

    html, body, [class*="css"] {{
        font-family: 'Inter', -apple-system, sans-serif;
        color: {P['text']};
        background: {P['bg']};
    }}

    /* ── Типографика ─────────────────────────────────────────────── */
    h1 {{
        font-family: 'Playfair Display', Georgia, serif !important;
        font-weight: 800 !important;
        font-size: 2rem !important;
        letter-spacing: -0.03em !important;
        color: {P['text']} !important;
        line-height: 1.1 !important;
    }}
    h2 {{
        font-family: 'Playfair Display', Georgia, serif !important;
        font-weight: 700 !important;
        font-size: 1.35rem !important;
        letter-spacing: -0.02em !important;
        color: {P['text']} !important;
        margin-top: 3rem !important;
        margin-bottom: 1.2rem !important;
    }}
    h3 {{
        font-family: 'Playfair Display', Georgia, serif !important;
        font-weight: 600 !important;
        font-size: 1.1rem !important;
        color: {P['text']} !important;
    }}

    /* ── Убираем Streamlit-шум ───────────────────────────────────── */
    #MainMenu {{visibility: hidden;}}
    footer {{visibility: hidden;}}
    header {{visibility: hidden;}}

    /* ── Sidebar ─────────────────────────────────────────────────── */
    [data-testid="stSidebar"] {{
        background: {P['white']};
        border-right: 1px solid {P['border']};
    }}
    [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3 {{
        font-family: 'Inter', sans-serif !important;
        font-size: 0.7rem !important;
        font-weight: 600 !important;
        text-transform: uppercase;
        letter-spacing: 0.1em;
        color: {P['text_secondary']} !important;
        margin-top: 2rem !important;
    }}
    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p {{
        font-size: 0.85rem;
        color: {P['text_secondary']};
    }}

    /* ── Кнопки ──────────────────────────────────────────────────── */
    .stButton > button {{
        border-radius: 0 !important;
        font-weight: 600 !important;
        font-size: 0.8rem !important;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        padding: 10px 28px !important;
        border: 1px solid {P['border']};
        background: {P['white']};
        color: {P['text']};
        transition: all 0.2s;
    }}
    .stButton > button:hover {{
        background: {P['purple']};
        color: {P['white']};
        border-color: {P['purple']};
    }}

    /* ── Табы ────────────────────────────────────────────────────── */
    .stTabs [data-baseweb="tab-list"] {{
        gap: 0;
        border-bottom: 2px solid {P['border']};
        background: transparent;
        padding: 0;
    }}
    .stTabs [data-baseweb="tab"] {{
        border-radius: 0 !important;
        padding: 12px 24px !important;
        font-family: 'Inter', sans-serif;
        font-weight: 600 !important;
        font-size: 0.8rem !important;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        color: {P['text_secondary']};
        background: transparent;
        border: none;
    }}
    .stTabs [aria-selected="true"] {{
        color: {P['purple']} !important;
        border-bottom: 2px solid {P['purple']} !important;
        margin-bottom: -2px;
    }}
    .stTabs [data-baseweb="tab-border"] {{
        display: none;
    }}

    /* ── Экспандеры ──────────────────────────────────────────────── */
    details {{
        background: {P['white']};
        border: 1px solid {P['border']};
        border-radius: 0 !important;
        overflow: hidden;
        margin-bottom: 8px;
    }}
    details summary {{
        font-family: 'Inter', sans-serif;
        font-weight: 500;
        font-size: 0.9rem;
        padding: 14px 20px;
    }}
    details[open] summary {{
        border-bottom: 1px solid {P['border']};
    }}

    /* ── Метрики ─────────────────────────────────────────────────── */
    [data-testid="stMetric"] {{
        background: {P['white']};
        border: 1px solid {P['border']};
        border-radius: 0 !important;
        padding: 20px 24px;
    }}
    [data-testid="stMetric"] label {{
        font-family: 'Inter', sans-serif !important;
        font-size: 0.7rem !important;
        font-weight: 600 !important;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        color: {P['text_secondary']} !important;
    }}
    [data-testid="stMetric"] [data-testid="stMetricValue"] {{
        font-family: 'Playfair Display', serif !important;
        font-size: 1.8rem !important;
        font-weight: 700 !important;
        color: {P['text']} !important;
    }}

    /* ── DataFrame ───────────────────────────────────────────────── */
    [data-testid="stDataFrame"] {{
        border: 1px solid {P['border']};
        border-radius: 0 !important;
    }}

    /* ── Alert-блоки ─────────────────────────────────────────────── */
    [data-testid="stAlert"] {{
        border-radius: 0 !important;
    }}

    /* ── Разделитель ─────────────────────────────────────────────── */
    hr {{
        border: none;
        border-top: 1px solid {P['border']};
        margin: 3rem 0;
    }}

    /* ── Заголовок страницы ──────────────────────────────────────── */
    .hero-header {{
        background: linear-gradient(135deg, {P['purple_dark']}, {P['purple_light']});
        padding: 48px 56px;
        margin: -1rem -1rem 2rem -1rem;
        color: white;
    }}
    .hero-header h1 {{
        color: white !important;
        font-size: 2.2rem !important;
        margin: 0 !important;
    }}
    .hero-header p {{
        color: rgba(255,255,255,0.75);
        font-size: 0.95rem;
        margin: 8px 0 0 0;
        font-family: 'Inter', sans-serif;
    }}

    /* ── Карточки KPI ────────────────────────────────────────────── */
    .kpi-grid {{
        display: grid;
        grid-template-columns: repeat(4, 1fr);
        gap: 1px;
        background: {P['border']};
        border: 1px solid {P['border']};
        margin: 24px 0;
    }}
    .kpi-cell {{
        background: {P['white']};
        padding: 28px 24px;
        text-align: center;
    }}
    .kpi-value {{
        font-family: 'Playfair Display', serif;
        font-size: 2rem;
        font-weight: 700;
        color: {P['text']};
        line-height: 1;
    }}
    .kpi-label {{
        font-family: 'Inter', sans-serif;
        font-size: 0.65rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.1em;
        color: {P['text_secondary']};
        margin-top: 10px;
    }}

    /* ── Карточки сценариев ──────────────────────────────────────── */
    .scenario-card {{
        background: {P['white']};
        border: 1px solid {P['border']};
        padding: 28px 32px;
        margin-bottom: 12px;
    }}
    .scenario-title {{
        font-family: 'Playfair Display', serif;
        font-size: 1.15rem;
        font-weight: 700;
        color: {P['text']};
        margin-bottom: 6px;
    }}
    .scenario-cat {{
        font-size: 0.7rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        color: {P['purple']};
        margin-bottom: 12px;
    }}
    .scenario-meta {{
        display: flex;
        gap: 1px;
        background: {P['border']};
        border: 1px solid {P['border']};
        margin: 16px 0;
    }}
    .scenario-meta-item {{
        background: {P['bg']};
        padding: 10px 16px;
        flex: 1;
        text-align: center;
    }}
    .scenario-meta-val {{
        font-family: 'Playfair Display', serif;
        font-size: 1.2rem;
        font-weight: 700;
        color: {P['text']};
    }}
    .scenario-meta-label {{
        font-size: 0.6rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        color: {P['text_secondary']};
        margin-top: 4px;
    }}

    /* ── Боли и формулировки ─────────────────────────────────────── */
    .pain-item {{
        background: {P['danger_bg']};
        border-left: 3px solid #DC2626;
        padding: 10px 16px;
        margin: 6px 0;
        font-size: 0.85rem;
        color: {P['danger']};
    }}
    .phrase-item {{
        background: {P['success_bg']};
        border-left: 3px solid #16A34A;
        padding: 10px 16px;
        margin: 6px 0;
        font-size: 0.85rem;
        font-style: italic;
        color: {P['success']};
    }}

    /* ── Бейджи ──────────────────────────────────────────────────── */
    .badge {{
        display: inline-block;
        padding: 3px 12px;
        font-size: 0.65rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        border: 1px solid;
    }}
    .badge-high {{ color: {P['success']}; border-color: #BBF7D0; background: {P['success_bg']}; }}
    .badge-medium {{ color: {P['warning']}; border-color: #FDE68A; background: {P['warning_bg']}; }}
    .badge-low {{ color: {P['danger']}; border-color: #FECACA; background: {P['danger_bg']}; }}

    /* ── Рекомендации ────────────────────────────────────────────── */
    .rec-card {{
        background: {P['white']};
        border: 1px solid {P['border']};
        border-left: 3px solid {P['purple']};
        padding: 20px 24px;
        margin-bottom: 8px;
    }}
    .rec-num {{
        display: inline-block;
        width: 22px;
        height: 22px;
        line-height: 22px;
        text-align: center;
        background: {P['purple']};
        color: white;
        font-size: 0.7rem;
        font-weight: 700;
        margin-right: 12px;
        vertical-align: middle;
    }}
    .rec-text {{
        font-size: 0.9rem;
        line-height: 1.5;
        color: {P['text']};
    }}

    /* ── Секции ──────────────────────────────────────────────────── */
    .section-marker {{
        display: inline-block;
        width: 8px;
        height: 8px;
        background: {P['purple']};
        margin-right: 8px;
        vertical-align: middle;
    }}
    .section-label {{
        font-family: 'Inter', sans-serif;
        font-size: 0.7rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.1em;
        color: {P['text_secondary']};
    }}

    /* ── Таблица категорий ───────────────────────────────────────── */
    .cat-row {{
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 14px 0;
        border-bottom: 1px solid {P['border']};
    }}
    .cat-name {{
        font-weight: 500;
        font-size: 0.9rem;
    }}
    .cat-count {{
        font-family: 'Playfair Display', serif;
        font-weight: 700;
        font-size: 1.1rem;
        color: {P['purple']};
    }}
    .cat-pct {{
        font-size: 0.8rem;
        color: {P['text_secondary']};
        margin-left: 8px;
    }}

    /* ── Загрузка ────────────────────────────────────────────────── */
    .upload-zone {{
        border: 2px dashed {P['border']};
        padding: 48px 32px;
        text-align: center;
        background: {P['white']};
    }}
    .upload-zone:hover {{
        border-color: {P['purple']};
    }}

    /* ── Примеры запросов ────────────────────────────────────────── */
    .example-item {{
        padding: 10px 0;
        border-bottom: 1px solid {P['border']};
        font-size: 0.85rem;
        color: {P['text_secondary']};
        line-height: 1.5;
    }}
    .example-num {{
        font-family: 'Playfair Display', serif;
        font-weight: 700;
        color: {P['purple']};
        margin-right: 8px;
    }}

    /* ── Адаптивность ────────────────────────────────────────────── */
    @media (max-width: 768px) {{
        .kpi-grid {{
            grid-template-columns: repeat(2, 1fr);
        }}
        .hero-header {{
            padding: 32px 24px;
        }}
        .hero-header h1 {{
            font-size: 1.6rem !important;
        }}
    }}
</style>
""", unsafe_allow_html=True)


# ── Sidebar ────────────────────────────────────────────────────────────

def render_sidebar():
    with st.sidebar:
        st.markdown(f"""
        <div style="padding:20px 0 24px 0;border-bottom:1px solid {P['border']};margin-bottom:24px">
            <div style="display:flex;align-items:center;gap:10px">
                <div style="width:28px;height:28px;background:{P['purple']};display:flex;
                align-items:center;justify-content:center">
                    <svg width="14" height="14" fill="none" viewBox="0 0 24 24" stroke="white" stroke-width="2.5">
                        <circle cx="12" cy="12" r="10"/>
                        <circle cx="12" cy="12" r="5"/>
                        <circle cx="12" cy="12" r="1.5"/>
                    </svg>
                </div>
                <span style="font-family:'Playfair Display',serif;font-weight:800;font-size:1rem;
                letter-spacing:-0.02em">Промпт-радар</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("### LLM API")
        use_llm = st.toggle("Использовать LLM", value=False)

        llm_client = None
        if use_llm:
            base_url = st.text_input("Base URL", value="https://api.openai.com/v1")
            api_key = st.text_input("API Key", type="password")
            model = st.text_input("Модель", value="gpt-4o-mini")

            if api_key:
                llm_client = LLMClient(base_url=base_url, api_key=api_key, model=model)
                if st.button("Проверить подключение"):
                    with st.spinner():
                        ok, msg = llm_client.test_connection()
                        st.success(msg) if ok else st.error(msg)
            else:
                st.caption("Введите API Key")

        st.markdown("---")
        st.markdown("### Параметры")
        n_clusters = st.slider("Кластеров", 5, 30, 15)

        st.markdown("---")
        st.markdown(f"""
        <div style="padding:16px 0;font-size:0.75rem;color:{P['text_secondary']};line-height:1.6">
            Промпт-радар v1.0<br>
            Аналитика ИИ-агентов
        </div>
        """, unsafe_allow_html=True)

    return llm_client, n_clusters


# ── Загрузка данных ────────────────────────────────────────────────────

def load_data():
    st.markdown(f"""
    <div class="section-label" style="margin-bottom:16px">
        <span class="section-marker"></span>Загрузка данных
    </div>
    """, unsafe_allow_html=True)

    col_upload, col_demo = st.columns([3, 2])
    df = None

    with col_upload:
        uploaded_file = st.file_uploader(
            "CSV или JSON с логами",
            type=["csv", "json", "tsv", "txt"],
        )
        if uploaded_file:
            try:
                df = parse_uploaded_file(uploaded_file)
                st.success(f"Загружено {len(df)} записей")
            except Exception as e:
                st.error(str(e))

    with col_demo:
        data_dir = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data"
        )
        csv_files = (
            [f for f in os.listdir(data_dir) if f.endswith(".csv")]
            if os.path.exists(data_dir) else []
        )
        if csv_files:
            demo_file = st.selectbox("Demo-датасет", csv_files)
            if st.button("Загрузить демо"):
                df = pd.read_csv(os.path.join(data_dir, demo_file))
                df["timestamp"] = pd.to_datetime(df["timestamp"])
                st.success(f"Загружено {len(df)} записей")

    return df


# ── KPI ────────────────────────────────────────────────────────────────

def render_kpi(summary: dict):
    sat = summary["avg_satisfaction"]
    sat_display = f"{sat}/5" if sat else "—"
    mode = "LLM" if summary.get("llm_enabled") else "Keywords"

    st.markdown(f"""
    <div class="kpi-grid">
        <div class="kpi-cell">
            <div class="kpi-value">{summary['total_requests']:,}</div>
            <div class="kpi-label">Запросов</div>
        </div>
        <div class="kpi-cell">
            <div class="kpi-value">{summary['unique_users']}</div>
            <div class="kpi-label">Пользователей</div>
        </div>
        <div class="kpi-cell">
            <div class="kpi-value">{summary['avg_tokens']:,}</div>
            <div class="kpi-label">Ср. токенов</div>
        </div>
        <div class="kpi-cell">
            <div class="kpi-value">{sat_display}</div>
            <div class="kpi-label">Удовлетворённость</div>
        </div>
    </div>
    <div class="kpi-grid">
        <div class="kpi-cell">
            <div class="kpi-value">{summary['num_categories']}</div>
            <div class="kpi-label">Категорий</div>
        </div>
        <div class="kpi-cell">
            <div class="kpi-value">{summary['num_use_cases']}</div>
            <div class="kpi-label">Сценариев</div>
        </div>
        <div class="kpi-cell">
            <div class="kpi-value">{summary['avg_response_time']}с</div>
            <div class="kpi-label">Ср. время ответа</div>
        </div>
        <div class="kpi-cell">
            <div class="kpi-value">{mode}</div>
            <div class="kpi-label">Режим анализа</div>
        </div>
    </div>
    """, unsafe_allow_html=True)


# ── Tab 1: Категории ──────────────────────────────────────────────────

def tab_categories(analytics: dict):
    cat_stats = analytics["category_stats"]
    cat_df = pd.DataFrame({
        "Категория": list(cat_stats.keys()),
        "Количество": list(cat_stats.values()),
    }).sort_values("Количество", ascending=False)
    total = cat_df["Количество"].sum()

    col_chart, col_list = st.columns([3, 2])

    with col_chart:
        fig = px.pie(
            cat_df, values="Количество", names="Категория",
            hole=0.55,
            color_discrete_sequence=CHART_COLORS,
        )
        fig.update_traces(
            textinfo="percent",
            textfont_size=11,
            marker=dict(line=dict(color=P["white"], width=2)),
        )
        fig.update_layout(
            height=400, margin=dict(t=20, b=20, l=20, r=20),
            showlegend=False,
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(family="Inter, sans-serif", size=12, color=P["text_secondary"]),
        )
        st.plotly_chart(fig, use_container_width=True)

    with col_list:
        for _, row in cat_df.iterrows():
            pct = round(row["Количество"] / total * 100, 1)
            st.markdown(f"""
            <div class="cat-row">
                <span class="cat-name">{row['Категория']}</span>
                <span>
                    <span class="cat-count">{row['Количество']}</span>
                    <span class="cat-pct">{pct}%</span>
                </span>
            </div>
            """, unsafe_allow_html=True)

    st.markdown(f"""
    <div class="section-label" style="margin:32px 0 16px 0">
        <span class="section-marker"></span>Сценарии по категориям
    </div>
    """, unsafe_allow_html=True)

    for cat, sc_list in analytics["scenarios_by_cat"].items():
        with st.expander(f"{cat}  —  {len(sc_list)} сценариев"):
            for sc in sorted(sc_list, key=lambda x: x["size"], reverse=True):
                terms = ", ".join(sc["top_terms"][:5])
                name = sc.get("name", f"Сценарий #{sc['id']}")
                st.markdown(f"**{name}**  ·  {sc['size']} запросов")
                st.caption(terms)


# ── Tab 2: Сценарии ───────────────────────────────────────────────────

def tab_scenarios(analytics: dict):
    col_h, col_s = st.columns([4, 1])
    with col_h:
        st.markdown(f"""
        <div class="section-label" style="margin-bottom:16px">
            <span class="section-marker"></span>Выделенные сценарии
        </div>
        """, unsafe_allow_html=True)
    with col_s:
        sort_by = st.selectbox(
            "Сортировка", ["size", "avg_satisfaction", "avg_response_time"],
            format_func=lambda x: {"size": "Объём", "avg_satisfaction": "Оценка", "avg_response_time": "Скорость"}[x],
            label_visibility="collapsed",
        )

    use_cases = analytics["use_cases"]
    sorted_cases = sorted(
        use_cases.items(),
        key=lambda x: x[1].get(sort_by, 0) or 0,
        reverse=True,
    )

    for uc_id, uc in sorted_cases:
        sat = uc.get("avg_satisfaction")
        sat_str = f"{sat}/5" if sat else "—"
        name = uc.get("name", f"Сценарий #{uc_id}")
        cat = uc["dominant_category"]
        size = uc["size"]
        auto = uc.get("automation_potential", "")
        auto_cls = {"high": "badge-high", "medium": "badge-medium", "low": "badge-low"}.get(auto, "")
        auto_badge = f'<span class="badge {auto_cls}">{auto}</span>' if auto else ""

        with st.expander(f"{name}  ·  {cat}  ·  {size} запросов  ·  {sat_str}"):
            st.markdown(f"""
            <div class="scenario-cat">{cat} {auto_badge}</div>
            """, unsafe_allow_html=True)

            desc = uc.get("description", uc.get("summary", ""))
            if desc:
                st.markdown(f'<p style="font-size:0.9rem;line-height:1.6;color:{P["text_secondary"]}">{desc}</p>',
                    unsafe_allow_html=True)

            # Метрики
            st.markdown(f"""
            <div class="scenario-meta">
                <div class="scenario-meta-item">
                    <div class="scenario-meta-val">{size}</div>
                    <div class="scenario-meta-label">Запросов</div>
                </div>
                <div class="scenario-meta-item">
                    <div class="scenario-meta-val">{uc.get("users", 0)}</div>
                    <div class="scenario-meta-label">Пользователей</div>
                </div>
                <div class="scenario-meta-item">
                    <div class="scenario-meta-val">{uc.get("avg_tokens", 0):,}</div>
                    <div class="scenario-meta-label">Ср. токенов</div>
                </div>
                <div class="scenario-meta-item">
                    <div class="scenario-meta-val">{sat_str}</div>
                    <div class="scenario-meta-label">Оценка</div>
                </div>
            </div>
            """, unsafe_allow_html=True)

            # Боли и формулировки
            col_l, col_r = st.columns(2)
            with col_l:
                pain_points = uc.get("pain_points", [])
                if pain_points:
                    st.markdown(f'<div class="section-label" style="margin-bottom:8px">Боли пользователей</div>',
                        unsafe_allow_html=True)
                    for p in pain_points:
                        st.markdown(f'<div class="pain-item">{p}</div>', unsafe_allow_html=True)

                rec = uc.get("recommendation", "")
                if rec:
                    st.markdown(f'<div class="section-label" style="margin:16px 0 8px">Рекомендация</div>',
                        unsafe_allow_html=True)
                    st.info(rec)

            with col_r:
                typical = uc.get("typical_phrases", [])
                if typical:
                    st.markdown(f'<div class="section-label" style="margin-bottom:8px">Типовые формулировки</div>',
                        unsafe_allow_html=True)
                    for t in typical:
                        st.markdown(f'<div class="phrase-item">"{t}"</div>', unsafe_allow_html=True)

            # Термины и примеры
            terms = ", ".join(uc.get("top_terms", []))
            st.markdown(f"""
            <div class="section-label" style="margin:20px 0 6px">Ключевые термины</div>
            <p style="font-size:0.85rem;color:{P['text_secondary']}">{terms}</p>
            """, unsafe_allow_html=True)

            examples = uc.get("examples", [])
            if examples:
                st.markdown(f'<div class="section-label" style="margin:16px 0 8px">Примеры запросов</div>',
                    unsafe_allow_html=True)
                for i, ex in enumerate(examples[:3], 1):
                    st.markdown(f'<div class="example-item"><span class="example-num">{i}.</span>{ex}...</div>',
                        unsafe_allow_html=True)


# ── Tab 3: Тренды ─────────────────────────────────────────────────────

def tab_trends(analytics: dict):
    timeline = pd.DataFrame(analytics["timeline"])

    if not timeline.empty:
        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=("Количество запросов", "Средний размер (токены)"),
            vertical_spacing=0.2,
        )
        fig.add_trace(go.Scatter(
            x=timeline["date"], y=timeline["requests"],
            mode="lines+markers", name="Запросы",
            line=dict(color=P["purple"], width=2),
            marker=dict(size=4),
            fill="tozeroy", fillcolor="rgba(74,0,180,0.04)",
        ), row=1, col=1)
        fig.add_trace(go.Scatter(
            x=timeline["date"], y=timeline["avg_tokens"],
            mode="lines+markers", name="Токены",
            line=dict(color=P["purple_light"], width=2),
            marker=dict(size=4),
            fill="tozeroy", fillcolor="rgba(123,47,224,0.04)",
        ), row=2, col=1)
        fig.update_layout(
            height=460, showlegend=False,
            margin=dict(t=50, b=20, l=50, r=20),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(family="Inter, sans-serif", size=11, color=P["text_secondary"]),
        )
        fig.update_xaxes(gridcolor="#F0F0F0", zeroline=False)
        fig.update_yaxes(gridcolor="#F0F0F0", zeroline=False)
        fig.update_annotations(font=dict(family="Inter, sans-serif", size=12, color=P["text"]))
        st.plotly_chart(fig, use_container_width=True)

    # Тепловая карта
    raw_df = analytics["raw_df"]
    if "timestamp" in raw_df.columns:
        raw_df = raw_df.copy()
        raw_df["hour"] = raw_df["timestamp"].dt.hour
        raw_df["weekday"] = raw_df["timestamp"].dt.day_name()

        heatmap_data = raw_df.groupby(["weekday", "hour"]).size().reset_index(name="count")
        weekday_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
        heatmap_data["weekday"] = pd.Categorical(heatmap_data["weekday"], categories=weekday_order, ordered=True)
        heatmap_pivot = heatmap_data.pivot(index="weekday", columns="hour", values="count").fillna(0)

        fig_heat = px.imshow(
            heatmap_pivot,
            labels=dict(x="Час", y="День недели", color="Запросов"),
            color_continuous_scale=[
                [0, "#FAFAF8"],
                [0.25, "#EDE9FE"],
                [0.5, "#C4B5FD"],
                [0.75, "#7B2FE0"],
                [1, "#4A00B4"],
            ],
            aspect="auto",
        )
        fig_heat.update_layout(
            height=300, margin=dict(t=50, b=20, l=20, r=20),
            paper_bgcolor="rgba(0,0,0,0)",
            font=dict(family="Inter, sans-serif", size=11, color=P["text_secondary"]),
        )
        st.markdown(f"""
        <div class="section-label" style="margin:32px 0 8px">
            <span class="section-marker"></span>Тепловая карта активности
        </div>
        """, unsafe_allow_html=True)
        st.plotly_chart(fig_heat, use_container_width=True)


# ── Tab 4: Пользователи ───────────────────────────────────────────────

def tab_users(analytics: dict):
    top_users = analytics["top_users"]
    if top_users:
        users_df = pd.DataFrame({
            "Пользователь": list(top_users.keys()),
            "Запросов": list(top_users.values()),
        }).sort_values("Запросов", ascending=True).tail(10)

        fig = px.bar(
            users_df, x="Запросов", y="Пользователь",
            orientation="h",
            color="Запросов",
            color_continuous_scale=[[0, "#EDE9FE"], [1, "#4A00B4"]],
        )
        fig.update_layout(
            height=360, margin=dict(t=20, b=20, l=20, r=20),
            showlegend=False,
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(family="Inter, sans-serif", size=11, color=P["text_secondary"]),
        )
        fig.update_xaxes(gridcolor="#F0F0F0", zeroline=False)
        fig.update_yaxes(gridcolor="#F0F0F0", zeroline=False)
        st.plotly_chart(fig, use_container_width=True)

    raw_df = analytics["raw_df"]
    if "satisfaction_score" in raw_df.columns:
        sat_values = pd.to_numeric(raw_df["satisfaction_score"], errors="coerce").dropna()
        if len(sat_values) > 0:
            st.markdown(f"""
            <div class="section-label" style="margin:32px 0 8px">
                <span class="section-marker"></span>Распределение оценок
            </div>
            """, unsafe_allow_html=True)
            fig_sat = px.histogram(
                x=sat_values, nbins=5,
                labels={"x": "Оценка", "y": "Количество"},
                color_discrete_sequence=[P["purple"]],
            )
            fig_sat.update_traces(marker=dict(line=dict(color=P["white"], width=1)))
            fig_sat.update_layout(
                height=260, margin=dict(t=20, b=20),
                bargap=0.2,
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font=dict(family="Inter, sans-serif", size=11, color=P["text_secondary"]),
            )
            fig_sat.update_xaxes(gridcolor="#F0F0F0", zeroline=False)
            fig_sat.update_yaxes(gridcolor="#F0F0F0", zeroline=False)
            st.plotly_chart(fig_sat, use_container_width=True)


# ── Рекомендации ───────────────────────────────────────────────────────

def render_recommendations(analytics: dict):
    st.markdown("---")
    st.markdown(f"""
    <div class="section-label" style="margin-bottom:20px">
        <span class="section-marker"></span>Рекомендации для CTO
    </div>
    """, unsafe_allow_html=True)

    if analytics.get("llm_insights"):
        st.markdown(analytics["llm_insights"])
        st.markdown("---")

    summary = analytics["summary"]
    recs = []

    if analytics["category_stats"]:
        top_cat = max(analytics["category_stats"], key=analytics["category_stats"].get)
        pct = analytics["category_stats"][top_cat] / summary["total_requests"] * 100
        recs.append(f"**Топ-категория:** {top_cat} — {pct:.1f}% запросов. Приоритизируйте автоматизацию.")

    low_sat = [uc for uc in analytics["use_cases"].values()
               if uc.get("avg_satisfaction") is not None and uc["avg_satisfaction"] < 3.0]
    if low_sat:
        recs.append(f"**Проблемные зоны:** {len(low_sat)} сценариев с оценкой ниже 3/5.")

    slow = [uc for uc in analytics["use_cases"].values()
            if uc.get("avg_response_time", 0) > 30]
    if slow:
        recs.append(f"**Производительность:** {len(slow)} сценариев с ответом >30с.")

    high_auto = [uc for uc in analytics["use_cases"].values()
                 if uc.get("automation_potential") == "high"]
    if high_auto:
        names = [uc.get("name", "") for uc in high_auto[:3]]
        recs.append(f"**Автоматизация:** {', '.join(names)}. Высокий потенциал.")

    for i, rec in enumerate(recs, 1):
        st.markdown(f"""
        <div class="rec-card">
            <span class="rec-num">{i}</span>
            <span class="rec-text">{rec}</span>
        </div>
        """, unsafe_allow_html=True)


# ── Главная ────────────────────────────────────────────────────────────

def main():
    st.markdown(f"""
    <div class="hero-header">
        <h1>Промпт-радар</h1>
        <p>Аналитика запросов к ИИ-агентам  ·  Классификация  ·  Сценарии  ·  Рекомендации</p>
    </div>
    """, unsafe_allow_html=True)

    llm_client, n_clusters = render_sidebar()
    df = load_data()

    if df is None:
        st.markdown(f"""
        <div style="text-align:center;padding:100px 40px;color:{P['text_secondary']}">
            <div style="font-family:'Playfair Display',serif;font-size:1.3rem;font-weight:700;
            color:{P['text']};margin-bottom:8px">Загрузите данные</div>
            <p style="font-size:0.9rem">CSV или JSON с логами запросов к ИИ-агенту</p>
        </div>
        """, unsafe_allow_html=True)
        st.stop()
        return

    with st.expander("Предпросмотр данных"):
        st.dataframe(df.head(10), use_container_width=True)

    mode_str = "LLM" if llm_client else "ключевые слова"
    with st.spinner(f"Анализ {len(df)} запросов ({mode_str})..."):
        analytics = build_analytics(df, llm=llm_client, n_clusters=n_clusters)

    render_kpi(analytics["summary"])

    tab1, tab2, tab3, tab4 = st.tabs(["Категории", "Сценарии", "Тренды", "Пользователи"])

    with tab1:
        tab_categories(analytics)
    with tab2:
        tab_scenarios(analytics)
    with tab3:
        tab_trends(analytics)
    with tab4:
        tab_users(analytics)

    render_recommendations(analytics)


if __name__ == "__main__":
    main()
