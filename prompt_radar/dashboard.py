"""
Интерактивный дашборд «Промпт-радар» на Streamlit.
Поддерживает загрузку файлов и LLM-аналитику через API.
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


# ── Парсинг загруженных файлов (inline для отказоустойчивости) ─────────

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

    # Нормализация колонок
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


st.set_page_config(
    page_title="Промпт-радар | Аналитика ИИ-агентов",
    page_icon="📡",
    layout="wide",
)

st.markdown("""
<style>
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 20px;
        border-radius: 12px;
        color: white;
        text-align: center;
    }
    .metric-value { font-size: 2em; font-weight: bold; }
    .metric-label { font-size: 0.9em; opacity: 0.8; }
    .upload-area {
        border: 2px dashed #667eea;
        border-radius: 12px;
        padding: 40px;
        text-align: center;
        margin: 20px 0;
    }
    .insight-box {
        background: #f8f9fa;
        border-left: 4px solid #667eea;
        padding: 16px;
        margin: 8px 0;
        border-radius: 0 8px 8px 0;
    }
    .pain-tag {
        background: #ffeaa7;
        padding: 2px 8px;
        border-radius: 4px;
        font-size: 0.85em;
        margin: 2px;
        display: inline-block;
    }
    .auto-high { color: #00b894; font-weight: bold; }
    .auto-medium { color: #fdcb6e; font-weight: bold; }
    .auto-low { color: #d63031; font-weight: bold; }
</style>
""", unsafe_allow_html=True)


# ── Боковая панель: настройки ──────────────────────────────────────────

def render_sidebar():
    """Рендерит боковую панель с настройками LLM и возврывает конфигурацию."""
    st.sidebar.markdown("## ⚙️ Настройки")

    # ── LLM API ────────────────────────────────────────────────────────
    st.sidebar.markdown("### 🤖 LLM API (опционально)")
    use_llm = st.sidebar.toggle("Использовать LLM для аналитики", value=False)

    llm_client = None
    if use_llm:
        base_url = st.sidebar.text_input(
            "Base URL API",
            value="https://api.openai.com/v1",
            help="OpenAI, vLLM, Ollama, LM Studio и др.",
        )
        api_key = st.sidebar.text_input(
            "API Key",
            type="password",
            help="Ключ API (sk-... для OpenAI)",
        )
        model = st.sidebar.text_input(
            "Модель",
            value="gpt-4o-mini",
            help="Название модели (gpt-4o-mini, llama3, mistral и т.д.)",
        )

        if api_key:
            llm_client = LLMClient(
                base_url=base_url,
                api_key=api_key,
                model=model,
            )
            if st.sidebar.button("🔌 Проверить подключение"):
                with st.spinner("Проверяю..."):
                    ok, msg = llm_client.test_connection()
                    if ok:
                        st.sidebar.success(msg)
                    else:
                        st.sidebar.error(f"Ошибка: {msg}")
        else:
            st.sidebar.warning("Введите API Key для активации LLM")

    # ── Настройки кластеризации ────────────────────────────────────────
    st.sidebar.markdown("### 📊 Параметры анализа")
    n_clusters = st.sidebar.slider(
        "Число кластеров (сценариев)",
        min_value=5, max_value=30, value=15,
        help="Больше кластеров = более детальные сценарии",
    )

    # ── Информация ─────────────────────────────────────────────────────
    st.sidebar.markdown("---")
    st.sidebar.markdown("### ℹ️ О проекте")
    st.sidebar.info(
        "**Промпт-радар** автоматически структурирует логи запросов к ИИ-агентам:\n"
        "- Классифицирует по категориям\n"
        "- Находит сценарии (use-cases)\n"
        "- Генерирует саммари\n"
        "- Строит дашборд с рекомендациями"
    )

    return llm_client, n_clusters


# ── Загрузка данных ────────────────────────────────────────────────────

def load_data():
    """Загружает данные: через файл или из demo-датасета."""
    st.markdown("## 📂 Загрузка данных")

    col_upload, col_demo = st.columns([2, 1])

    df = None
    source_name = None

    with col_upload:
        uploaded_file = st.file_uploader(
            "Загрузите CSV или JSON с логами запросов",
            type=["csv", "json", "tsv", "txt"],
            help="Файл должен содержать колонку с текстом запроса. "
                 "Опционально: timestamp, user_id, token_count, satisfaction_score",
        )
        if uploaded_file:
            try:
                df = parse_uploaded_file(uploaded_file)
                source_name = uploaded_file.name
                st.success(f"Загружено {len(df)} записей из {uploaded_file.name}")
            except Exception as e:
                st.error(f"Ошибка парсинга: {e}")

    with col_demo:
        st.markdown("#### Или используйте демо-датасет")
        data_dir = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data"
        )
        csv_files = (
            [f for f in os.listdir(data_dir) if f.endswith(".csv")]
            if os.path.exists(data_dir) else []
        )
        if csv_files:
            demo_file = st.selectbox("Demo-файл", csv_files)
            if st.button("📊 Загрузить демо-данные"):
                df = pd.read_csv(os.path.join(data_dir, demo_file))
                df["timestamp"] = pd.to_datetime(df["timestamp"])
                source_name = demo_file
                st.success(f"Загружено {len(df)} записей из {demo_file}")

    # Подсказка по формату
    with st.expander("📋 Поддерживаемые форматы данных"):
        st.markdown("""
        **Минимальный формат** — CSV/JSON с одной колонкой текста запросов.

        | Колонка | Обязательна | Описание |
        |---------|-------------|----------|
        | request_text / text / query / prompt | Да | Текст запроса пользователя |
        | timestamp / created_at | Нет | Дата/время (генерируется если нет) |
        | user_id / user | Нет | ID пользователя |
        | token_count / tokens | Нет | Кол-во токенов (оценка если нет) |
        | response_time_sec | Нет | Время ответа в секундах |
        | satisfaction_score / rating | Нет | Оценка 1-5 |
        | category / true_category | Нет | Истинная категория (для валидации) |

        **JSON** — массив объектов или объект с ключом `logs`/`requests`/`data`.
        """)

    return df, source_name


# ── KPI карточки ───────────────────────────────────────────────────────

def render_kpi(summary: dict):
    """Рендерит блок KPI-карточек."""
    st.markdown("## 📊 Обзор")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Всего запросов", f"{summary['total_requests']:,}")
    with col2:
        st.metric("Уникальных пользователей", summary["unique_users"])
    with col3:
        st.metric("Средний размер (токены)", f"{summary['avg_tokens']:,}")
    with col4:
        sat = summary["avg_satisfaction"]
        st.metric("Удовлетворённость", f"{sat}/5" if sat else "N/A")

    col5, col6, col7, col8 = st.columns(4)
    with col5:
        st.metric("Категорий", summary["num_categories"])
    with col6:
        st.metric("Сценариев", summary["num_use_cases"])
    with col7:
        st.metric("Ср. время ответа", f"{summary['avg_response_time']}с")
    with col8:
        mode = "🤖 LLM" if summary.get("llm_enabled") else "🔤 Keywords"
        st.metric("Режим анализа", mode)
    st.markdown("---")


# ── Tab 1: Категории ──────────────────────────────────────────────────

def tab_categories(analytics: dict):
    st.markdown("### Распределение по категориям")
    cat_stats = analytics["category_stats"]
    cat_df = pd.DataFrame({
        "Категория": list(cat_stats.keys()),
        "Количество": list(cat_stats.values()),
    }).sort_values("Количество", ascending=False)

    col_chart, col_table = st.columns([2, 1])
    with col_chart:
        fig = px.pie(
            cat_df, values="Количество", names="Категория",
            hole=0.4, color_discrete_sequence=px.colors.qualitative.Set2,
        )
        fig.update_layout(height=450, margin=dict(t=20, b=20))
        st.plotly_chart(fig, use_container_width=True)
    with col_table:
        cat_df["Доля %"] = (cat_df["Количество"] / cat_df["Количество"].sum() * 100).round(1)
        st.dataframe(cat_df, use_container_width=True, hide_index=True)

    st.markdown("### Сценарии по категориям")
    for cat, sc_list in analytics["scenarios_by_cat"].items():
        with st.expander(f"{cat} ({len(sc_list)} сценариев)"):
            for sc in sorted(sc_list, key=lambda x: x["size"], reverse=True):
                terms = ", ".join(sc["top_terms"][:5])
                name = sc.get("name", "")
                label = f"**{name}**" if name else f"**Сценарий #{sc['id']}**"
                st.markdown(f"{label} — {sc['size']} запросов")
                st.caption(f"Ключевые слова: {terms}")


# ── Tab 2: Сценарии ───────────────────────────────────────────────────

def tab_scenarios(analytics: dict):
    st.markdown("### Выделенные сценарии (use-cases)")

    sort_by = st.selectbox(
        "Сортировать по", ["size", "avg_satisfaction", "avg_response_time"],
        format_func=lambda x: {
            "size": "Кол-во запросов",
            "avg_satisfaction": "Удовлетворённость",
            "avg_response_time": "Время ответа",
        }[x],
    )

    use_cases = analytics["use_cases"]
    sorted_cases = sorted(
        use_cases.items(),
        key=lambda x: x[1].get(sort_by, 0) or 0,
        reverse=True,
    )

    for uc_id, uc in sorted_cases:
        sat_display = f"{uc['avg_satisfaction']}/5" if uc.get('avg_satisfaction') else "—"
        name = uc.get("name", f"Сценарий #{uc_id}")
        auto = uc.get("automation_potential", "")

        auto_icon = {"high": "🟢", "medium": "🟡", "low": "🔴"}.get(auto, "⚪")
        header = f"🎯 {name} | {uc['dominant_category']} | {uc['size']} запросов | ⭐ {sat_display}"

        with st.expander(header):
            # Описание
            desc = uc.get("description", uc.get("summary", ""))
            st.markdown(f"**Описание:** {desc}")

            # Боли и потенциал автоматизации
            col_a, col_b = st.columns(2)
            with col_a:
                pain_points = uc.get("pain_points", [])
                if pain_points:
                    st.markdown("**Боли пользователей:**")
                    for p in pain_points:
                        st.markdown(f"- {p}")

                auto = uc.get("automation_potential", "")
                if auto:
                    st.markdown(f"**Потенциал автоматизации:** {auto_icon} {auto}")

            with col_b:
                typical = uc.get("typical_phrases", [])
                if typical:
                    st.markdown("**Типовые формулировки:**")
                    for t in typical:
                        st.markdown(f'- _{t}_')

                rec = uc.get("recommendation", "")
                if rec:
                    st.markdown(f"**Рекомендация:** {rec}")

            # Метрики
            m1, m2, m3 = st.columns(3)
            m1.metric("Запросов", uc["size"])
            m2.metric("Пользователей", uc.get("users", 0))
            m3.metric("Ср. токены", f"{uc.get('avg_tokens', 0):,}")

            # Ключевые термины и примеры
            st.markdown("**Ключевые термины:** " + ", ".join(uc.get("top_terms", [])))

            st.markdown("**Примеры запросов:**")
            for i, ex in enumerate(uc.get("examples", [])[:3], 1):
                st.markdown(f"{i}. *{ex}...*")


# ── Tab 3: Тренды ─────────────────────────────────────────────────────

def tab_trends(analytics: dict):
    st.markdown("### Динамика по времени")
    timeline = pd.DataFrame(analytics["timeline"])

    if not timeline.empty:
        fig_timeline = make_subplots(
            rows=2, cols=1,
            subplot_titles=("Количество запросов", "Средний размер (токены)"),
            vertical_spacing=0.15,
        )
        fig_timeline.add_trace(
            go.Scatter(
                x=timeline["date"], y=timeline["requests"],
                mode="lines+markers", name="Запросы",
                line=dict(color="#667eea", width=2),
            ), row=1, col=1,
        )
        fig_timeline.add_trace(
            go.Scatter(
                x=timeline["date"], y=timeline["avg_tokens"],
                mode="lines+markers", name="Ср. токены",
                line=dict(color="#f093fb", width=2),
            ), row=2, col=1,
        )
        fig_timeline.update_layout(height=500, showlegend=False, margin=dict(t=40, b=20))
        st.plotly_chart(fig_timeline, use_container_width=True)

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
            color_continuous_scale="Viridis", aspect="auto",
        )
        fig_heat.update_layout(height=350, margin=dict(t=40, b=20))
        st.markdown("### Тепловая карта активности")
        st.plotly_chart(fig_heat, use_container_width=True)


# ── Tab 4: Пользователи ───────────────────────────────────────────────

def tab_users(analytics: dict):
    st.markdown("### Топ активных пользователей")
    top_users = analytics["top_users"]
    if top_users:
        users_df = pd.DataFrame({
            "Пользователь": list(top_users.keys()),
            "Запросов": list(top_users.values()),
        }).sort_values("Запросов", ascending=True)
        fig_users = px.bar(
            users_df, x="Запросов", y="Пользователь",
            orientation="h", color="Запросов",
            color_continuous_scale="Viridis",
        )
        fig_users.update_layout(height=400, margin=dict(t=20, b=20))
        st.plotly_chart(fig_users, use_container_width=True)

    raw_df = analytics["raw_df"]
    if "satisfaction_score" in raw_df.columns:
        sat_values = pd.to_numeric(raw_df["satisfaction_score"], errors="coerce").dropna()
        if len(sat_values) > 0:
            st.markdown("### Распределение оценок удовлетворённости")
            fig_sat = px.histogram(
                x=sat_values, nbins=5,
                labels={"x": "Оценка", "y": "Количество"},
                color_discrete_sequence=["#667eea"],
            )
            fig_sat.update_layout(height=300, margin=dict(t=20, b=20))
            st.plotly_chart(fig_sat, use_container_width=True)


# ── Рекомендации CTO ──────────────────────────────────────────────────

def render_recommendations(analytics: dict):
    st.markdown("---")
    st.markdown("## 💡 Рекомендации для CTO")

    # LLM-инсайты (если есть)
    if analytics.get("llm_insights"):
        st.markdown("### AI-аналитика")
        st.markdown(analytics["llm_insights"])
        st.markdown("---")

    summary = analytics["summary"]
    recommendations = []

    if analytics["category_stats"]:
        top_cat = max(analytics["category_stats"], key=analytics["category_stats"].get)
        top_cat_pct = analytics["category_stats"][top_cat] / summary["total_requests"] * 100
        recommendations.append(
            f"**Топ-категория:** «{top_cat}» — {top_cat_pct:.1f}% всех запросов. "
            f"Стоит приоритизировать автоматизацию и улучшение инструментов."
        )

    low_sat = [
        (uid, uc) for uid, uc in analytics["use_cases"].items()
        if uc.get("avg_satisfaction") is not None and uc["avg_satisfaction"] < 3.0
    ]
    if low_sat:
        recommendations.append(
            f"**Проблемные зоны:** {len(low_sat)} сценариев с удовлетворённостью ниже 3/5. "
            f"Требуется анализ качества ответов."
        )

    slow_cases = [
        (uid, uc) for uid, uc in analytics["use_cases"].items()
        if uc.get("avg_response_time", 0) > 30
    ]
    if slow_cases:
        recommendations.append(
            f"**Производительность:** {len(slow_cases)} сценариев со средним временем ответа >30с."
        )

    # Автоматизация
    high_auto = [
        uc for uc in analytics["use_cases"].values()
        if uc.get("automation_potential") == "high"
    ]
    if high_auto:
        names = [uc.get("name", "") for uc in high_auto[:3]]
        recommendations.append(
            f"**Высокий потенциал автоматизации:** {', '.join(names)}. "
            f"Рекомендуется приоритизировать автоматизацию этих сценариев."
        )

    for rec in recommendations:
        st.markdown(f"- {rec}")


# ── Главная функция ────────────────────────────────────────────────────

def main():
    st.title("📡 Промпт-радар")
    st.caption("Аналитика запросов пользователей к ИИ-агентам | Upload → Classify → Cluster → Report")

    llm_client, n_clusters = render_sidebar()
    df, source_name = load_data()

    if df is None:
        st.info("⬆️ Загрузите файл с логами или выберите демо-датасет для начала анализа")
        st.stop()
        return

    # Предпросмотр данных
    with st.expander("👀 Предпросмотр данных", expanded=False):
        st.dataframe(df.head(10), use_container_width=True)
        st.caption(f"Колонки: {', '.join(df.columns)}")

    # Запуск анализа
    with st.spinner(f"Анализирую {len(df)} запросов..." + (" (LLM-режим)" if llm_client else " (режим ключевых слов)")):
        analytics = build_analytics(df, llm=llm_client, n_clusters=n_clusters)

    render_kpi(analytics["summary"])

    tab1, tab2, tab3, tab4 = st.tabs([
        "📂 Категории", "🔍 Сценарии", "📈 Тренды", "👥 Пользователи"
    ])

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
