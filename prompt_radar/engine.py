"""
Ядро «Промпт-радара»: классификация, кластеризация, саммаризация.
Поддерживает два режима: LLM (через API) и fallback (TF-IDF + ключевые слова).
"""

import json
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from collections import Counter, defaultdict
from typing import Optional

from prompt_radar.llm_client import LLMClient


# ── Предопределённые категории ─────────────────────────────────────────

DEFAULT_CATEGORIES = [
    "Генерация текста",
    "Написание кода",
    "Анализ данных",
    "Работа с документами",
    "Обучение и онбординг",
    "Работа с клиентами",
    "Прочее",
]

CATEGORY_KEYWORDS = {
    "Генерация текста": [
        "напиши", "составь", "создай текст", "письмо", "пост", "рассылка",
        "слоган", "текст", "отчёт", "резюме", "саммари", "выжимка",
        "продающий", "рекламный", "деловое письмо", "благодарност",
    ],
    "Написание кода": [
        "скрипт", "код", "программа", "напиши на python", "api", "сервис",
        "функция", "класс", "модуль", "тест", "рефакторинг", "оптимизируй",
        "docker", "ci/cd", "middleware", "endpoint", "component", "websocket",
        "code review", "unit-тест", "баг", "memory leak",
    ],
    "Анализ данных": [
        "анализ", "данные", "график", "дашборд", "метрик", "корреляц",
        "прогноз", "сегментаци", "воронк", "a/b", "аномали", "тренд",
        "визуализ", "scatter", "тепловая карта", "когорт", "rfm",
        "конверси", "отток", "выручк",
    ],
    "Работа с документами": [
        "jira", "confluence", "задача", "epic", "документац", "почта",
        "письм", "входящие", "договор", "презентаци", "сравни версии",
        "таблиц", "извлеки", "переведи", "проверь документ", "шаблон",
    ],
    "Обучение и онбординг": [
        "объясни", "обучен", "онбординг", "инструкци", "тест для проверки",
        "обучающий материал", "как работает", "что такое", "зачем нужен",
        "простым языком", "принцип работы", "стратеги", "план миграции",
        "оцен рис", "техническое задание",
    ],
    "Работа с клиентами": [
        "клиент", "жалоб", "ответ клиент", "faq", "скрипт звонк",
        "коммерческое предложение", "холодн", "переговор", "скидк",
        "follow-up", "презентация продукта", "партнёр",
    ],
}


# ── Классификация ──────────────────────────────────────────────────────

def classify_keyword(text: str) -> list[str]:
    """Классификация по ключевым словам (fallback)."""
    text_lower = text.lower()
    scores = {}
    for category, keywords in CATEGORY_KEYWORDS.items():
        score = sum(1 for kw in keywords if kw in text_lower)
        if score > 0:
            scores[category] = score
    if not scores:
        return ["Прочее"]
    max_score = max(scores.values())
    threshold = max_score * 0.6
    return [cat for cat, s in scores.items() if s >= threshold]


def classify_llm(texts: list[str], llm: LLMClient, batch_size: int = 20) -> list[list[str]]:
    """Классификация через LLM API (батчами)."""
    all_results = []
    categories = [c for c in DEFAULT_CATEGORIES if c != "Прочее"]
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        try:
            batch_results = llm.classify_batch(batch, categories)
            all_results.extend(batch_results)
        except Exception:
            # Fallback на ключевые слова для этого батча
            all_results.extend([classify_keyword(t) for t in batch])
    return all_results


# ── Кластеризация ──────────────────────────────────────────────────────

def extract_use_cases(
    df: pd.DataFrame,
    n_clusters: int = 15,
    top_n: int = 5,
    llm: Optional[LLMClient] = None,
) -> dict:
    """
    Кластеризует запросы и выделяет use-cases.
    Если передан llm — использует его для саммаризации кластеров.
    """
    texts = df["request_text"].tolist()

    # Корректируем число кластеров
    actual_clusters = min(n_clusters, max(2, len(texts) // 3))

    # TF-IDF векторизация
    vectorizer = TfidfVectorizer(
        max_features=5000,
        ngram_range=(1, 2),
        min_df=max(1, len(texts) // 100),
        max_df=0.95,
    )
    tfidf_matrix = vectorizer.fit_transform(texts)

    # KMeans кластеризация
    kmeans = KMeans(n_clusters=actual_clusters, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(tfidf_matrix)

    df = df.copy()
    df["cluster"] = cluster_labels
    feature_names = vectorizer.get_feature_names_out()

    use_cases = {}
    for cluster_id in range(actual_clusters):
        cluster_df = df[df["cluster"] == cluster_id]
        if len(cluster_df) == 0:
            continue

        # Топ-термины
        center = kmeans.cluster_centers_[cluster_id]
        top_indices = center.argsort()[-8:][::-1]
        top_terms = [feature_names[i] for i in top_indices]

        # Категория
        if "category" in cluster_df.columns:
            dominant_cat = cluster_df["category"].value_counts().index[0]
        elif "true_category" in cluster_df.columns:
            dominant_cat = cluster_df["true_category"].value_counts().index[0]
        else:
            dominant_cat = "Неизвестно"

        # Примеры
        examples = []
        for _, row in cluster_df.head(top_n).iterrows():
            text = str(row["request_text"])[:300].replace("\n", " ").strip()
            examples.append(text)

        # Метрики
        avg_tokens = cluster_df["token_count"].mean() if "token_count" in cluster_df.columns else 0
        avg_response = cluster_df["response_time_sec"].mean() if "response_time_sec" in cluster_df.columns else 0
        sat_col = pd.to_numeric(cluster_df.get("satisfaction_score", pd.Series(dtype=float)), errors="coerce")
        avg_sat = sat_col.mean()

        uc = {
            "top_terms": top_terms,
            "dominant_category": dominant_cat,
            "size": len(cluster_df),
            "examples": examples,
            "avg_tokens": round(avg_tokens),
            "avg_response_time": round(avg_response, 1),
            "avg_satisfaction": round(avg_sat, 2) if not np.isnan(avg_sat) else None,
            "users": cluster_df["user_id"].nunique() if "user_id" in cluster_df.columns else 0,
        }

        # LLM-саммаризация кластера
        if llm:
            try:
                llm_summary = llm.summarize_cluster(examples, top_terms)
                uc["llm_summary"] = llm_summary
                uc["name"] = llm_summary.get("name", f"Сценарий #{cluster_id}")
                uc["description"] = llm_summary.get("description", "")
                uc["pain_points"] = llm_summary.get("pain_points", [])
                uc["automation_potential"] = llm_summary.get("automation_potential", "unknown")
                uc["typical_phrases"] = llm_summary.get("typical_phrases", [])
                uc["recommendation"] = llm_summary.get("recommendation", "")
            except Exception:
                uc["name"] = f"Сценарий #{cluster_id}"
                uc["description"] = _generate_fallback_summary(uc)
        else:
            uc["name"] = f"Сценарий #{cluster_id}"
            uc["description"] = _generate_fallback_summary(uc)

        use_cases[cluster_id] = uc

    return use_cases


def _generate_fallback_summary(use_case: dict) -> str:
    """Генерирует текстовое описание без LLM."""
    terms = ", ".join(use_case["top_terms"][:5])
    cat = use_case["dominant_category"]
    return (
        f"Категория: {cat}. "
        f"{use_case['size']} запросов от {use_case['users']} пользователей. "
        f"Ключевые слова: {terms}."
    )


# ── Парсинг файлов ────────────────────────────────────────────────────

def parse_uploaded_file(uploaded_file) -> pd.DataFrame:
    """
    Парсит загруженный файл (CSV или JSON) в DataFrame.
    Автоматически определяет колонку с текстом запроса.
    """
    name = uploaded_file.name.lower()
    content = uploaded_file.read()

    if name.endswith(".json"):
        try:
            data = json.loads(content)
            if isinstance(data, list):
                df = pd.DataFrame(data)
            elif isinstance(data, dict):
                # Попытка найти массив в ключах
                for key in ["logs", "requests", "data", "records", "entries"]:
                    if key in data and isinstance(data[key], list):
                        df = pd.DataFrame(data[key])
                        break
                else:
                    df = pd.DataFrame([data])
            else:
                raise ValueError("Неизвестная структура JSON")
        except json.JSONDecodeError as e:
            raise ValueError(f"Ошибка парсинга JSON: {e}")
    elif name.endswith(".csv"):
        df = pd.read_csv(pd.io.common.BytesIO(content))
    elif name.endswith((".tsv", ".txt")):
        df = pd.read_csv(pd.io.common.BytesIO(content), sep="\t")
    else:
        raise ValueError(f"Неподдерживаемый формат: {name}")

    # Нормализация колонок
    df = _normalize_columns(df)
    return df


def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Нормализует названия колонок и находит колонку с текстом."""
    # Маппинг возможных названий колонок
    text_candidates = [
        "request_text", "text", "query", "prompt", "message",
        "request", "input", "user_message", "user_input",
        "запрос", "текст", "сообщение", "промпт",
    ]
    time_candidates = [
        "timestamp", "created_at", "date", "time", "datetime",
        "дата", "время",
    ]
    user_candidates = [
        "user_id", "userId", "user", "author", "client_id",
        "пользователь",
    ]
    token_candidates = [
        "token_count", "tokens", "num_tokens", "token_count",
        "токены",
    ]
    response_candidates = [
        "response_time_sec", "response_time", "latency",
        "время_ответа",
    ]
    satisfaction_candidates = [
        "satisfaction_score", "satisfaction", "rating", "score",
        "оценка",
    ]
    category_candidates = [
        "true_category", "category", "label", "класс",
        "категория",
    ]

    def find_col(candidates):
        for c in candidates:
            if c in df.columns:
                return c
            for col in df.columns:
                if col.lower().strip() == c.lower():
                    return col
        return None

    # Проверяем обязательную колонку с текстом
    text_col = find_col(text_candidates)
    if text_col is None:
        # Если нет явной колонки с текстом, берём самую длинную строковую
        str_cols = df.select_dtypes(include=["object"]).columns
        if len(str_cols) > 0:
            avg_lengths = {col: df[col].astype(str).str.len().mean() for col in str_cols}
            text_col = max(avg_lengths, key=avg_lengths.get)
        else:
            raise ValueError(
                "Не удалось найти колонку с текстом запроса. "
                f"Доступные колонки: {list(df.columns)}"
            )

    # Переименовываем в стандартные имена
    rename_map = {}
    if text_col != "request_text":
        rename_map[text_col] = "request_text"

    for candidates, target in [
        (time_candidates, "timestamp"),
        (user_candidates, "user_id"),
        (token_candidates, "token_count"),
        (response_candidates, "response_time_sec"),
        (satisfaction_candidates, "satisfaction_score"),
        (category_candidates, "true_category"),
    ]:
        col = find_col(candidates)
        if col and col != target:
            rename_map[col] = target

    df = df.rename(columns=rename_map)

    # Удаляем строки без текста
    df = df.dropna(subset=["request_text"])
    df["request_text"] = df["request_text"].astype(str).str.strip()
    df = df[df["request_text"].str.len() > 10]

    # Генерируем недостающие колонки
    if "id" not in df.columns:
        df["id"] = range(1, len(df) + 1)
    if "timestamp" not in df.columns:
        df["timestamp"] = pd.Timestamp.now()
    if "user_id" not in df.columns:
        df["user_id"] = "unknown"
    if "token_count" not in df.columns:
        # Оценка: ~4 символа на токен для русского
        df["token_count"] = (df["request_text"].str.len() / 4).astype(int)
    if "response_time_sec" not in df.columns:
        df["response_time_sec"] = 0.0

    return df


# ── Полный пайплайн ────────────────────────────────────────────────────

def build_analytics(
    df: pd.DataFrame,
    llm: Optional[LLMClient] = None,
    n_clusters: int = 15,
) -> dict:
    """
    Полный пайплайн аналитики.
    Принимает DataFrame (не путь к файлу) — поддерживает загруженные файлы.
    """
    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])

    has_llm = llm is not None

    # ── 1. Классификация ───────────────────────────────────────────────
    if "true_category" in df.columns and df["true_category"].notna().any():
        df["category"] = df["true_category"]
        # Если есть LLM и нет true_category — до-классифицируем пропуски
        missing = df["category"].isna() | (df["category"] == "")
        if has_llm and missing.any():
            texts = df.loc[missing, "request_text"].tolist()
            cats = classify_llm(texts, llm)
            df.loc[missing, "category"] = [c[0] if c else "Прочее" for c in cats]
    elif has_llm:
        texts = df["request_text"].tolist()
        cats = classify_llm(texts, llm)
        df["category"] = [c[0] if c else "Прочее" for c in cats]
    else:
        df["category"] = df["request_text"].apply(lambda t: classify_keyword(t)[0])

    cat_counts = df["category"].value_counts()

    # ── 2. Кластеризация / Use-cases ───────────────────────────────────
    use_cases = extract_use_cases(df, n_clusters=n_clusters, llm=llm)

    # ── 3. Временные тренды ────────────────────────────────────────────
    df["date"] = df["timestamp"].dt.date
    timeline = df.groupby("date").agg(
        requests=("id", "count"),
        avg_tokens=("token_count", "mean"),
        avg_satisfaction=(
            "satisfaction_score",
            lambda x: pd.to_numeric(x, errors="coerce").mean()
            if "satisfaction_score" in df.columns else np.nan
        ),
    ).reset_index()
    timeline["date"] = timeline["date"].astype(str)
    timeline["avg_tokens"] = timeline["avg_tokens"].round(0)
    timeline["avg_satisfaction"] = timeline["avg_satisfaction"].round(2)

    # ── 4. Топ пользователей ──────────────────────────────────────────
    top_users = df["user_id"].value_counts().head(10).to_dict() if "user_id" in df.columns else {}

    # ── 5. Общая статистика ────────────────────────────────────────────
    sat_col = pd.to_numeric(df.get("satisfaction_score", pd.Series(dtype=float)), errors="coerce")
    summary = {
        "total_requests": len(df),
        "unique_users": df["user_id"].nunique() if "user_id" in df.columns else "N/A",
        "date_range": f"{df['timestamp'].min().date()} — {df['timestamp'].max().date()}",
        "avg_tokens": round(df["token_count"].mean()),
        "avg_response_time": round(df["response_time_sec"].mean(), 1),
        "avg_satisfaction": round(sat_col.mean(), 2) if sat_col.notna().any() else None,
        "num_categories": len(cat_counts),
        "num_use_cases": len(use_cases),
        "llm_enabled": has_llm,
    }

    # ── 6. Сценарии по категориям ──────────────────────────────────────
    scenarios_by_cat = defaultdict(list)
    for uc_id, uc in use_cases.items():
        scenarios_by_cat[uc["dominant_category"]].append({
            "id": uc_id,
            "size": uc["size"],
            "top_terms": uc["top_terms"],
            "name": uc.get("name", ""),
        })

    # ── 7. LLM-инсайты ────────────────────────────────────────────────
    llm_insights = None
    if has_llm:
        try:
            llm_insights = llm.generate_insights(
                cat_counts.to_dict(),
                list(use_cases.values()),
            )
        except Exception:
            pass

    return {
        "summary": summary,
        "category_stats": cat_counts.to_dict(),
        "use_cases": use_cases,
        "timeline": timeline.to_dict(orient="records"),
        "top_users": top_users,
        "scenarios_by_cat": dict(scenarios_by_cat),
        "raw_df": df,
        "llm_insights": llm_insights,
    }


# ── Обратная совместимость (старый API с путём к файлу) ────────────────

def build_analytics_from_csv(csv_path: str, llm: Optional[LLMClient] = None) -> dict:
    """Обёртка для обратной совместимости."""
    df = pd.read_csv(csv_path)
    return build_analytics(df, llm=llm)
