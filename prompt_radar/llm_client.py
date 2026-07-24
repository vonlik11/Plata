"""
LLM-клиент для «Промпт-радара».
Поддерживает OpenAI-совместимые API (OpenAI, vLLM, Ollama, LM Studio и т.д.).
"""

import json
import re
import requests
from typing import Optional


def _parse_json_from_response(text: str) -> dict | list:
    """Извлекает JSON из ответа LLM (учитывает markdown-блоки)."""
    # Попытка найти JSON в блоке ```json ... ```
    match = re.search(r"```(?:json)?\s*\n?(.*?)\n?\s*```", text, re.DOTALL)
    if match:
        text = match.group(1).strip()
    # Попытка найти JSON-объект или массив
    for start_char, end_char in [("{", "}"), ("[", "]")]:
        start = text.find(start_char)
        end = text.rfind(end_char)
        if start != -1 and end != -1 and end > start:
            try:
                return json.loads(text[start:end + 1])
            except json.JSONDecodeError:
                continue
    return json.loads(text)


class LLMClient:
    """Клиент для OpenAI-совместимых API."""

    def __init__(self, base_url: str, api_key: str, model: str, timeout: int = 120):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model = model
        self.timeout = timeout

    def _call(self, messages: list[dict], temperature: float = 0.1, max_tokens: int = 2000) -> str:
        """Отправляет запрос к LLM API."""
        url = f"{self.base_url}/chat/completions"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        resp = requests.post(url, json=payload, headers=headers, timeout=self.timeout)
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"]

    def classify_batch(self, texts: list[str], categories: list[str]) -> list[list[str]]:
        """
        Классифицирует батч запросов.
        Возвращает список списков категорий для каждого запроса.
        """
        cat_list = "\n".join(f"- {c}" for c in categories)
        text_block = ""
        for i, t in enumerate(texts):
            snippet = t[:500].replace("\n", " ")
            text_block += f'\n[{i}] "{snippet}"'

        prompt = f"""Ты — аналитик логов ИИ-агентов. Классифицируй каждый запрос пользователя.

Доступные категории:
{cat_list}

Запросы для классификации:
{text_block}

Ответь ТОЛЬКО валидным JSON-массивом массивов, где каждый внутренний массив содержит категории для соответствующего запроса.
Пример: [["Генерация текста", "Работа с документами"], ["Написание кода"], ...]
Если запрос не подходит ни под одну категорию, используй ["Прочее"]."""

        result = self._call([{"role": "user", "content": prompt}], temperature=0.0)
        parsed = _parse_json_from_response(result)

        # Валидация
        if isinstance(parsed, list) and len(parsed) == len(texts):
            return parsed
        # Fallback: повторить для каждого отдельно
        return [[c] for c in categories[:len(texts)]]

    def summarize_cluster(self, examples: list[str], cluster_terms: list[str]) -> dict:
        """
        Генерирует саммари для кластера запросов.
        Возвращает dict с полями: name, description, pain_points, automation_potential, typical_phrases.
        """
        examples_text = "\n".join(f"- {e[:300]}" for e in examples[:5])
        terms_text = ", ".join(cluster_terms[:8])

        prompt = f"""Ты — продуктовый аналитик. Проанализируй группу похожих запросов к ИИ-агенту и создай структурированное описание.

Ключевые термины кластера: {terms_text}

Примеры запросов:
{examples_text}

Ответь строго в JSON-формате:
{{
  "name": "Краткое название сценария (2-5 слов)",
  "description": "Описание что делают пользователи и зачем (2-3 предложения)",
  "pain_points": ["Боль 1", "Боль 2"],
  "automation_potential": "high/medium/low — потенциал автоматизации",
  "typical_phrases": ["типовая формулировка 1", "типовая формулировка 2"],
  "recommendation": "Рекомендация CTO что делать (1-2 предложения)"
}}"""

        result = self._call([{"role": "user", "content": prompt}], temperature=0.2)
        return _parse_json_from_response(result)

    def generate_insights(self, category_stats: dict, use_cases: list[dict]) -> str:
        """
        Генерирует стратегические инсайты для CTO на основе всей аналитики.
        """
        cat_text = json.dumps(category_stats, ensure_ascii=False, indent=2)
        uc_text = json.dumps([
            {"name": uc.get("name", ""), "size": uc.get("size", 0),
             "automation": uc.get("automation_potential", "unknown")}
            for uc in use_cases[:10]
        ], ensure_ascii=False, indent=2)

        prompt = f"""Ты — AI-аналитик. На основе данных об использовании ИИ-агентов сформируй 3-5 ключевых инсайтов для CTO.

Распределение по категориям:
{cat_text}

Топ сценариев:
{uc_text}

Для каждого инсайта укажи:
1. Констатация факта (что происходит)
2. Рекомендация (что делать)
3. Приоритет (high/medium/low)

Ответь в формате markdown."""

        return self._call([{"role": "user", "content": prompt}], temperature=0.3)

    def test_connection(self) -> tuple[bool, str]:
        """Проверяет доступность API."""
        try:
            result = self._call([
                {"role": "user", "content": "Ответь одним словом: OK"}
            ], max_tokens=10)
            return True, f"Подключение успешно. Модель: {self.model}"
        except Exception as e:
            return False, str(e)
