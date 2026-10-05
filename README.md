Выполнил `Дьяконов Олег`
Группа `БВТ2506`
---

#### Здесь я разберу каждую ячейку кода по отдельности и объясню, что она делает. Так станет понятно, как все части складываются в общую картину

---
### Установка зависимостей
```
pip install sentence-transformers numpy pandas matplotlib scikit-learn
```
> Устанавливаем библиотеки для работы с эмбеддингами, вычислений и визуализации
---

### Импорт библиотек
```
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sentence_transformers import SentenceTransformer
from sklearn.manifold import TSNE
from sklearn.metrics.pairwise import cosine_similarity
import warnings
warnings.filterwarnings('ignore')
```
> `entence-transformers` - генерация эмбеддингов
> `sklearn.manifold.TSNE` - визуализация в 2D
> `sklearn.metrics.pairwise.cosine_similarity` - сравнение векторов
---

### Загрузка данных

```
with open('code_corpus.json', 'r', encoding='utf-8') as f:
    code_corpus = json.load(f)

with open('eval_questions.json', 'r', encoding='utf-8') as f:
    eval_questions = json.load(f)
```
> `code_corpus.json` - 200 фрагментов кода (Python + Java)
> `eval_questions.json` — 25 вопросов с правильными ответами
---

### Подготовка тестов для эмбеддингов

```
corpus_texts = [f"{item['code']} {item['description']}" for item in code_corpus]
queries = [q['query'] for q in eval_questions]
correct_ids = [q['correct_chunk_id'] for q in eval_questions]
```
> Склеиваем код и описание в один текст
> Так модель видит синтаксис и смысловое описание
---

### Загрузка моделей
```
models = {}
for name in ["paraphrase-multilingual-MiniLM-L12-v2", 
             "paraphrase-multilingual-mpnet-base-v2"]:
    models[name] = SentenceTransformer(name)
```
> `MiniLM-L12` — 384 размерность, 80MB, быстрая
> `mpnet-base` — 768 размерность, 420MB, точнее
---

### Генерация эмбеддингов
```
embeddings = {}
for name, model in models.items():
    embeddings[name] = model.encode(
        corpus_texts,
        show_progress_bar=True,
        batch_size=32
    )
```
> Превращаем 200 фрагментов кода в матрицы чисел
> MiniLM → (200, 384), mpnet → (200, 768)
---

### Функция поиска 
```
def search(query, model, corpus_emb, top_k=3):
    q_emb = model.encode([query])
    sim = cosine_similarity(q_emb, corpus_emb)[0]
    idx = np.argsort(sim)[::-1][:top_k]
    return idx, sim[idx]
```
> Косинусное сходство между запросом и всеми фрагментами
> Сортируем по убыванию → берём топ-3
---

### Тестирование и расчет Precision@3
```
results = {}
for name, model in models.items():
    results[name] = []
    for i, q in enumerate(queries):
        idx, _ = search(q, model, embeddings[name])
        found = [corpus_ids[j] for j in idx]
        results[name].append({
            'query': q,
            'correct': correct_ids[i],
            'found': found,
            'ok': correct_ids[i] in found
        })
    ok = sum(1 for r in results[name] if r['ok'])
    print(f"{name}: Precision@3 = {ok/len(queries):.2%}")
```
> Для каждого вопроса проверяем: попал ли правильный ID в топ-3?
> Precision@3 = доля вопросов, где ответ найден
---

### t-SNE визуализация
```
best = embeddings["paraphrase-multilingual-mpnet-base-v2"]
tsne = TSNE(n_components=2, random_state=42, perplexity=30)
coords = tsne.fit_transform(best)

colors = {'auth': '#E74C3C', 'database': '#3498DB', 
          'http': '#2ECC71', 'validation': '#F39C12', 
          'utils': '#9B59B6'}

for cat, color in colors.items():
    mask = [c == cat for c in corpus_categories]
    plt.scatter(coords[mask, 0], coords[mask, 1], 
                c=color, label=cat, alpha=0.7)
plt.legend()
plt.show()
```
> Проекция 200 векторов в 2D
> Видно, что фрагменты одной категории лежат рядом — модель уловила смысловые кластеры
---

# Запуск проекта
```
# 1. Установить зависимости
pip install -r requirements.txt

# 2. Запустить Jupyter
jupyter notebook

# 3. Открыть main.ipynb и выполнить все ячейки
```
