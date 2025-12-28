import os
import re
import time
import sqlite3
from typing import List, Tuple, Optional, Dict
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss # используем для быстрого поиска по косинусному сходству (через inner product)

DB_MODEL_NAME = os.getenv(
    "ST_MODEL_NAME",
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
)

# Выбираем, какой вектор из qa_vec используем для индекса:
WHICH_VEC = os.getenv("WHICH_VEC", "q")

# Сколько кандидатов достаём из FAISS (используем для диагностики и отбора топа)
TOP_N_DEFAULT = int(os.getenv("TOP_N", "50"))

# Сколько ответов возвращаем в бот (топ-K)
FINAL_K_DEFAULT = int(os.getenv("FINAL_K", "5"))

# Если лучший similarity ниже порога, отклоняем запрос
SEM_THR_DEFAULT = float(os.getenv("SEM_THR", "0.55"))

# Кешируем эмбеддинги запросов в БД, чтобы не считать их повторно
CACHE_QUERY_EMB_TO_DB = True

# Оставляем как константу для совместимости (в этом алгоритме не используем)
MAX_FTS_TOKENS = 20

_st_model: Optional[SentenceTransformer] = None
_faiss_index = None
_faiss_ids: Optional[np.ndarray] = None  # (N,) int64
_sem_thr: float = SEM_THR_DEFAULT


def norm(s: str) -> str:
    """Нормализуем текст: схлопываем пробелы и обрезаем края."""
    return re.sub(r"\s+", " ", (s or "").strip())


def blob_to_vec(b: bytes, dim: int) -> np.ndarray:
    """Достаём float32-вектор из BLOB."""
    return np.frombuffer(b, dtype=np.float32, count=dim)


def vec_to_blob(v: np.ndarray) -> bytes:
    """Пишем float32-вектор в BLOB."""
    v = np.asarray(v, dtype=np.float32)
    return v.tobytes()


def l2_normalize_rows(X: np.ndarray) -> np.ndarray:
    """Нормализуем строки матрицы, чтобы inner product = cosine similarity."""
    X = np.asarray(X, dtype=np.float32)
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return X / norms


def query_hash(text: str) -> str:
    """Строим стабильный хеш запроса для кеширования эмбеддинга."""
    import hashlib
    return hashlib.sha1(norm(text).encode("utf-8")).hexdigest()


def ensure_query_cache_table(con: sqlite3.Connection) -> None:
    """Создаём таблицу кеша эмбеддингов запросов, если её ещё нет."""
    cur = con.cursor()
    cur.execute("""
    CREATE TABLE IF NOT EXISTS qa_query_cache (
        query_hash TEXT PRIMARY KEY,
        query_text TEXT,
        created_ts INTEGER NOT NULL,
        dim INTEGER NOT NULL,
        q_emb BLOB NOT NULL
    );
    """)
    con.commit()


def get_cached_query_emb(con: sqlite3.Connection, qh: str) -> Optional[np.ndarray]:
    """Пробуем достать эмбеддинг запроса из кеша."""
    cur = con.cursor()
    row = cur.execute(
        "SELECT q_emb, dim FROM qa_query_cache WHERE query_hash=?;",
        (qh,),
    ).fetchone()
    if not row:
        return None
    blob, dim = row
    return blob_to_vec(blob, int(dim))


def put_cached_query_emb(con: sqlite3.Connection, qh: str, query_text: str, emb: np.ndarray) -> None:
    """Сохраняем эмбеддинг запроса в кеш."""
    cur = con.cursor()
    cur.execute(
        """
        INSERT OR REPLACE INTO qa_query_cache(query_hash, query_text, created_ts, dim, q_emb)
        VALUES(?, ?, ?, ?, ?);
        """,
        (qh, query_text, int(time.time()), int(emb.shape[0]), vec_to_blob(emb)),
    )
    con.commit()


def load_all_questions(con: sqlite3.Connection) -> List[Tuple[int, str]]:
    """
    Возвращаем список всех вопросов (qa_id, question).
    Оставляем для совместимости с bot.py (там мы кешируем вопросы при старте).
    В этом алгоритме этот кеш напрямую не используем.
    """
    cur = con.cursor()
    rows = cur.execute("SELECT id, question FROM qa ORDER BY id;").fetchall()
    return [(int(i), q) for i, q in rows]


def load_all_embeddings(
    con: sqlite3.Connection,
    model_name: str,
    which_vec: str = "q",
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Загружаем все эмбеддинги из qa_vec и строим матрицу X.

    Возвращаем:
      ids: (N,) int64
      X:   (N, D) float32, L2-нормированная
    """
    vec_col = "q_vec" if which_vec == "q" else "a_vec"

    cur = con.cursor()
    rows = cur.execute(
        f"""
        SELECT qa_id, dim, {vec_col}
        FROM qa_vec
        WHERE model_name = ?
        ORDER BY qa_id;
        """,
        (model_name,),
    ).fetchall()

    # Если по model_name ничего не нашли, делаем fallback и грузим все вектора без фильтра
    if not rows:
        rows = cur.execute(
            f"SELECT qa_id, dim, {vec_col} FROM qa_vec ORDER BY qa_id;"
        ).fetchall()

    ids: List[int] = []
    vecs: List[np.ndarray] = []

    for qa_id, dim, blob in rows:
        if blob is None:
            continue
        v = np.frombuffer(blob, dtype=np.float32, count=int(dim))
        ids.append(int(qa_id))
        vecs.append(v)

    if not vecs:
        raise RuntimeError(
            "Не нашли эмбеддинги в qa_vec. Проверим, что таблица заполнена и вектора (q_vec/a_vec) не NULL."
        )

    X = np.vstack(vecs).astype(np.float32)
    X = l2_normalize_rows(X)
    ids_arr = np.asarray(ids, dtype=np.int64)

    return ids_arr, X


def build_faiss_index(X: np.ndarray):
    """
    Строим FAISS индекс по inner product.
    Так как X уже L2-нормирован, inner product соответствует cosine similarity.
    """
    d = int(X.shape[1])
    index = faiss.IndexFlatIP(d)
    index.add(np.ascontiguousarray(X, dtype=np.float32))
    return index


def dense_topn(
    query: str,
    model: SentenceTransformer,
    index,
    ids: np.ndarray,
    top_n: int,
) -> Tuple[List[int], List[float]]:
    """
    Считаем эмбеддинг запроса и забираем top-N из FAISS.

    Возвращаем:
      top_ids: список qa_id
      top_sims: соответствующие similarity
    """
    q_vec = model.encode([query], normalize_embeddings=True, show_progress_bar=False).astype(np.float32)
    sims, idxs = index.search(q_vec, int(top_n))

    sims = sims[0]
    idxs = idxs[0]

    top_ids = [int(ids[i]) for i in idxs if i != -1]
    top_sims = [float(s) for s, i in zip(sims, idxs) if i != -1]
    return top_ids, top_sims

def init_models_once(
    con: Optional[sqlite3.Connection] = None,
    st_model_name: str = DB_MODEL_NAME,
    which_vec: str = WHICH_VEC,
    sem_thr: float = SEM_THR_DEFAULT,
) -> None:
    """
    Вызываем один раз при старте бота:
    - загружаем SentenceTransformer
    - строим FAISS индекс по эмбеддингам из qa_vec
    """
    global _st_model, _faiss_index, _faiss_ids, _sem_thr

    _sem_thr = float(sem_thr)

    if _st_model is None:
        _st_model = SentenceTransformer(st_model_name)

    if _faiss_index is None or _faiss_ids is None:
        if con is None:
            raise RuntimeError(
                "init_models_once(con=...) требует открытое соединение sqlite, чтобы загрузить эмбеддинги qa_vec."
            )

        ids, X = load_all_embeddings(con, model_name=st_model_name, which_vec=which_vec)
        _faiss_ids = ids
        _faiss_index = build_faiss_index(X)


def hybrid_search(
    con: sqlite3.Connection,
    query: str,
    final_k: int = FINAL_K_DEFAULT,
) -> Tuple[Optional[tuple], List[tuple], Dict]:
    """
    Возвращаем:
      best_row, top_rows, debug_info

    Делаем так:
      1) получаем top-N по FAISS (cosine similarity)
      2) если лучший similarity < sem_thr — отклоняем (возвращаем пусто)
      3) иначе возвращаем top-K строк из qa
    """
    query = norm(query)
    if not query:
        return None, [], {}

    dbg: Dict[str, object] = {}

    # При желании кешируем эмбеддинги запросов в БД
    if CACHE_QUERY_EMB_TO_DB:
        ensure_query_cache_table(con)

    # Убеждаемся, что модель и индекс загружены
    if _st_model is None or _faiss_index is None or _faiss_ids is None:
        init_models_once(con=con)

    # Берем эмбеддинг запроса из кеша, иначе считаем и кладём в кеш
    qh = query_hash(query)
    if CACHE_QUERY_EMB_TO_DB:
        q_vec = get_cached_query_emb(con, qh)
    else:
        q_vec = None

    if q_vec is None:
        q_vec = _st_model.encode([query], normalize_embeddings=True, show_progress_bar=False)[0].astype(np.float32)
        if CACHE_QUERY_EMB_TO_DB:
            put_cached_query_emb(con, qh, query, q_vec)

    # Делаем поиск по FAISS
    q_vec_2d = q_vec.reshape(1, -1).astype(np.float32)
    sims, idxs = _faiss_index.search(q_vec_2d, int(max(final_k, TOP_N_DEFAULT)))

    sims = sims[0]
    idxs = idxs[0]

    top_ids = [int(_faiss_ids[i]) for i in idxs if i != -1]
    top_sims = [float(s) for s, i in zip(sims, idxs) if i != -1]

    dbg["sem_thr"] = float(_sem_thr)
    dbg["topn_ids"] = top_ids[:TOP_N_DEFAULT]
    dbg["topn_sims"] = top_sims[:TOP_N_DEFAULT]

    if not top_ids:
        return None, [], dbg

    best_sim = float(top_sims[0]) if top_sims else -1.0
    dbg["best_sim"] = best_sim

    # Отклоняем запрос, если похожести недостаточно
    if best_sim < float(_sem_thr):
        dbg["rejected"] = True
        return None, [], dbg

    dbg["rejected"] = False

    # Берём top-K для выдачи в бот
    out_ids = top_ids[: int(final_k)]
    placeholders = ",".join("?" for _ in out_ids)

    cur = con.cursor()
    cur.execute(
        f"""
        SELECT id, page, question, answer_text, source_url
        FROM qa
        WHERE id IN ({placeholders});
        """,
        out_ids,
    )
    rows = cur.fetchall()
    row_map = {int(r[0]): r for r in rows}
    ordered = [row_map[i] for i in out_ids if i in row_map]

    best = ordered[0] if ordered else None
    dbg["top_ids"] = out_ids

    return best, ordered, dbg
