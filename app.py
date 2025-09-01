# app.py
# -*- coding: utf-8 -*-
"""
Flask-приложение: сайт "Космос: новости, 3D и Луна"
Функции:
 - Главная: NASA Eyes (iframe)
 - Новости: список, просмотр, добавление (author, image, audio)
 - Удаление новостей
 - Автоудаление фейков (ИИ): порог FAKE_THRESHOLD
 - Фаза Луны
 - Фавикон, фон. музыка (копируется из /mnt/data при наличии)
Запуск:
 pip install flask markdown2
 python app.py
 Открой: http://127.0.0.1:5000/
"""
from __future__ import annotations

import math
import os
import re
import shutil
import sqlite3
from datetime import datetime, timezone
from typing import List, Optional
import requests
from bs4 import BeautifulSoup
from sentence_transformers import SentenceTransformer, util

# Загружаем модель
model = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

# Функция парсинга новостей NASA (можно расширить для других сайтов)
def fetch_nasa_news():
    url = "https://www.nasa.gov/news/"
    r = requests.get(url, timeout=10)
    soup = BeautifulSoup(r.text, "html.parser")
    articles = [a.get_text(strip=True) for a in soup.find_all("h3")]
    return articles[:10]  # последние 10 новостей

def check_news(user_news: str):
    trusted_news = fetch_nasa_news()
    if not trusted_news:
        return "⚠️ Не удалось загрузить новости NASA."

    # Векторизация
    embeddings_trusted = model.encode(trusted_news, convert_to_tensor=True)
    embedding_user = model.encode(user_news, convert_to_tensor=True)

    # Сравнение
    scores = util.cos_sim(embedding_user, embeddings_trusted)[0]
    max_score = float(scores.max())

    if max_score > 0.8:
        return f"✅ Новость подтверждена (схожесть {max_score:.2f})"
    elif max_score > 0.5:
        return f"❓ Новость требует проверки (схожесть {max_score:.2f})"
    else:
        return f"❌ Новость вероятно фейк (схожесть {max_score:.2f})"

# Пример
print(check_news("Телескоп «Джеймс Уэбб» обнаружил новую планету."))

from flask import (
    Flask, request, redirect, url_for, g, abort, jsonify,
    render_template_string, flash
)
from werkzeug.utils import secure_filename

# Optional markdown support
try:
    import markdown2
    md = markdown2.Markdown(extras=["fenced-code-blocks", "tables", "strike", "target-blank-links"])
except Exception:
    md = None

# ---------------- CONFIG ----------------
APP_TITLE = "Космос: новости, 3D и Луна"
BASE_DIR = os.path.dirname(__file__)
DB_PATH = os.path.join(BASE_DIR, "news.db")
UPLOAD_FOLDER = os.path.join(BASE_DIR, "static", "uploads")
MUSIC_FOLDER = os.path.join(BASE_DIR, "static", "music")
ALLOWED_IMG = {"png", "jpg", "jpeg", "gif", "webp"}
ALLOWED_AUDIO = {"mp3", "ogg", "wav", "m4a"}
MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16 MB
FAKE_THRESHOLD = 70.0  # если score > 70 — считается фейком и отклоняется/удаляется

# ensure folders
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(MUSIC_FOLDER, exist_ok=True)

# try to copy provided mp3 (system uploaded) to static music
SRC_MP3 = "/mnt/data/209be3da467651a.mp3"
DEST_MP3 = os.path.join(MUSIC_FOLDER, "space.mp3")
if os.path.exists(SRC_MP3) and not os.path.exists(DEST_MP3):
    try:
        shutil.copy2(SRC_MP3, DEST_MP3)
        print("Copied user mp3 to static/music/space.mp3")
    except Exception as e:
        print("Failed to copy mp3:", e)

app = Flask(__name__)
app.config['JSON_AS_ASCII'] = False
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MUSIC_FOLDER'] = MUSIC_FOLDER
app.config['MAX_CONTENT_LENGTH'] = MAX_CONTENT_LENGTH
app.secret_key = 'change-this-in-production'

# ----------------- DB -------------------
def get_db():
    if 'db' not in g:
        g.db = sqlite3.connect(DB_PATH)
        g.db.row_factory = sqlite3.Row
    return g.db

@app.teardown_appcontext
def close_db(exception):
    db = g.pop('db', None)
    if db is not None:
        db.close()

def init_db():
    db = get_db()
    # create table if not exists with required columns
    db.execute("""
        CREATE TABLE IF NOT EXISTS news (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            title TEXT NOT NULL,
            author TEXT,
            content TEXT NOT NULL,
            image TEXT,
            audio TEXT,
            created_at TEXT NOT NULL,
            fake_score REAL NOT NULL
        )
    """)
    db.commit()

# if older DB existed with fewer columns, try to ensure columns exist (safe ALTER)
def ensure_columns():
    db = get_db()
    cur = db.execute("PRAGMA table_info(news)")
    cols = {r['name'] for r in cur.fetchall()}
    # add columns if missing (SQLite supports ADD COLUMN)
    if 'author' not in cols:
        db.execute("ALTER TABLE news ADD COLUMN author TEXT")
    if 'audio' not in cols:
        db.execute("ALTER TABLE news ADD COLUMN audio TEXT")
    db.commit()

with app.app_context():
    init_db()
    ensure_columns()

    # ---------------- Новости для главной страницы ----------------
def get_latest_news(limit: int = 5) -> List[sqlite3.Row]:
    """
    Возвращает последние новости для отображения на главной странице.
    По умолчанию — 5 последних новостей.
    """
    db = get_db()
    cur = db.execute("SELECT id, title, author, created_at, image, fake_score FROM news ORDER BY id DESC LIMIT ?", (limit,))
    return cur.fetchall()

# ----------------- Улучшенная эвристика фейков -----------------
def improved_fake_score(text: str, title: str = "") -> float:
    """
    Более продвинутая эвристика ИИ для оценки вероятности фейка.
    Возвращает 0..100, где 100 — очень вероятно фейк.
    """
    if not text:
        return 0.0

    combined = (title + "\n" + text).strip()
    score = 0.0

    # 1. Проверка сенсационных слов и слов с заглавными буквами
    for w in SENSATIONAL:
        if w.lower() in combined.lower():
            score += 10  # увеличенный вес

    # 2. Специфические космические маркеры
    for pat in SPACE_FAKE_TELLS:
        if re.search(pat, combined, flags=re.IGNORECASE):
            score += 18  # увеличенный вес

    # 3. Проверка ссылок
    if not URL_REGEX.search(combined):
        score += 12

    # 4. Пунктуация
    excls = combined.count("!")
    ques = combined.count("?")
    score += min(excls * 2.0, 20)  # чуть сильнее
    score += min(ques * 1.5, 15)

    # 5. Пропорция слов в верхнем регистре
    words = re.findall(r"[A-Za-zА-Яа-яЁё0-9]+", combined)
    if words:
        caps_words = sum(1 for w in words if UPPERCASE_WORD.match(w))
        ratio = caps_words / max(1, len(words))
        score += min(ratio * 120, 25)  # чуть сильнее

    # 6. Проверка длины текста
    avg_len = sum(len(w) for w in words) / max(1, len(words))
    if len(combined) < 280:
        score += 6
    if avg_len < 4.0:
        score += 5

    # 7. Контекстные проверки (новые правила)
    if re.search(r"\b(сенсация|шок|эксклюзив)\b", combined, flags=re.IGNORECASE):
        score += 7

    # Ограничение до 100
    return float(max(0.0, min(100.0, score)))


# ---------------- Fake heuristic & moon phase ----------------
SENSATIONAL = ["ШОК", "СРОЧНО", "НЕВЕРОЯТНО", "СЕНСАЦИЯ", "РАЗОБЛАЧЕНО", "СКАНДАЛ", "EXCLUSIVE", "BREAKING"]
SPACE_FAKE_TELLS = [r"плоская\s+земля", r"рептилоид", r"\bНЛО\b", r"заговор"]
URL_REGEX = re.compile(r"https?://")
UPPERCASE_WORD = re.compile(r"^[A-ZА-ЯЁ]{4,}$")

def fake_score_heuristic(text: str, title: str = "") -> float:
    """
    Возвращает 0..100 — чем выше, тем вероятнее фейк.
    Простая эвристика — сочетание признаков.
    """
    if not text:
        return 0.0
    t = (title + "\n" + text).strip()
    score = 0.0
    # sensational words
    for w in SENSATIONAL:
        if w.lower() in t.lower():
            score += 8
    # space-specific markers
    for pat in SPACE_FAKE_TELLS:
        if re.search(pat, t, flags=re.IGNORECASE):
            score += 15
    # lack of links -> suspicious
    if not URL_REGEX.search(t):
        score += 12
    # punctuation emphasis
    excls = t.count("!")
    ques = t.count("?")
    score += min(excls * 1.5, 15)
    score += min(ques * 1.2, 12)
    # uppercase words ratio
    words = re.findall(r"[A-Za-zА-Яа-яЁё0-9]+", t)
    if words:
        caps_words = sum(1 for w in words if UPPERCASE_WORD.match(w))
        ratio = caps_words / max(1, len(words))
        score += min(ratio * 100, 20)
    # length heuristics
    avg_len = sum(len(w) for w in words) / max(1, len(words))
    if len(t) < 280:
        score += 6
    if avg_len < 4.0:
        score += 4
    return float(max(0.0, min(100.0, score)))

# moon phase (approx)
EPOCH = datetime(2000, 1, 6, 18, 14, tzinfo=timezone.utc)
SYNODIC = 29.530588853
PHASE_NAMES = ["Новолуние","Растущий серп","Первая четверть","Растущая луна","Полнолуние","Убывающая луна","Последняя четверть","Стареющий серп"]

def moon_phase_info(now: Optional[datetime] = None) -> dict:
    now = now or datetime.now(timezone.utc)
    days = (now - EPOCH).total_seconds() / 86400.0
    cycle = (days % SYNODIC) / SYNODIC
    illum = 0.5 * (1 - math.cos(2 * math.pi * cycle))
    idx = int((cycle * 8) + 0.5) % 8
    return {"cycle": cycle, "illum": illum, "illum_pct": round(illum*100,1), "name": PHASE_NAMES[idx], "utc": now.strftime("%Y-%m-%d %H:%M UTC")}

# ---------------- Cleanup (ИИ удаляет фейки) --------------
def cleanup_fakes():
    """
    Удаляет из БД все новости с fake_score > FAKE_THRESHOLD,
    а также удаляет связанные файлы (image, audio).
    Вызывается при старте и при просмотре списка новостей.
    """
    db = get_db()
    cur = db.execute("SELECT id, image, audio FROM news WHERE fake_score > ?", (FAKE_THRESHOLD,))
    rows = cur.fetchall()
    if not rows:
        return
    for r in rows:
        # delete files
        if r['image']:
            fp = os.path.join(app.config['UPLOAD_FOLDER'], r['image'])
            try:
                if os.path.exists(fp):
                    os.remove(fp)
            except Exception:
                pass
        if r['audio']:
            fp = os.path.join(app.config['UPLOAD_FOLDER'], r['audio'])
            try:
                if os.path.exists(fp):
                    os.remove(fp)
            except Exception:
                pass
    db.execute("DELETE FROM news WHERE fake_score > ?", (FAKE_THRESHOLD,))
    db.commit()

with app.app_context():
    # one-time cleanup at start
    cleanup_fakes()
# ---------------- Новый раздел "О нас" -----------------
ABOUT_PAGE = """
<!doctype html>
<html lang="ru">
<head>
{{ head|safe }}
</head>
<body>
<header>
  <div class="brand">
    <div class="favicon"></div>
    <div>
      <div style="font-weight:800">Космос: новости, 3D и Луна</div>
      <div class="hint">О сайте</div>
    </div>
  </div>
  <nav>
    <a class="navlink" href="{{ url_for('index') }}">Главная</a>
    <a class="navlink" href="{{ url_for('news_list') }}">Новости</a>
    <a class="navlink" href="{{ url_for('moon') }}">Луна</a>
    <a class="navlink active" href="{{ url_for('about') }}">О нас</a>
  </nav>
</header>
<main>
  <div style="max-width:800px;margin:18px auto">
    <div class="card">
      <h2>О нас</h2>
      <p>Этот сайт был разработан любителем космоса и астрономии в целом. Здесь вы можете постить новости, смотреть текущую фазу Луны и изучать космос с помощью 3D-модели на главной странице.</p>
      <p>Прошу подписаться на мой Telegram: <a href="https://t.me/SpaceNewsMoon" target="_blank">https://t.me/SpaceNewsMoon</a>. Там также будут выходить важные новости о космосе.</p>
      <p>Удачи и хорошего дня!</p>
    </div>
  </div>
</main>
<footer><div class="hint">© {{ year }}</div></footer>
</body>
</html>
"""

@app.route("/about")
def about():
    return render_template_string(ABOUT_PAGE, head=BASE_HEAD, title="О нас — " + APP_TITLE,
                                  year=datetime.utcnow().year)


# ---------------- Templates (single-file approach) ------------
BASE_HEAD = r"""
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>{{ title }}</title>
  <link rel="icon" type="image/png" href="https://i.pinimg.com/originals/94/f7/fe/94f7fe26904aa465de8644c7525b712b.jpg">
  <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;600;800&display=swap" rel="stylesheet">
  <style>
    :root{--bg:#0b1020;--card:#121931;--text:#e8eeff;--muted:#a6b1d0;--accent:#6aa7ff}
    *{box-sizing:border-box}
    body{margin:0;font-family:Inter,system-ui,-apple-system,Segoe UI,Roboto,Arial;background:radial-gradient(1000px 500px at 20% -10%,#17213f,transparent),var(--bg);color:var(--text)}
    header{padding:14px 18px;display:flex;gap:12px;align-items:center;justify-content:space-between;background:linear-gradient(180deg, rgba(11,16,32,.9), rgba(11,16,32,.4));position:sticky;top:0;z-index:5}
    .brand{display:flex;gap:12px;align-items:center}
    .favicon{width:44px;height:44px;border-radius:10px;background-image:url('https://i.pinimg.com/originals/94/f7/fe/94f7fe26904aa465de8644c7525b712b.jpg');background-size:cover;background-position:center;border:1px solid rgba(255,255,255,.06)}
    nav{display:flex;gap:10px}
    a.navlink{color:var(--muted);text-decoration:none;padding:8px 10px;border-radius:8px}
    a.navlink.active{color:var(--text);background:rgba(255,255,255,.02);border:1px solid rgba(255,255,255,.04)}
    main{padding:18px;max-width:1100px;margin:16px auto;}
    .grid{display:grid;grid-template-columns:1fr 360px;gap:16px}
    .card{background:linear-gradient(180deg,rgba(18,25,49,.9),rgba(18,25,49,.7));border:1px solid rgba(255,255,255,.06);border-radius:12px;padding:14px}
    .title{font-weight:700}
    .hint{font-size:13px;color:var(--muted)}
    form input,form textarea{width:100%;padding:10px;border-radius:8px;border:1px solid rgba(255,255,255,.06);background:#0e1530;color:var(--text)}
    form button{margin-top:8px;padding:10px 12px;border-radius:8px;background:linear-gradient(180deg,#2a3a73,#1f2b56);border:none;color:#fff}
    img.upl{max-width:100%;border-radius:8px;margin-top:8px}
    .news-item{padding:12px;border-radius:8px;background:rgba(255,255,255,.02);margin-bottom:12px}
    .meta{font-size:12px;color:var(--muted)}
    footer{padding:18px;text-align:center;color:var(--muted)}
    @media (max-width:900px){.grid{grid-template-columns:1fr}}
  </style>
"""

INDEX_PAGE = """
<!doctype html>
<html lang="ru">
<head>
{{ head|safe }}
</head>
<body>
  <header>
    <div class="brand">
      <div class="favicon" aria-hidden="true"></div>
      <div>
        <div style="font-weight:800">Космос: новости, 3D и Луна</div>
        <div class="hint">Обзор, новости и визуализации</div>
      </div>
    </div>
    <nav>
      <a class="navlink {% if active=='home' %}active{% endif %}" href="{{ url_for('index') }}">Главная</a>
      <a class="navlink {% if active=='news' %}active{% endif %}" href="{{ url_for('news_list') }}">Новости</a>
      <a class="navlink {% if active=='moon' %}active{% endif %}" href="{{ url_for('moon') }}">Луна</a>
    </nav>
  </header>

  <main>
    <!-- 3D-модель -->
    <div class="card">
      <h2>3D космос (NASA Eyes)</h2>
      <iframe src="https://eyes.nasa.gov/apps/solar-system/#/home" 
              style="width:100%;height:620px;border-radius:10px;border:none;margin-top:8px;" 
              allowfullscreen></iframe>
      <div class="hint">Источник: NASA Eyes on the Solar System. 
        Если iframe не загружается — проверьте блокировщики.</div>
    </div>

   <!-- Новости -->
<div class="card" style="margin-top:16px">
  <h2>Последние новости</h2>
  {% if latest_news %}
    {% for n in latest_news %}
      <article class="news-item">
        <div style="display:flex;justify-content:space-between;align-items:center">
          <div>
            <div style="font-weight:700">
              <a href="{{ url_for('view_news', nid=n['id']) }}" 
                 style="color:var(--text);text-decoration:none">{{ n['title'] }}</a>
            </div>
            <div class="meta">
              {{ n['created_at'] }} · Автор: {{ n['author'] or '—' }} · Фейк: {{ '%.1f' % n['fake_score'] }}
            </div>
          </div>
          {% if n['image'] %}
            <img src="{{ url_for('static', filename='uploads/' + n['image']) }}" 
                 alt="" style="width:72px;height:72px;object-fit:cover;border-radius:8px;margin-left:12px">
          {% endif %}
        </div>
      </article>
    {% endfor %}
  {% else %}
    <div class="hint">Новостей ещё нет.</div>
  {% endif %}
</div>

<!-- Раздел "О нас" после новостей -->
<div class="card" style="margin-top:16px">
  <h2>О нас</h2>
  <p>Этот сайт был разработан любителем космоса и астрономии в целом. Здесь вы можете постить новости, смотреть текущую фазу Луны и изучать космос с помощью 3D-модели на главной странице.</p>
  <p>Прошу подписаться на мой Telegram: <a href="https://t.me/SpaceNewsMoon" target="_blank">https://t.me/SpaceNewsMoon</a>. Там также будут выходить важные новости о космосе.</p>
  <p>Удачи и хорошего дня!</p>
</div>
  </main>

  <footer>
    <div class="hint">© {{ year }} · Демонстрационный проект</div>
    <div style="margin-top:8px">
      {% if music_exists %}
        <audio id="bgm" src="{{ url_for('static', filename='music/space.mp3') }}" autoplay loop controls style="width:260px;border-radius:8px;">Your browser does not support the audio element.</audio>
      {% else %}
        <div class="hint">Музыка не найдена в static/music/space.mp3</div>
      {% endif %}
    </div>
  </footer>
</body>
</html>
"""

NEWS_LIST = """
<!doctype html>
<html lang="ru">
<head>
{{ head|safe }}
</head>
<body>
<header>
  <div class="brand">
    <div class="favicon"></div>
    <div>
      <div style="font-weight:800">Космос: новости, 3D и Луна</div>
      <div class="hint">Новости и публикации</div>
    </div>
  </div>
  <nav>
    <a class="navlink" href="{{ url_for('index') }}">Главная</a>
    <a class="navlink active" href="{{ url_for('news_list') }}">Новости</a>
    <a class="navlink" href="{{ url_for('moon') }}">Луна</a>
  </nav>
</header>
<main>
  <div class="grid">
    <div>
      <div class="card">
        <h2>Новости</h2>
        <div style="margin-top:8px">
          <a href="{{ url_for('add_news') }}"><button>Добавить новость</button></a>
        </div>
        <div style="margin-top:12px">
          {% if items %}
            {% for n in items %}
              <article class="news-item">
                <div style="display:flex;justify-content:space-between;align-items:center">
                  <div>
                    <div style="font-weight:700"><a href="{{ url_for('view_news', nid=n['id']) }}" style="color:var(--text);text-decoration:none">{{ n['title'] }}</a></div>
                    <div class="meta">{{ n['created_at'] }} · Автор: {{ n['author'] or '—' }} · Фейк: {{ '%.1f' % n['fake_score'] }}</div>
                  </div>
                  {% if n['image'] %}
                    <img src="{{ url_for('static', filename='uploads/' + n['image']) }}" alt="" style="width:72px;height:72px;object-fit:cover;border-radius:8px;margin-left:12px">
                  {% endif %}
                </div>
              </article>
            {% endfor %}
          {% else %}
            <div class="hint">Новостей ещё нет.</div>
          {% endif %}
        </div>
      </div>
    </div>

    <aside>
      <div class="card">
        <h3>Про фейк-оценку</h3>
        <p class="hint">Оценка фейка — эвристическая. Если оценка > {{ threshold }}, ИИ автоматически удаляет новость.</p>
        <div style="margin-top:10px"><a href="{{ url_for('cleanup_route') }}"><button>В ручную: запустить проверку фейков</button></a></div>
      </div>
    </aside>
  </div>
</main>
<footer><div class="hint">© {{ year }}</div></footer>
</body>
</html>
"""

ADD_NEWS = """
<!doctype html>
<html lang="ru">
<head>
{{ head|safe }}
</head>
<body>
<header>
  <div class="brand">
    <div class="favicon"></div>
    <div>
      <div style="font-weight:800">Космос: новости, 3D и Луна</div>
      <div class="hint">Добавление новости</div>
    </div>
  </div>
  <nav>
    <a class="navlink" href="{{ url_for('index') }}">Главная</a>
    <a class="navlink" href="{{ url_for('news_list') }}">Новости</a>
    <a class="navlink" href="{{ url_for('moon') }}">Луна</a>
  </nav>
</header>
<main>
  <div style="max-width:900px;margin:18px auto">
    <div class="card">
      <h2>Добавить новость</h2>
      {% with messages = get_flashed_messages() %}
        {% if messages %}
          <div style="color:#ffd166">{{ messages[0] }}</div>
        {% endif %}
      {% endwith %}
      <form method="post" enctype="multipart/form-data">
        <label>Заголовок</label>
        <input name="title" required>
        <label style="margin-top:8px;display:block">Автор</label>
        <input name="author" placeholder="Имя автора (необязательно)">
        <label style="margin-top:8px;display:block">Текст (markdown)</label>
        <textarea name="content" rows="8" required></textarea>
        <label style="margin-top:8px;display:block">Приложить изображение (jpg/png/webp)</label>
        <input type="file" name="image" accept="image/*">
        <label style="margin-top:8px;display:block">Приложить аудиофайл (mp3/ogg/wav)</label>
        <input type="file" name="audio" accept="audio/*">
        <button type="submit">Опубликовать</button>
      </form>
    </div>
  </div>
</main>
<footer><div class="hint">© {{ year }}</div></footer>
</body>
</html>
"""


VIEW_NEWS = """
<!doctype html>
<html lang="ru">
<head>
{{ head|safe }}
</head>
<body>
<header>
  <div class="brand">
    <div class="favicon"></div>
    <div>
      <div style="font-weight:800">Космос: новости, 3D и Луна</div>
      <div class="hint">Просмотр новости</div>
    </div>
  </div>
  <nav>
    <a class="navlink" href="{{ url_for('index') }}">Главная</a>
    <a class="navlink" href="{{ url_for('news_list') }}">Новости</a>
    <a class="navlink" href="{{ url_for('moon') }}">Луна</a>
  </nav>
</header>
<main>
  <div style="max-width:900px;margin:18px auto">
    <div class="card">
      <h2>{{ item['title'] }}</h2>
      <div class="meta">{{ item['created_at'] }} · Автор: {{ item['author'] or '—' }} · Фейк-оценка: {{ '%.1f' % item['fake_score'] }}</div>
      {% if item['image'] %}
        <img class="upl" src="{{ url_for('static', filename='uploads/' + item['image']) }}" alt="">
      {% endif %}
      <div style="margin-top:12px">
        {% if markdown %}
          {{ markdown(item['content']) | safe }}
        {% else %}
          <pre style="white-space:pre-wrap">{{ item['content'] }}</pre>
        {% endif %}
      </div>
      {% if item['audio'] %}
        <div style="margin-top:12px">
          <audio controls src="{{ url_for('static', filename='uploads/' + item['audio']) }}" style="width:100%"></audio>
        </div>
      {% endif %}
      <div style="margin-top:12px">
        <form method="post" action="{{ url_for('delete_news', nid=item['id']) }}" onsubmit="return confirm('Удалить новость?');">
          <button type="submit" style="background:#bf3b3b">Удалить новость</button>
        </form>
      </div>
    </div>
  </div>
</main>
<footer><div class="hint">© {{ year }}</div></footer>
</body>
</html>
"""

MOON_PAGE = """
<!doctype html>
<html lang="ru">
<head>
{{ head|safe }}
</head>
<body>
<header>
  <div class="brand">
    <div class="favicon"></div>
    <div>
      <div style="font-weight:800">Космос: новости, 3D и Луна</div>
      <div class="hint">Фаза Луны</div>
    </div>
  </div>
  <nav>
    <a class="navlink" href="{{ url_for('index') }}">Главная</a>
    <a class="navlink" href="{{ url_for('news_list') }}">Новости</a>
    <a class="navlink active" href="{{ url_for('moon') }}">Луна</a>
  </nav>
</header>
<main>
  <div style="max-width:800px;margin:18px auto">
    <div class="card">
      <h2>Фаза Луны</h2>
      <div style="display:flex;gap:18px;align-items:center">
        <svg width="140" height="140" viewBox="0 0 100 100">
          <circle cx="50" cy="50" r="45" fill="#d9e1ff"/>
          {% set shift = (moon.cycle - 0.5) * 76 %}
          <ellipse cx="{{ 50 + shift }}" cy="50" rx="45" ry="45" fill="#0b1020"/>
        </svg>
        <div>
          <div style="font-weight:700">{{ moon.name }}</div>
          <div class="hint">Освещено: {{ moon.illum_pct }}% · Обновлено: {{ moon.utc }}</div>
        </div>
      </div>
    </div>
  </div>
</main>
<footer><div class="hint">© {{ year }}</div></footer>
</body>
</html>
"""

# ---------------- Routes -----------------

@app.route("/")
def index():
    latest_news = get_latest_news(limit=5)  # последние 5 новостей
    return render_template_string(
        INDEX_PAGE, 
        head=BASE_HEAD, 
        title=APP_TITLE, 
        active='home',
        year=datetime.utcnow().year, 
        music_exists=os.path.exists(DEST_MP3),
        latest_news=latest_news
    )

@app.route("/news")
def news_list():
    # cleanup fakes each time (ИИ-контроль)
    cleanup_fakes()
    db = get_db()
    cur = db.execute("SELECT id, title, author, created_at, image, fake_score FROM news ORDER BY id DESC")
    items = cur.fetchall()
    return render_template_string(NEWS_LIST, head=BASE_HEAD, title="Новости — " + APP_TITLE,
                                  items=items, year=datetime.utcnow().year, threshold=FAKE_THRESHOLD)

@app.route("/news/<int:nid>")
def view_news(nid):
    db = get_db()
    cur = db.execute("SELECT * FROM news WHERE id=?", (nid,))
    item = cur.fetchone()
    if not item:
        abort(404)
    return render_template_string(VIEW_NEWS, head=BASE_HEAD, title=item['title'] + " — " + APP_TITLE,
                                  item=item, markdown=(md.convert if md else None), year=datetime.utcnow().year)

@app.route("/add_news", methods=["GET", "POST"])
def add_news():
    if request.method == "GET":
        return render_template_string(ADD_NEWS, head=BASE_HEAD, title="Добавить новость — " + APP_TITLE,
                                      year=datetime.utcnow().year)
    # POST: process form
    title = request.form.get("title", "").strip()
    author = request.form.get("author", "").strip()
    content = request.form.get("content", "").strip()
    if not title or not content:
        flash("Заполните заголовок и текст")
        return redirect(url_for("add_news"))

    # save image if present
    image_file = request.files.get("image")
    image_name = None
    if image_file and image_file.filename:
        fname = secure_filename(image_file.filename)
        ext = fname.rsplit(".", 1)[-1].lower()
        if ext in ALLOWED_IMG:
            fname = datetime.utcnow().strftime("%Y%m%d%H%M%S_") + fname
            dest = os.path.join(app.config['UPLOAD_FOLDER'], fname)
            image_file.save(dest)
            image_name = fname
        else:
            flash("Недопустимый формат изображения")
            return redirect(url_for("add_news"))

    # save audio if present
    audio_file = request.files.get("audio")
    audio_name = None
    if audio_file and audio_file.filename:
        afname = secure_filename(audio_file.filename)
        aext = afname.rsplit(".", 1)[-1].lower()
        if aext in ALLOWED_AUDIO:
            afname = datetime.utcnow().strftime("%Y%m%d%H%M%S_") + afname
            adest = os.path.join(app.config['UPLOAD_FOLDER'], afname)
            audio_file.save(adest)
            audio_name = afname
        else:
            flash("Недопустимый формат аудиофайла")
            return redirect(url_for("add_news"))

    # compute fake score
# score = fake_score_heuristic(content, title)
    score = improved_fake_score(content, title)

    # If score > threshold => reject and delete uploaded files (ИИ блокирует)
    if score > FAKE_THRESHOLD:
        # remove uploaded files if any
        if image_name:
            try:
                os.remove(os.path.join(app.config['UPLOAD_FOLDER'], image_name))
            except Exception:
                pass
        if audio_name:
            try:
                os.remove(os.path.join(app.config['UPLOAD_FOLDER'], audio_name))
            except Exception:
                pass
        flash(f"Новость отклонена ИИ: слишком высокая вероятность фейка ({score:.1f}%).")
        return redirect(url_for("news_list"))

    created_at = datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")
    db = get_db()
    db.execute("INSERT INTO news (title, author, content, image, audio, created_at, fake_score) VALUES (?,?,?,?,?,?,?)",
               (title, author, content, image_name, audio_name, created_at, score))
    db.commit()
    flash(f"Новость опубликована (оценка фейка: {score:.1f}%).")
    return redirect(url_for("news_list"))

@app.route("/delete_news/<int:nid>", methods=["POST"])
def delete_news(nid):
    db = get_db()
    cur = db.execute("SELECT image, audio FROM news WHERE id=?", (nid,))
    r = cur.fetchone()
    if not r:
        flash("Новость не найдена")
        return redirect(url_for("news_list"))
    # delete files
    if r['image']:
        fp = os.path.join(app.config['UPLOAD_FOLDER'], r['image'])
        try:
            if os.path.exists(fp):
                os.remove(fp)
        except Exception:
            pass
    if r['audio']:
        fp = os.path.join(app.config['UPLOAD_FOLDER'], r['audio'])
        try:
            if os.path.exists(fp):
                os.remove(fp)
        except Exception:
            pass
    db.execute("DELETE FROM news WHERE id=?", (nid,))
    db.commit()
    flash("Новость удалена")
    return redirect(url_for("news_list"))

@app.route("/cleanup")
def cleanup_route():
    cleanup_fakes()
    flash("Проверка фейков выполнена (удалены подозрительные новости).")
    return redirect(url_for("news_list"))

@app.route("/moon")
def moon():
    return render_template_string(MOON_PAGE, head=BASE_HEAD, title="Луна — " + APP_TITLE,
                                  moon=moon_phase_info(), year=datetime.utcnow().year)

@app.route("/api/moon")
def api_moon():
    return jsonify(moon_phase_info())

@app.route("/health")
def health():
    return {"ok": True, "utc": datetime.utcnow().isoformat()}
# ---------------- Дополнения после health -----------------

@app.route("/cleanup_fakes_api")
def cleanup_fakes_api():
    """API для удаления фейковых новостей вручную."""
    cleanup_fakes()
    return {"ok": True, "message": "Подозрительные новости удалены", "utc": datetime.utcnow().isoformat()}

@app.route("/check_uploads")
def check_uploads():
    """Проверка существования всех файлов в базе. Возвращает список пропавших файлов."""
    db = get_db()
    cur = db.execute("SELECT id, image, audio FROM news")
    missing_files = []
    for r in cur.fetchall():
        for ftype in ["image", "audio"]:
            if r[ftype] and not os.path.exists(os.path.join(UPLOAD_FOLDER, r[ftype])):
                missing_files.append({"id": r["id"], "missing": r[ftype]})
    return jsonify({"ok": True, "missing_files": missing_files})

@app.route("/check_duplicates")
def check_duplicates():
    """Проверка на дубли заголовков в базе."""
    db = get_db()
    cur = db.execute("SELECT title, COUNT(*) as cnt FROM news GROUP BY title HAVING cnt > 1")
    duplicates = [{"title": r["title"], "count": r["cnt"]} for r in cur.fetchall()]
    return jsonify({"ok": True, "duplicates": duplicates})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
