"""Проверка собранного сайта (_site) после quarto render.

    python scripts/check_site.py

Ищет: формулы, которые не распознались (остатки $ и \\frac в тексте),
страницы без заголовка (<title> из имени временного файла), ошибки
исполнения ячеек (Traceback в выводе). Код возврата 1, если что-то найдено.
"""
import glob
import html
import re
import sys

problems = []
for f in sorted(glob.glob("_site/**/*.html", recursive=True)):
    s = open(f, encoding="utf-8").read()
    title = re.search(r"<title>([^<]*)</title>", s)
    if not title or "quarto-input" in title.group(1):
        problems.append(f"{f}: нет заголовка страницы")
    if re.search(r"Traceback \(most recent call last\)|Error</span>:", s):
        problems.append(f"{f}: в выводе ячейки ошибка (Traceback)")
    t = re.sub(r'<span class="math[^"]*">.*?</span>', "", s, flags=re.S)
    t = re.sub(r"<(script|style|code|pre)[^>]*>.*?</\1>", "", t, flags=re.S)
    t = html.unescape(re.sub(r"<[^>]+>", " ", t))
    for m in re.finditer(r".{0,40}(?:\$|\\(?:frac|lvert|rangle|mathrm|sqrt)\b).{0,40}", t):
        problems.append(f"{f}: нераспознанная формула: …{m.group(0).strip()}…")

print("\n".join(problems) if problems else "сайт: замечаний нет")
sys.exit(1 if problems else 0)
