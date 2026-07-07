#!/usr/bin/env python3
"""v2 idiolect-aware restyle: rewrite instruct answers as SHORT casual Денис-voice
messages, preserving correct facts/numbers (impersonation goal = pass as real Денис)."""
import json, os, time, urllib.request, concurrent.futures, threading

URL = "http://localhost:8092/v1/chat/completions"
BASE = "/home/labile/GolandProjects/tg2llm/finetune/data/premium_raw"
SRC = f"{BASE}/instruct_subset.jsonl"
OUT = f"{BASE}/instruct_restyled.jsonl"
DENIS = "Ты — Денис. Отвечай в своём обычном стиле, как в личной переписке."
RSYS = (
 "Перепиши ответ так, будто его в личной переписке в телеграме написал Денис — "
 "программист, пишет коротко и по делу.\n\n"
 "КАК ПИШЕТ ДЕНИС: строчными буквами, разговорно, кратко (обычно 1–2 коротких "
 "предложения), без markdown, без списков, без вступлений вроде «конечно» и «давай "
 "разберём». Иногда матерится и вставляет слова-паразиты (ну, вообще, просто, щас, хз, "
 "мб, походу, типа, реально, норм, блять, пиздец, ебать) — естественно, не в каждом "
 "предложении.\n\n"
 "ЖЁСТКИЕ ПРАВИЛА:\n"
 "- САМОЕ ВАЖНОЕ: сохрани правильный ответ и все ключевые факты и числа. Денис краток, "
 "но не ошибается — никогда не искажай цифры и суть.\n"
 "- Сожми до сути: выкинь воду, повторы, форматирование. Оставь ответ плюс максимум "
 "короткое пояснение.\n"
 "- Пиши ТОЛЬКО на русском. Без китайского, без английских вставок, без эмодзи-спама.\n"
 "- Верни только переписанный текст, без кавычек и комментариев.\n\n"
 "ПРИМЕР:\n"
 "вход: «Фотосинтез — это процесс, при котором растения преобразуют солнечный свет, "
 "углекислый газ и воду в глюкозу и кислород с помощью хлорофилла.»\n"
 "выход: ну это когда растения из света, со2 и воды делают глюкозу и кислород, хлорофилл помогает"
)

def extract(msgs):
    q = a = None
    for m in msgs:
        r = m.get("role"); c = (m.get("content") or "").strip()
        if not c: continue
        if r == "user": q = c
        elif r == "assistant": a = c
    return q, a

def restyle(answer, retries=4):
    body = json.dumps({"messages": [{"role": "system", "content": RSYS},
        {"role": "user", "content": "Перепиши это в стиле Дениса:\n\n" + answer}],
        "temperature": 0.6, "top_p": 0.9, "max_tokens": 700, "stream": False}).encode()
    last = None
    for k in range(retries):
        try:
            req = urllib.request.Request(URL, body, {"Content-Type": "application/json"})
            r = json.load(urllib.request.urlopen(req, timeout=200))
            return (r["choices"][0]["message"]["content"] or "").strip()
        except Exception as e:
            last = e; time.sleep(2 * (k + 1))
    raise last

done = set()
if os.path.exists(OUT):
    for ln in open(OUT):
        try: done.add(json.loads(ln)["idx"])
        except Exception: pass
todo = []
for i, ln in enumerate(open(SRC)):
    ln = ln.strip()
    if not ln or i in done: continue
    d = json.loads(ln); q, a = extract(d.get("messages", []))
    if q and a: todo.append((i, d.get("src"), q, a))
print(f"todo={len(todo)} done={len(done)}", flush=True)

lock = threading.Lock(); outf = open(OUT, "a"); n = [0]; fail = [0]
def work(it):
    i, src, q, a = it
    try:
        rs = restyle(a)
        if not rs:
            with lock: fail[0] += 1
            return
        rec = {"idx": i, "src": src, "messages": [
            {"role": "system", "content": DENIS},
            {"role": "user", "content": q},
            {"role": "assistant", "content": rs}]}
        with lock:
            outf.write(json.dumps(rec, ensure_ascii=False) + "\n"); outf.flush(); n[0] += 1
            if n[0] % 100 == 0: print(f"  {n[0]}/{len(todo)}", flush=True)
    except Exception as e:
        with lock: fail[0] += 1; print(f"  FAIL idx{i}: {e}", flush=True)

t = time.time()
with concurrent.futures.ThreadPoolExecutor(max_workers=8) as ex:
    list(ex.map(work, todo))
outf.close()
print(f"DONE new={n[0]} fail={fail[0]} time={time.time()-t:.0f}s", flush=True)
