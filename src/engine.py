import os
import re
import csv
import zipfile
import random
import unicodedata
import subprocess
from difflib import SequenceMatcher

import pandas as pd
from PyPDF2 import PdfReader


# =========================
# Texto / Similaridade
# =========================
def normalize_text(s: str) -> str:
    s = str(s).strip().lower()
    s = ''.join(c for c in unicodedata.normalize('NFD', s) if unicodedata.category(c) != 'Mn')
    s = re.sub(r"[^\w\s']", " ", s)  # mantém apóstrofo
    s = re.sub(r"\s+", " ", s).strip()
    return s


def similarity(a: str, b: str) -> float:
    return SequenceMatcher(None, normalize_text(a), normalize_text(b)).ratio()


def is_correct(user_answer: str, accepted_answers, threshold=0.86):
    best_ans, best_sc = "", 0.0
    for ans in accepted_answers:
        sc = similarity(user_answer, ans)
        if sc > best_sc:
            best_ans, best_sc = ans, sc
    return (best_sc >= threshold), best_ans, best_sc


def tokenize_en(s: str):
    s = normalize_text(s)
    return re.findall(r"[a-z]+(?:'[a-z]+)?", s)


# =========================
# Download helpers
# =========================
UA = "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120 Safari/537.36"

def wget(url: str, out_path: str, tries=10) -> bool:
    cmd = [
        "wget", "-O", out_path, url,
        "--user-agent", UA,
        f"--tries={tries}", "--waitretry=3", "--timeout=20", "--retry-connrefused"
    ]
    r = subprocess.run(cmd, capture_output=True, text=True)
    return os.path.exists(out_path) and os.path.getsize(out_path) > 50_000


# =========================
# Oxford 3000 (A1–B1)
# =========================
OXFORD_URLS = [
    "https://www.oxfordlearnersdictionaries.com/external/pdf/wordlists/oxford-3000-5000/The_Oxford_3000_by_CEFR_level.pdf",
    "https://www.oxfordlearnersdictionaries.co.uk/us/external/pdf/wordlists/oxford-3000-5000/The_Oxford_3000_by_CEFR_level.pdf",
    "https://raw.githubusercontent.com/XA2005/CEFR-World-List/main/The_Oxford_3000_by_CEFR_level.pdf",
]

LEVELS = ["A1", "A2", "B1", "B2"]


def download_oxford_pdf(pdf_path="oxford_3000_cefr.pdf") -> str:
    if os.path.exists(pdf_path):
        os.remove(pdf_path)

    for url in OXFORD_URLS:
        if wget(url, pdf_path):
            return pdf_path

    raise RuntimeError(
        "Falha ao baixar o PDF do Oxford 3000. "
        "Tente novamente ou faça upload manual do arquivo com o nome oxford_3000_cefr.pdf."
    )


def extract_pages_text(pdf_path: str):
    reader = PdfReader(pdf_path)
    return [(p.extract_text() or "") for p in reader.pages]


def parse_oxford_words_by_level(pages_text):
    text = "\n".join(pages_text).replace("\xa0", " ")
    text = re.sub(r"[ \t]+", " ", text)

    found = []
    for lv in LEVELS:
        m = re.search(rf"\b{lv}\b", text)
        if m:
            found.append((lv, m.start()))
    found.sort(key=lambda x: x[1])

    if len(found) < 2:
        return {}

    blocks = {}
    for i, (lv, start) in enumerate(found):
        end = found[i + 1][1] if i + 1 < len(found) else len(text)
        blocks[lv] = text[start:end]

    pos_pattern = r"\b(adj|adv|aux|conj|det|modal|n|num|prep|pron|v)\.?\b"

    def extract_items(block: str):
        b = re.sub(pos_pattern, " ", block, flags=re.I)
        b = re.sub(r"\b\d+\b", " ", b)
        b = re.sub(r"\s+", " ", b).strip()

        tokens = re.findall(r"[A-Za-z][A-Za-z'\-]*", b)
        out, seen = [], set()
        for t in tokens:
            w = t.lower()
            if len(w) >= 2 and w not in seen:
                seen.add(w)
                out.append(w)
        return out

    return {lv: extract_items(bl) for lv, bl in blocks.items()}


def build_oxford_a1_b1_set(pdf_path="oxford_3000_cefr.pdf"):
    pages = extract_pages_text(pdf_path)
    words_by_level = parse_oxford_words_by_level(pages)

    A1_words = words_by_level.get("A1", [])
    A2_words = words_by_level.get("A2", [])
    B1_words = words_by_level.get("B1", [])

    bank = list(dict.fromkeys(A1_words + A2_words + B1_words))

    # Para filtrar frases: apenas tokens padrão (a-z e apóstrofo)
    oxford_set = set([w for w in bank if re.fullmatch(r"[a-z]+(?:'[a-z]+)?", w)])
    return oxford_set, {"A1": A1_words, "A2": A2_words, "B1": B1_words}


# =========================
# Frases EN-PT (ManyThings / Tatoeba)
# =========================
MANYTHINGS_ZIP_URL = "https://www.manythings.org/anki/por-eng.zip"


def download_and_load_sentence_pairs(zip_path="por-eng.zip", extract_dir="tatoeba_por_eng"):
    if not os.path.exists(zip_path):
        if not wget(MANYTHINGS_ZIP_URL, zip_path):
            # wget usa limite de tamanho mínimo; zip é pequeno, então validamos diferente:
            if not os.path.exists(zip_path) or os.path.getsize(zip_path) < 10_000:
                raise RuntimeError("Falha ao baixar por-eng.zip do ManyThings.")

    if not os.path.exists(extract_dir):
        os.makedirs(extract_dir, exist_ok=True)
        with zipfile.ZipFile(zip_path, "r") as z:
            z.extractall(extract_dir)

    txt_files = [f for f in os.listdir(extract_dir) if f.endswith(".txt")]
    if not txt_files:
        raise RuntimeError("Não encontrei .txt dentro do zip por-eng.zip")

    pairs_path = os.path.join(extract_dir, txt_files[0])
    df = pd.read_csv(pairs_path, sep="\t", header=None, names=["id", "en", "pt"], quoting=3)
    df = df.dropna(subset=["en", "pt"])
    df["en"] = df["en"].astype(str).str.strip()
    df["pt"] = df["pt"].astype(str).str.strip()
    return df


def filter_sentences_only_oxford(df, oxford_set, min_words=3, max_words=12):
    def word_count_simple(s):
        return len(re.findall(r"[A-Za-zÀ-ÿ']+", str(s)))

    def looks_ok_en(s):
        s = str(s)
        if re.search(r"[_#@/\\\[\]{}<>]", s):
            return False
        return True

    def only_oxford(en_sentence):
        toks = tokenize_en(en_sentence)
        if not toks:
            return False
        return all(t in oxford_set for t in toks)

    filtered = df[
        df["en"].apply(looks_ok_en) &
        df["en"].apply(word_count_simple).between(min_words, max_words) &
        df["pt"].apply(word_count_simple).between(min_words, max_words) &
        df["en"].apply(only_oxford)
    ].copy()

    filtered = filtered.drop_duplicates(subset=["en", "pt"])
    return filtered


# =========================
# Teste (10 palavras + 10 frases)
# =========================
def run_test(oxford_set, filtered_sentences_df, n_words=10, n_sentences=10, threshold=0.86, seed=None):
    if seed is not None:
        random.seed(seed)

    chosen_words = random.sample(list(oxford_set), k=min(n_words, len(oxford_set)))
    chosen_rows = filtered_sentences_df.sample(n=min(n_sentences, len(filtered_sentences_df)), random_state=seed)
    chosen_sentences = list(zip(chosen_rows["en"].tolist(), chosen_rows["pt"].tolist()))

    total = len(chosen_words) + len(chosen_sentences)
    correct = 0

    print("\n🟦 TESTE A1–B1 — 10 palavras + 10 frases (randômicas filtradas)")
    print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n")

    # Palavras (autoavaliação)
    print("🟩 Parte 1 — Palavras (autoavaliação)\n")
    for i, w in enumerate(chosen_words, 1):
        _ = input(f"{i:02d}) Traduza a palavra: '{w}' → ")
        s = input("   Você considera que acertou? (s/n) → ").strip().lower()
        if s.startswith("s"):
            correct += 1
            print("   ✅ Marcado como correto.\n")
        else:
            print("   ❌ Marcado como incorreto.\n")

    # Frases (correção automática)
    print("\n🟨 Parte 2 — Frases (randômicas, só vocabulário Oxford A1–B1)\n")
    base = len(chosen_words)
    for j, (en, pt) in enumerate(chosen_sentences, 1):
        user = input(f"{base + j:02d}) Traduza: \"{en}\" → ")
        ok, best, sc = is_correct(user, [pt], threshold=threshold)
        if ok:
            correct += 1
            print(f"   ✅ Correto! (similaridade {sc:.2f})\n")
        else:
            print(f"   ❌ Gabarito: {pt} (similaridade: {sc:.2f})\n")

    grade = (correct / total) * 10 if total else 0.0
    print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    print(f"📌 Acertos: {correct}/{total}")
    print(f"🏁 Nota final: {grade:.1f} / 10")
    print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")


def prepare_all(min_words=3, max_words=12):
    pdf = download_oxford_pdf()
    oxford_set, counts = build_oxford_a1_b1_set(pdf)
    df_pairs = download_and_load_sentence_pairs()
    filtered = filter_sentences_only_oxford(df_pairs, oxford_set, min_words=min_words, max_words=max_words)
    return oxford_set, filtered, counts
