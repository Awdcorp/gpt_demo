import os
import zipfile
import requests
import io

DATA_DIR = "data"
os.makedirs(DATA_DIR, exist_ok=True)

def download(url, fname):
    print(f"📥 Downloading {fname}...")
    r = requests.get(url)
    r.raise_for_status()
    return r.content

def add_text_to_corpus(text, corpus_file):
    for line in text.splitlines():
        line = line.strip()
        if len(line) > 5:
            corpus_file.write(line + "\n")

with open(os.path.join(DATA_DIR, "merged_corpus.txt"), "w", encoding="utf-8") as out:
    # 1. DailyDialog
    dd_url = "http://yanran.li/files/ijcnlp_dailydialog.zip"
    dd_bytes = download(dd_url, "DailyDialog.zip")
    with zipfile.ZipFile(io.BytesIO(dd_bytes)) as z:
        with z.open("ijcnlp_dailydialog/dialogues_text.txt") as f:
            add_text_to_corpus(f.read().decode("utf-8"), out)

    # 2. Cornell Movie Dialogs
    cm_url = "https://www.cs.cornell.edu/~cristian/data/cornell_movie_dialogs_corpus.zip"
    cm_bytes = download(cm_url, "CornellDialogs.zip")
    with zipfile.ZipFile(io.BytesIO(cm_bytes)) as z:
        with z.open("cornell movie-dialogs corpus/movie_lines.txt") as f:
            add_text_to_corpus(f.read().decode("iso-8859-1"), out)

print("✅ Done! Merged corpus saved to: data/merged_corpus.txt")
