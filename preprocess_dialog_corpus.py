import os
import zipfile
import requests
import io

# Create data folder
DATA_DIR = "data"
os.makedirs(DATA_DIR, exist_ok=True)
output_path = os.path.join(DATA_DIR, "cleaned_corpus.txt")

def download(url, fname):
    print(f"📥 Downloading {fname}...")
    r = requests.get(url)
    r.raise_for_status()
    return r.content

# Process DailyDialog: replaces __eou__ with newlines
def add_dailydialog(text, corpus_file):
    lines = text.replace("__eou__", "\n").splitlines()
    for line in lines:
        line = line.strip()
        if len(line) > 5:
            corpus_file.write(line + "\n")

# Process Cornell Movie Dialogs: extracts speaker + utterance
def add_cornell_with_speakers(text, corpus_file):
    lines = text.splitlines()
    for line in lines:
        parts = line.split("+++$+++")
        if len(parts) >= 5:
            speaker = parts[-2].strip()       # BIANCA
            dialog = parts[-1].strip()        # "They do not!"
            if len(dialog) > 5 and speaker:
                corpus_file.write(f"{speaker}: {dialog}\n")

# Main merge logic
with open(output_path, "w", encoding="utf-8") as out:
    # 1. DailyDialog
    dd_url = "http://yanran.li/files/ijcnlp_dailydialog.zip"
    dd_bytes = download(dd_url, "DailyDialog.zip")
    with zipfile.ZipFile(io.BytesIO(dd_bytes)) as z:
        with z.open("ijcnlp_dailydialog/dialogues_text.txt") as f:
            add_dailydialog(f.read().decode("utf-8"), out)

    # 2. Cornell Dialogs with speaker names
    cm_url = "https://www.cs.cornell.edu/~cristian/data/cornell_movie_dialogs_corpus.zip"
    cm_bytes = download(cm_url, "CornellDialogs.zip")
    with zipfile.ZipFile(io.BytesIO(cm_bytes)) as z:
        with z.open("cornell movie-dialogs corpus/movie_lines.txt") as f:
            add_cornell_with_speakers(f.read().decode("iso-8859-1"), out)

print(f"✅ Done! Speaker-aware corpus saved to: {output_path}")
