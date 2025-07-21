# generate_toy_corpus.py

lines = [
    "How are you? I am fine.",
    "How are you? I am good.",
    "How are you? I'm doing well.",
    "How are you? I am great.",
    "How are you? I'm okay."
]

with open("data/toy_corpus.txt", "w", encoding="utf-8") as f:
    for _ in range(200):  # 5 lines × 200 = 1000 lines
        for line in lines:
            f.write(line + "\n")

print("✅ Toy corpus saved to 'data/toy_corpus.txt'")
