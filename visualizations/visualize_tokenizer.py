# visualize_tokenizer.py
import streamlit as st
from tokenizers import Tokenizer
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import os
import json
from collections import Counter

# Load tokenizer from file
TOKENIZER_PATH = "tokenizer/tokenizer.json"
assert os.path.exists(TOKENIZER_PATH), "❌ tokenizer.json not found!"
tokenizer = Tokenizer.from_file(TOKENIZER_PATH)

# Load corpus
DATA_PATH = "data/corpus.txt"
with open(DATA_PATH, "r", encoding="utf-8") as f:
    raw_corpus = f.read()

def run():
    st.header("🧠 How Tokenization Works (Step-by-Step)")

    # Stage 0: Intro
    st.subheader("📘 What is Tokenization?")
    st.markdown("""
    Tokenization is how we teach an AI model to understand text. It breaks words into smaller parts called tokens, then converts them to numbers. Those numbers go into the model.

    For example:
    - Text: "Hello, world!"
    - Tokens: `["Hello", ",", "world", "!"]`
    - Token IDs: `[123, 4, 245, 6]`
    """)

    # Stage 1: Corpus view
    st.subheader("📓 What Data Was Used to Train the Tokenizer?")
    st.code(raw_corpus, language="text")

    # Word frequency chart
    words = raw_corpus.replace("\n", " ").split()
    freq = Counter(words)
    df_freq = pd.DataFrame(freq.items(), columns=["Word", "Frequency"]).sort_values(by="Frequency", ascending=False)
    st.markdown("**📊 Word Frequencies in Corpus** (Top 20)")
    st.dataframe(df_freq.head(20))
    st.bar_chart(df_freq.set_index("Word").head(20))

    # Full tokenization of corpus
    st.subheader("🔠 Full Corpus Tokenization")
    encoded_full = tokenizer.encode(raw_corpus)
    all_tokens = encoded_full.tokens
    all_ids = encoded_full.ids
    df_all_full = pd.DataFrame({"Token": all_tokens, "Token ID": all_ids})
    st.dataframe(df_all_full)
    st.download_button("⬇ Download All Token IDs as CSV", data=df_all_full.to_csv(index=False), file_name="corpus_token_ids.csv")

    st.markdown("### 📈 Frequency Bar Chart of All Tokens")
    token_freq = Counter(all_tokens)
    df_freq_tokens = pd.DataFrame({
        "Token": list(token_freq.keys()),
        "Frequency": list(token_freq.values())
    })
    fig_all = px.bar(df_freq_tokens, x="Token", y="Frequency", title="Token Frequency in Corpus")
    st.plotly_chart(fig_all)

    st.markdown("### 🫒 3D Token ID Visualization")
    fig3d_all = go.Figure(data=[go.Scatter3d(
        x=list(range(len(all_tokens))),
        y=all_ids,
        z=[token_freq[t] for t in all_tokens],
        mode='markers+text',
        marker=dict(size=6, color=all_ids, colorscale='Viridis', showscale=True),
        text=all_tokens,
        textposition="top center"
    )])
    fig3d_all.update_layout(
        title="🫒 Full Corpus 3D View of Token IDs",
        scene=dict(
            xaxis_title="Token Index",
            yaxis_title="Token ID",
            zaxis_title="Frequency"
        )
    )
    st.plotly_chart(fig3d_all)

    # Stage 2: Tokenizer settings
    with st.expander("⚙️ Tokenizer Settings"):
        st.markdown("""
        - Model: Byte Pair Encoding (BPE)
        - Pre-tokenizer: Whitespace
        - Vocab size: 1000
        - Min frequency: 2
        - Special tokens: [PAD], [UNK], [CLS], [SEP], [MASK]
        """)

    # Stage 3: Try your own input
    st.subheader("🧪 Try Typing Anything Below")
    text = st.text_input("Try a sentence:", "Hello, how are you?")

    if text.strip():
        output = tokenizer.encode(text)
        tokens = output.tokens
        ids = output.ids

        st.markdown("### 🔎 What Happened:")
        for i, (tok, tok_id) in enumerate(zip(tokens, ids)):
            st.write(f"{i+1}. Token: `{tok}` → ID: `{tok_id}`")

        df = pd.DataFrame({"Token": tokens, "Token ID": ids})
        st.dataframe(df)

        fig = px.bar(df, x="Token", y="Token ID", title="📊 Token ID Distribution")
        st.plotly_chart(fig)

        fig3d = go.Figure(data=[go.Scatter3d(
            x=list(range(len(tokens))),
            y=ids,
            z=[1]*len(tokens),
            mode='markers+text',
            marker=dict(size=10, color=ids, colorscale='Viridis', showscale=True),
            text=tokens,
            textposition="top center"
        )])
        fig3d.update_layout(
            title="🫒 Simulated 3D View of Token IDs",
            scene=dict(xaxis_title="Token Index", yaxis_title="Token ID", zaxis_title="Layer")
        )
        st.plotly_chart(fig3d)

        st.code(f"Token IDs: {ids}")
        st.code(f"Tokens: {tokens}")

    # Stage 4: Vocab and Merges
    with st.expander("📦 View Tokenizer Files"):
        if os.path.exists("tokenizer/vocab.json"):
            with open("tokenizer/vocab.json") as f:
                vocab = json.load(f)
            st.markdown("### First 20 Tokens in Vocab")
            st.json(dict(list(vocab.items())[:20]))
        if os.path.exists("tokenizer/merges.txt"):
            with open("tokenizer/merges.txt") as f:
                merges = f.readlines()
            st.markdown("### First 20 Merge Rules")
            st.code("".join(merges[:20]), language="text")

    # Stage 5: Merge Demo
    with st.expander("🧬 Example: How 'Hello' Might Be Merged"):
        st.markdown("If 'Hello' wasn’t common in training, it gets split:")
        st.code("""
        ["H", "e", "l", "l", "o"]
         → merge (H, e) → ["He", "l", "l", "o"]
         → merge (He, l) → ["Hel", "l", "o"]
         → merge (Hel, l) → ["Hell", "o"]
         → merge (Hell, o) → ["Hello"]
        """)
        st.markdown("This depends on the merge rules learned from the corpus.")

    # Stage 6: Learning Recap
    with st.expander("🧠 How Did The Tokenizer Learn?"):
        st.markdown("""
        1. Read the corpus character by character
        2. Count how often each pair appeared (like (l, o), (He, l))
        3. Merge the most frequent pair into one
        4. Repeat until vocab is full (1000 tokens)
        5. Save those merges in merges.txt
        """)

    # Stage 7: Merge History Viewer
    with st.expander("🔍 View Merge History of Any Token"):
        selected_token = st.selectbox("Choose a token to explore:", sorted(set(df_all_full["Token"])))
        if selected_token:
            with open("tokenizer/merges.txt") as f:
                merge_rules = [tuple(line.strip().split()) for line in f if line.strip() and not line.startswith("#")]

            def get_merge_steps(token, merges):
                steps = []
                chars = list(token)
                if len(chars) == 1:
                    return [chars]
                current = chars[:]
                steps.append(current[:])
                while True:
                    pairs = [(current[i], current[i+1]) for i in range(len(current)-1)]
                    found = False
                    for pair in pairs:
                        if pair in merges:
                            i = pairs.index(pair)
                            current = current[:i] + [''.join(pair)] + current[i+2:]
                            steps.append(current[:])
                            found = True
                            break
                    if not found:
                        break
                return steps

            steps = get_merge_steps(selected_token, merge_rules)
            st.markdown("### 🧬 Merge Steps:")
            for i, s in enumerate(steps):
                st.markdown(f"**Step {i}**: {s}")

    st.success("✅ Now you fully understand what happened when you typed 'Hello'. Tokenization isn't magic — it's pattern-based compression!")
