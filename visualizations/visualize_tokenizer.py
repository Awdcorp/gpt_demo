# visualizations/visualize_tokenizer.py

import streamlit as st
from tokenizers import Tokenizer
import pandas as pd
import plotly.express as px
import os

# Load tokenizer from file
TOKENIZER_PATH = "tokenizer/tokenizer.json"
assert os.path.exists(TOKENIZER_PATH), "❌ tokenizer.json not found!"
tokenizer = Tokenizer.from_file(TOKENIZER_PATH)

def run():
    st.header("🔤 Tokenizer Visualization")
    st.write("Enter any input text to see how it is tokenized using your trained BPE tokenizer.")

    # User input
    text = st.text_input("Enter text:", "The meaning of life is 42")

    if not text.strip():
        st.warning("Please enter some text to tokenize.")
        return

    # Encode text
    output = tokenizer.encode(text)
    tokens = output.tokens
    ids = output.ids

    st.subheader("📋 Tokenized Output")
    df = pd.DataFrame({
        "Token": tokens,
        "Token ID": ids
    })
    st.dataframe(df)

    # Plot as bar chart
    fig = px.bar(df, x="Token", y="Token ID", title="📊 Token IDs")
    st.plotly_chart(fig)

    # Show raw lists
    st.code(f"Token IDs: {ids}")
    st.code(f"Tokens: {tokens}")
