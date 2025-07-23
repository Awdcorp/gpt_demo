
import streamlit as st
import torch
import plotly.graph_objects as go
import plotly.express as px
import os
import numpy as np
from gpt2_tokenizer_wrapper import GPT2TokenizerWrapper
from config import tokenizer_path, vocab_size, embed_dim
from glob import glob
from sklearn.decomposition import PCA

def run():
    st.set_page_config(page_title="📈 Embedding Evolution", layout="wide")
    st.title("📈 Token Embedding Evolution Over Training Progress")

    with st.expander("📘 What does this visualization show? (Click to expand)"):
        st.markdown("""
        When you input a token like `hello`, the model looks up its embedding — a high-dimensional vector.

        - These embeddings are **learned during training** by updating a special matrix called the **embedding layer**.
        - For each training step, the model compares predicted vs actual token, and adjusts the embedding vectors.

        What you're seeing here:

        1. **How each dimension of the embedding changes over time** (line chart)
        2. **How far it drifts from the initial state** (cosine similarity)
        3. **Its trajectory in 3D embedding space** (PCA scatter)

        This helps understand how the model is learning to represent words.
        """)

    tokenizer = GPT2TokenizerWrapper()
    ckpt_paths = sorted(glob("checkpoints/embeddings_e*.pt"))
    if not ckpt_paths:
        st.warning("❌ No embedding checkpoints found in 'checkpoints/'")
        st.stop()

    token_text = st.text_input("🔍 Enter a token to visualize (e.g. 'hello')", value="hello")
    if token_text:
        token_ids = tokenizer.encode(token_text)
        if not token_ids:
            st.warning("❌ Token not found in vocab.")
            st.stop()

        token_id = token_ids[0]
        st.markdown(f"Token ID for `{token_text}`: `{token_id}`")

        vectors = []
        labels = []
        for path in ckpt_paths:
            matrix = torch.load(path)
            if token_id >= matrix.shape[0]:
                st.warning(f"⚠️ Token ID {token_id} out of bounds in {path}")
                continue
            vec = matrix[token_id].numpy()
            vectors.append(vec)
            label = os.path.splitext(os.path.basename(path))[0].replace("embeddings_", "")
            labels.append(label)

        vectors = np.stack(vectors)
        num_steps = vectors.shape[0]

        # --- Section 1: Line Chart of Embedding Dimensions ---
        st.subheader("📊 Change in Each Embedding Dimension Over Time")
        for_dim = st.slider("🧬 Show up to how many dimensions?", 16, embed_dim, 64)
        fig_dim = go.Figure()
        for d in range(for_dim):
            fig_dim.add_trace(go.Scatter(x=labels, y=vectors[:, d], mode="lines", name=f"dim {d}"))
        fig_dim.update_layout(title="Embedding Values Over Time", xaxis_title="Step", yaxis_title="Value")
        st.plotly_chart(fig_dim, use_container_width=True)
        st.markdown("")
        st.markdown("🧠 This chart displays how each dimension of the embedding vector for the selected token evolves over training steps. Each colored line represents one dimension in the embedding space (e.g., `dim 0`, `dim 1`, ...). This helps visualize how the model tweaks the internal vector representation for a word like `hello`.")
        st.markdown("""
                When you input a token like `hello`, the model looks up its embedding — a high-dimensional vector.

        - These embeddings are **learned during training** by updating a special matrix called the **embedding layer**.
        - For each training step, the model compares predicted vs actual token, and adjusts the embedding vectors.
                    
                        embedding("hello") → [0.12, -1.08, 0.45, ..., 0.77]  ← 64 numbers (if embed_dim=64)
                        This is a 64-dimensional embedding vector. Each of those 64 numbers is called a dimension of the embedding.

                    So in the chart:
                    You're tracking how each of these numbers (dim 0, dim 1, ..., dim 63) changes over time during training.
                    The y-axis shows the value of that dimension (e.g., dim 3 = -1.2 at step X).
                    The x-axis shows training steps (e.g., after 1200 batches or final).

        This helps understand how the model is learning to represent words.
        """)

        # --- Section 2: Cosine Similarity to Initial ---
        st.subheader("📏 Cosine Similarity to Initial Vector")
        base = vectors[0]
        norms = np.linalg.norm(vectors, axis=1)
        sim = np.dot(vectors, base) / (np.linalg.norm(base) * norms)
        fig_sim = px.line(x=labels, y=sim, labels={'x': 'Step', 'y': 'Cosine Similarity'})
        st.plotly_chart(fig_sim, use_container_width=True)
        st.markdown("📏 This line shows how much the embedding vector has changed compared to its starting point. Cosine similarity of 1.0 means no change — lower values mean the token has learned something new. It’s a measure of how 'different' the final embedding is from the initial random state.")

        # --- Section 3: PCA Movement ---
        st.subheader("🧭 Movement in Embedding Space (PCA)")
        pca = PCA(n_components=3)
        X_proj = pca.fit_transform(vectors)
        fig_3d = px.scatter_3d(
            x=X_proj[:, 0], y=X_proj[:, 1], z=X_proj[:, 2],
            title="Trajectory of Token Vector in PCA Space",
            labels={'x': 'PC1', 'y': 'PC2', 'z': 'PC3'},
            hover_name=labels,
            color=range(num_steps),
            color_continuous_scale='Viridis'
        )
        st.plotly_chart(fig_3d, use_container_width=True)
        st.markdown("🧭 This 3D scatter plot projects the high-dimensional embedding vector over time into 3D using PCA. You can see how the token 'moves' in vector space during training — closer points are similar states, and far ones indicate significant learning.")

        st.success("✅ Embedding evolution visualized!")

if __name__ == "__main__":
    run()
