import streamlit as st
import torch
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
from tokenizer.tokenizer import GPTTokenizer
from model.gpt_model import MiniGPT
from config import *  # model hyperparams

def run():
    st.set_page_config(page_title="🧠 Embedding Visualizer", layout="wide")
    st.title("🧠 Embedding Layer Visualizer")

    # Load model and tokenizer
    tokenizer = GPTTokenizer(tokenizer_path)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = MiniGPT(vocab_size, embed_dim, max_seq_len, num_heads, ff_dim, num_layers).to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()

    # Get embedding weight matrix
    embedding_matrix = model.backbone.embedding.token_embedding.weight.detach().cpu().numpy()

    # --- Section 1: Explore Token Embedding ---
    st.header("🔍 1. Explore a Token's Embedding Vector")
    token = st.text_input("Enter a token (e.g. 'Hello')", value="Hello")

    if token:
        ids = tokenizer.encode(token)
        if not ids:
            st.warning("❌ Token not found in vocab.")
        else:
            token_id = ids[0]
            vector = embedding_matrix[token_id]

            st.markdown(f"**Token ID**: `{token_id}`")
            st.markdown(f"**Vector Shape**: `{vector.shape}`")

            df = pd.DataFrame({"Dim": list(range(len(vector))), "Value": vector})
            fig = px.line(df, x="Dim", y="Value", title=f"📈 Embedding Vector for '{token}' (ID {token_id})")
            st.plotly_chart(fig, use_container_width=True)

            vector_str = np.array2string(vector, precision=6, separator=', ', suppress_small=True, max_line_width=150)
            st.code(vector_str, language="json")

    # --- Section 2: Embeddings for Input Sentence ---
    st.header("📝 2. Visualize Embeddings for a Sentence")
    sentence = st.text_input("Type a sentence:", value="Hello, how are you?")

    if sentence.strip():
        ids = tokenizer.encode(sentence)
        tokens = [tokenizer.decode([i]) for i in ids]

        token_tensor = torch.tensor([ids], dtype=torch.long).to(device)
        with torch.no_grad():
            embed_out = model.backbone.embedding(token_tensor)

        embed_out = embed_out[0].cpu().numpy()  # [seq_len, embed_dim]

        st.markdown("### 🔢 Embedding Matrix")
        st.dataframe(pd.DataFrame(embed_out, index=tokens))

        st.markdown("### 🧊 3D Plot of Token Embeddings")
        x, y, z = [], [], []
        for i, vec in enumerate(embed_out):
            x.append(i)
            y.append(np.mean(vec))
            z.append(np.linalg.norm(vec))

        fig3d = go.Figure(data=[
            go.Scatter3d(
                x=x,
                y=y,
                z=z,
                mode='markers+text',
                text=tokens,
                marker=dict(size=10, color=list(range(len(tokens))), colorscale='Viridis', showscale=True)
            )
        ])
        fig3d.update_layout(scene=dict(
            xaxis_title="Token Index",
            yaxis_title="Mean Value",
            zaxis_title="Vector Norm"
        ), height=600)
        st.plotly_chart(fig3d, use_container_width=True)

    # --- Section 3: Cosine Similarity ---
    st.header("📐 3. Compare Token Similarity")
    t1 = st.text_input("Token 1", value="hello")
    t2 = st.text_input("Token 2", value="hi")

    if t1 and t2:
        id1_list = tokenizer.encode(t1)
        id2_list = tokenizer.encode(t2)
        if id1_list and id2_list:
            id1 = id1_list[0]
            id2 = id2_list[0]
            v1 = embedding_matrix[id1]
            v2 = embedding_matrix[id2]
            cos_sim = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
            st.markdown(f"**Cosine Similarity** between `{t1}` and `{t2}`: `{cos_sim:.4f}`")
        else:
            st.warning("❌ One or both tokens not found in vocab.")

    # --- Section 4: Embedding Projection (Optional) ---
    st.header("🧭 4. PCA Projection of All Embeddings")
    if st.button("🔄 Run PCA (top 500 tokens)"):
        from sklearn.decomposition import PCA

        top_ids = np.arange(min(500, vocab_size))
        X = embedding_matrix[top_ids]
        pca = PCA(n_components=3)
        X_proj = pca.fit_transform(X)

        tokens = [tokenizer.decode([i]) for i in top_ids]
        fig_pca = px.scatter_3d(
            x=X_proj[:,0], y=X_proj[:,1], z=X_proj[:,2], text=tokens,
            title="PCA Projection of Top Token Embeddings"
        )
        st.plotly_chart(fig_pca, use_container_width=True)

    st.success("✅ Embedding visualizer ready! You can now explore how text is converted into vectors.")

    # --- Section 5: Simulate Embedding Update ---
    st.header("🔄 5. Simulate How an Embedding Vector Is Learned")

    st.markdown(
        """
        This section shows **how a token's vector (like -2.192)** is actually learned using gradient descent.
        """
    )

    sim_dim = st.slider("🔢 Embedding Dimension", 16, 128, 64, step=16)  # 🔁 Renamed from embed_dim
    lr = st.number_input("📉 Learning Rate", value=0.1, format="%.3f")

    np.random.seed(42)
    initial_vector = np.random.randn(sim_dim)
    gradient = np.random.randn(sim_dim)

    # 🚀 Embedding update step
    new_vector = initial_vector - lr * gradient

    # 📈 Plot vector updates
    st.subheader("📈 Change in Embedding Vector (Initial → Updated)")
    fig = go.Figure()
    fig.add_trace(go.Scatter(y=initial_vector, mode="lines+markers", name="Initial Vector"))
    fig.add_trace(go.Scatter(y=gradient, mode="lines+markers", name="Gradient"))
    fig.add_trace(go.Scatter(y=new_vector, mode="lines+markers", name="Updated Vector"))
    fig.update_layout(xaxis_title="Dimension", yaxis_title="Value")
    st.plotly_chart(fig, use_container_width=True)

    # 🔢 Updated vector values
    st.markdown("### 🔢 Updated Embedding Vector")
    vector_str1 = np.array2string(new_vector, precision=6, separator=', ', suppress_small=True, max_line_width=150)
    st.code(vector_str1, language="json")

    # ✅ Explanation
    st.markdown(
        """
        ### ✅ Explanation:
        - A token like `"he"` starts with a **random vector**
        - During training, gradients are calculated from **prediction errors**
        - The update rule is:  
        ```
        new_vector = old_vector - learning_rate × gradient
        ```
        - After many updates, the embedding learns useful meaning (e.g., values like `-2.192`)
        """
    )

if __name__ == "__main__":
    run()