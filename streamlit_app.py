import streamlit as st

# Set page title and layout
st.set_page_config(page_title="MiniGPT Visualizer", layout="wide")
st.title("🧠 MiniGPT Model Visualizer")

# Sidebar stage selection
stage = st.sidebar.radio("📌 Choose a Stage to Visualize", [
    "Embeddings",
    "Tokenizer",
    "Embedding Projections (PCA/t-SNE)",
    "Embedding Evolution",  # ✅ NEW ENTRY
    "Transformer",
    "Step-by-Step Transformer",
    "Full Flow",
    "Prediction + Logits"
])

# Load and run the selected visual module
if stage == "Tokenizer":
    from visualizations import visualize_tokenizer
    visualize_tokenizer.run()

elif stage == "Embeddings":
    from visualizations import visualize_embeddings
    visualize_embeddings.run()

elif stage == "Embedding Projections (PCA/t-SNE)":
    from visualizations import visualize_embedding_projection
    visualize_embedding_projection.run()

elif stage == "Embedding Evolution":
    from visualizations import visualize_embedding_evolution
    visualize_embedding_evolution.run()

elif stage == "Transformer":
    from visualizations import visualize_transformer
    visualize_transformer.run()

elif stage == "Step-by-Step Transformer":
    from visualizations import visualize_step_by_step
    visualize_step_by_step.run()

elif stage == "Full Flow":
    from visualizations import visualize_full_flow
    visualize_full_flow.run()

elif stage == "Prediction + Logits":
    from visualizations import visualize_logits
    visualize_logits.run()

else:
    st.warning("❌ Invalid stage selected.")
