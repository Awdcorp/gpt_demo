import streamlit as st

# Set page title and layout
st.set_page_config(page_title="MiniGPT Visualizer", layout="wide")
st.title("🧠 MiniGPT Model Visualizer")

# Sidebar stage selection
stage = st.sidebar.radio("📌 Choose a Stage to Visualize", [
    "Embeddings",
    "Tokenizer",
    "Embedding Evolution",  # ✅ NEW ENTRY
    "Transformer",
    "Feedforward Layer Visualizer",
    "logits softmax",
    "Full Flow",
])

# Load and run the selected visual module
if stage == "Tokenizer":
    from visualizations import visualize_tokenizer
    visualize_tokenizer.run()

elif stage == "Embeddings":
    from visualizations import visualize_embeddings
    visualize_embeddings.run()

elif stage == "Embedding Evolution":
    from visualizations import visualize_embedding_evolution
    visualize_embedding_evolution.run()

elif stage == "Transformer":
    from visualizations import visualize_transformer
    visualize_transformer.run()

elif stage == "Feedforward Layer Visualizer":
    from visualizations import visualize_feedforward
    visualize_feedforward.run()

elif stage == "logits softmax":
    from visualizations import visualize_logits_softmax
    visualize_logits_softmax.run()

elif stage == "Full Flow":
    from visualizations import visualize_full_flow
    visualize_full_flow.run()

else:
    st.warning("❌ Invalid stage selected.")
