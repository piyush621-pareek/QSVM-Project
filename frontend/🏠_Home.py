import streamlit as st

st.set_page_config(page_title="Q-Detect: Home", layout="wide")

st.title("⚛️ Q-Detect: Quantum Support Vector Machines")
st.subheader("Bridging Classical Data Science with Quantum Mechanics")

st.write("""
Welcome to the Q-Detect platform. This application is a full-stack exploration of **Quantum Kernel Estimation**. 
Please use the sidebar to navigate through the platform:
* **📖 Theory and Math:** Understand the underlying physics and equations.
* **🔬 The Lab:** Visualize the Bloch sphere, train hybrid models, and test datasets.
* **📚 Live Research:** Pull the latest academic papers directly from ArXiv.
""")

st.info("👈 Select a page from the sidebar to begin.")