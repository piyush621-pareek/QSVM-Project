import streamlit as st

st.set_page_config(page_title="Theory & Math", layout="wide")
st.title("📖 The Mathematics of Quantum SVMs")

col1, col2 = st.columns(2)

with col1:
    st.header("1. Classical SVMs")
    st.write("A standard SVM attempts to find the optimal hyperplane to separate data. The objective function is:")
    st.latex(r"\min_{w,b} \frac{1}{2} ||w||^2")
    st.write("Subject to the classification constraint:")
    st.latex(r"y_i(w \cdot x_i + b) \geq 1")
    
    st.subheader("The Classical Kernel Trick")
    st.write("When data is tangled, we map it to a higher dimension using a kernel function:")
    st.latex(r"K(x_i, x_j) = \phi(x_i)^T \phi(x_j)")

with col2:
    st.header("2. Quantum Kernel Estimation")
    st.write("Instead of a classical mapping, we encode data into a quantum state using a Quantum Feature Map (a series of unitary gates):")
    st.latex(r"|\Phi(x)\rangle = U_{\Phi(x)} |0\rangle^{\otimes n}")
    
    st.subheader("Measuring Similarity")
    st.write("The Quantum Kernel is calculated by finding the inner product of the two quantum states:")
    st.latex(r"K(x, z) = |\langle \Phi(x) | \Phi(z) \rangle|^2")

st.markdown("---")
st.info("💡 **Why do this?** Quantum Hilbert spaces are exponentially large. By estimating kernels on a quantum computer, we can explore feature spaces that are mathematically impossible to compute on classical hardware.")