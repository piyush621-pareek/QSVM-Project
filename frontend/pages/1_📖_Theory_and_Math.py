import streamlit as st
import numpy as np
import plotly.graph_objects as go
from sklearn.datasets import make_circles

st.set_page_config(page_title="Theory & Math - Q-Detect", layout="wide")

# --- Title and Intro ---
st.title("📖 Mathematical Foundations of Q-Detect")
st.write("""
Q-Detect bridges the gap between classical Statistical Learning Theory and Quantum Information Science. 
To understand how we classify data using qubits, we must first look at the geometry of Support Vector Machines.
""")

st.markdown("---")

# --- Section 1: Classical SVM & The Kernel Trick ---
st.header("1. Classical Support Vector Machines (SVM)")

col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("The Linear Hyperplane")
    st.write("""
    In a binary classification problem, we seek a **hyperplane** that separates two classes with the maximum margin. 
    The decision boundary is defined by the equation:
    """)
    st.latex(r"f(x) = w^T x + b = 0")
    st.write("We minimize the weight vector to maximize the margin:")
    st.latex(r"\min_{w,b} \frac{1}{2} ||w||^2 \quad \text{s.t.} \quad y_i(w^T x_i + b) \geq 1")

with col2:
    st.subheader("The Kernel Trick")
    st.write("""
    When data is not linearly separable (like concentric circles), we use a **Kernel Function** $K(x, z)$ 
    to map data into a higher-dimensional Feature Space $\mathcal{F}$ where a linear separation exists.
    """)
    st.latex(r"K(x, z) = \langle \phi(x), \phi(z) \rangle_{\mathcal{F}}")
    st.info("💡 Concept: We 'lift' the data from 2D to 3D to make it sliceable by a flat plane.")

# --- Visual Demo of the Kernel Trick ---
st.write("### Pictorial Representation: Lifting Data to 3D")


# Let's actually generate a 3D math plot to show the "Lifting"
X, y = make_circles(n_samples=400, factor=0.3, noise=0.05, random_state=42)
# The "Radial Basis Function" (RBF) mapping simulation: z = x^2 + y^2
z = X[:, 0]**2 + X[:, 1]**2 

fig_math = go.Figure()
fig_math.add_trace(go.Scatter3d(
    x=X[:, 0], y=X[:, 1], z=z,
    mode='markers',
    marker=dict(size=3, color=y, colorscale='RdBu', opacity=0.8),
    name="Lifted Feature Space"
))
fig_math.update_layout(title="Mathematical Transformation into Higher Dimensions", scene=dict(
    xaxis_title='X', yaxis_title='Y', zaxis_title='Z (Mapped Feature)'))
st.plotly_chart(fig_math, use_container_width=True)

st.markdown("---")

# --- Section 2: Quantum Support Vector Machines ---
st.header("2. The Quantum Advantage")

col3, col4 = st.columns(2)

with col3:
    st.subheader("Quantum Feature Maps")
    st.write("""
    In a **QSVM**, we replace the classical function $\phi(x)$ with a **Quantum Circuit** $U_{\Phi(x)}$. 
    This circuit encodes classical data $x$ into a quantum state vector (Statevector) in a **Hilbert Space**:
    """)
    st.latex(r"|\Phi(x)\rangle = U_{\Phi(x)} |0\rangle^{\otimes n}")
    st.write("""
    Because a Hilbert Space with $n$ qubits has $2^n$ dimensions, we can explore massive 
    mathematical spaces that are classically intractable.
    """)

with col4:
    st.subheader("Quantum Kernel Estimation")
    st.write("""
    The similarity between two data points is calculated by measuring the 'overlap' (fidelity) 
    between their two quantum states.
    """)
    st.latex(r"K(x, z) = |\langle \Phi(x) | \Phi(z) \rangle|^2")
    st.write("""
    We compute this by running the circuit for $x$, then the inverse for $z$, and 
    measuring the probability of the qubits returning to the $|0\rangle$ state.
    """)

st.markdown("---")

# --- Section 3: The Bloch Sphere Representation ---
st.header("3. Geometric Representation: The Bloch Sphere")


st.write("""
A single qubit can be visualized as a point on the surface of a unit sphere. 
When Q-Detect processes a single data point, it performs a rotation:
""")
st.latex(r"|\psi\rangle = \cos(\theta/2)|0\rangle + e^{i\phi}\sin(\theta/2)|1\rangle")
st.write("""
In our application, we map your normalized data values to the angle $\theta$. 
This changes the 'coordinates' of the arrow on the sphere, effectively 'positioning' 
your data in the quantum realm for classification.
""")

st.success("🎯 **Final Summary:** By combining Scikit-Learn's optimization with Qiskit's Quantum Kernels, we solve non-linear problems (like the circles shown above) with higher dimensional precision than standard linear algorithms.")
