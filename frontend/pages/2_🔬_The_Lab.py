
import streamlit as st
import requests
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

BACKEND_URL = "http://127.0.0.1:8000"
st.set_page_config(page_title="The Lab", layout="wide")
st.title("🔬 The Quantum ML Lab")

tab_viz, tab_train = st.tabs(["Data & Visual Problem Setup", "Train Hybrid Models"])

with tab_viz:
    st.header("Visualizing the Non-Linear Problem")
    st.write("Select a dataset below. If you choose 'Concentric Circles', you will see a problem that a classical linear SVM cannot solve without mapping to higher dimensions.")
    
    dataset_view_choice = st.radio("Select Dataset to Visualize:", ["Financial Fraud (Simulated)", "Concentric Circles (Hard)"])
    
    try:
        # Pass the choice to the backend
        viz_data = requests.get(f"{BACKEND_URL}/visualize-data/?dataset={dataset_view_choice}").json()
        df = pd.DataFrame({
            "Feature 1": viz_data["x_coords"], "Feature 2": viz_data["y_coords"], 
            "Class": ["Class 1 (Red)" if y == 1 else "Class 0 (Blue)" for y in viz_data["labels"]]
        })
        
        colA, colB = st.columns([2, 1])
        with colA:
            fig_scatter = px.scatter(df, x="Feature 1", y="Feature 2", color="Class", 
                                     color_discrete_map={"Class 1 (Red)": "red", "Class 0 (Blue)": "blue"},
                                     title=f"2D Representation of {dataset_view_choice}")
            st.plotly_chart(fig_scatter, use_container_width=True)
            
        with colB:
            st.markdown("### The Problem")
            if dataset_view_choice == "Concentric Circles (Hard)":
                st.error("❌ **Linear Failure:** Try to draw a single straight line to separate the red dots from the blue dots. You can't. A classical linear SVM will fail completely here.")
                st.success("✅ **Quantum Solution:** The quantum feature map will encode these 2D coordinates into the rotation angles of qubits (like the sphere below), lifting the data into a high-dimensional mathematical space where a flat plane can easily slice between the red and blue data.")
            else:
                st.warning("⚠️ **Complex Clustering:** The data is highly overlapping. A classical SVM will struggle to draw a clean boundary without overfitting.")
                
        st.markdown("---")
        st.subheader("Quantum Encoding (The Bloch Sphere)")
        st.write("This represents one data point from the graph above, mathematically mapped onto a single Qubit.")
        u, v = viz_data["bloch_vector"][0], viz_data["bloch_vector"][2]
        fig_bloch = go.Figure()
        fig_bloch.add_trace(go.Mesh3d(x=[0], y=[0], z=[0], alphahull=0, opacity=0.1, color='cyan'))
        fig_bloch.add_trace(go.Scatter3d(x=[0, u], y=[0, 0], z=[0, v], mode='lines+markers', 
                                         line=dict(color='red', width=5), marker=dict(size=[0, 8], color='red')))
        fig_bloch.update_layout(scene=dict(xaxis=dict(range=[-1,1]), yaxis=dict(range=[-1,1]), zaxis=dict(range=[-1,1])))
        st.plotly_chart(fig_bloch, use_container_width=True)
        
    except Exception as e:
        st.error(f"Backend connection failed. Ensure `uvicorn main:app --reload` is running. Error: {e}")

with tab_train:
    st.header("Train the Hybrid Quantum Model")
    
    dataset_choice = st.selectbox("Select Benchmark Dataset to Train", ["Financial Fraud (Simulated)", "Concentric Circles (Hard)"])
    n_samples = st.slider("Number of Samples to Train On", 40, 200, 100)

    if st.button("Train Models"):
        with st.spinner('Building Quantum Circuits and training scikit-learn models...'):
            try:
                response = requests.post(f"{BACKEND_URL}/train-qsvm/", json={"dataset_name": dataset_choice, "n_samples": n_samples})
                data = response.json()
                
                col1, col2, col3 = st.columns(3)
                diff = data['quantum_accuracy'] - data['classical_accuracy']
                
                col1.metric("Classical Accuracy (Linear)", f"{data['classical_accuracy'] * 100:.1f}%")
                col2.metric("Quantum Accuracy", f"{data['quantum_accuracy'] * 100:.1f}%", delta=f"{diff * 100:.1f}%" if diff != 0 else None)
                col3.metric("Execution Time", f"{data['execution_time_seconds']} sec")
                
                chart_data = pd.DataFrame([data['classical_accuracy'], data['quantum_accuracy']],
                                          index=["Classical SVM", "Quantum SVM"], columns=["Accuracy"])
                st.bar_chart(chart_data)
            except Exception as e:
                st.error(f"Training failed. {e}")