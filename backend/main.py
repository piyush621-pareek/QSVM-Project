from fastapi import FastAPI
from pydantic import BaseModel
from sqlalchemy import create_engine, Column, Integer, String, Float
from sqlalchemy.orm import declarative_base, sessionmaker
import time
import numpy as np
import math

# --- ML & Quantum Imports ---
from sklearn.datasets import make_classification, make_circles
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score
from qiskit.circuit.library import ZZFeatureMap
from qiskit.primitives import StatevectorSampler
from qiskit_machine_learning.kernels import FidelityQuantumKernel

# --- Database Setup ---
SQLALCHEMY_DATABASE_URL = "sqlite:///./qsvm_experiments.db"
engine = create_engine(SQLALCHEMY_DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

class Experiment(Base):
    __tablename__ = "experiments"
    id = Column(Integer, primary_key=True, index=True)
    dataset_name = Column(String, index=True)
    classical_accuracy = Column(Float)
    quantum_accuracy = Column(Float)
    execution_time = Column(Float)

Base.metadata.create_all(bind=engine)

# ==========================================
# THIS CREATES THE 'app' THAT WAS MISSING!
app = FastAPI(title="QSVM Backend API")
# ==========================================

class TrainRequest(BaseModel):
    dataset_name: str
    n_samples: int

@app.get("/visualize-data/")
def visualize_data(dataset: str = "Financial Fraud"):
    if dataset == "Concentric Circles (Hard)":
        X, y = make_circles(n_samples=500, noise=0.1, factor=0.3, random_state=42)
    else:
        X, y = make_classification(n_samples=500, n_features=2, n_informative=2, 
                                   n_redundant=0, n_clusters_per_class=2, 
                                   class_sep=0.8, random_state=42)
        
    scaler = MinMaxScaler(feature_range=(0, math.pi))
    X_scaled = scaler.fit_transform(X)
    
    sample_val = float(X_scaled[0][0])
    bloch_x = math.sin(sample_val)
    bloch_z = math.cos(sample_val)

    return {
        "x_coords": X[:, 0].tolist(), "y_coords": X[:, 1].tolist(), "labels": y.tolist(),
        "bloch_vector": [bloch_x, 0.0, bloch_z], "sample_value": sample_val
    }

@app.post("/train-qsvm/")
def train_models(request: TrainRequest):
    start_time = time.time()
    
    if request.dataset_name == "Concentric Circles (Hard)":
        X, y = make_circles(n_samples=request.n_samples, noise=0.1, factor=0.3, random_state=42)
    else:
        X, y = make_classification(n_samples=request.n_samples, n_features=2, n_informative=2, 
                                   n_redundant=0, n_clusters_per_class=2, class_sep=0.8, random_state=42)

    scaler = MinMaxScaler(feature_range=(0, 1))
    X_scaled = scaler.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.25, random_state=42)

    svm_classic = SVC(kernel='linear')
    svm_classic.fit(X_train, y_train)
    acc_classic = accuracy_score(y_test, svm_classic.predict(X_test))

    feature_map = ZZFeatureMap(feature_dimension=X_train.shape[1], reps=2, entanglement='linear')
    qkernel = FidelityQuantumKernel(feature_map=feature_map)
    svm_quantum = SVC(kernel=qkernel.evaluate)
    svm_quantum.fit(X_train, y_train)
    acc_quantum = accuracy_score(y_test, svm_quantum.predict(X_test))
    
    exec_time = round(time.time() - start_time, 2)

    db = SessionLocal()
    new_exp = Experiment(dataset_name=request.dataset_name, classical_accuracy=float(acc_classic),
                         quantum_accuracy=float(acc_quantum), execution_time=exec_time)
    db.add(new_exp)
    db.commit()
    db.refresh(new_exp)
    db.close()

    return {
        "dataset": request.dataset_name, "classical_accuracy": float(acc_classic),
        "quantum_accuracy": float(acc_quantum), "execution_time_seconds": exec_time,
        "experiment_id": new_exp.id
    }