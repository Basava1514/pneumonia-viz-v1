# docs/generate_diagram.py
from graphviz import Digraph
dot = Digraph("Pipeline", format="png")
dot.attr(rankdir="TB")
dot.node("X","Chest X-ray", shape="box")
dot.node("U","U-Net\nLung Segmentation", shape="box")
dot.node("M","Masked X-ray", shape="box")
dot.node("V","ViT Classifier", shape="box")
dot.node("E","Explainability\n(Grad-CAM, ViT-CX, LIME, SHAP)", shape="box")
dot.node("S","Streamlit Dashboard", shape="box")
dot.node("F","Feedback Dataset\n(jsonl) & Auto-Retrain", shape="box")
dot.edges(["XU","UM","MV","VE","VS","SF"])
dot.render("docs/architecture_diagram", cleanup=True)
print("Saved docs/architecture_diagram.png")
