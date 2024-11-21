from graphviz import Digraph

# Create a new directed graph
dot = Digraph(comment="Memory-Based Scheduling Framework", format="png")

# Define graph attributes
dot.attr(rankdir="TB", size="8,10")

# Add nodes for each core component
dot.node("A", "Client Interface", shape="box", style="rounded,filled", fillcolor="lightblue")
dot.node("B", "Query Analyzer", shape="box", style="rounded,filled", fillcolor="green")
dot.node("C", "Scheduler", shape="box", style="rounded,filled", fillcolor="lightcoral")
dot.node("D", "Execution Engine", shape="box", style="rounded,filled", fillcolor="gold")
dot.node("E", "Monitoring Module", shape="box", style="rounded,filled", fillcolor="plum")

# Add edges to represent the workflow
dot.edge("A", "B", label="Submit Query")
dot.edge("B", "C", label="Memory Prediction")
dot.edge("C", "D", label="Query Scheduling")
dot.edge("D", "E", label="Track Execution")
dot.edge("E", "C", label="Feedback for Dynamic Adjustment")

# Add labels to represent the middleware layer and server
dot.attr(label="Memory-Based Scheduling Framework\nMiddleware Layer Between Client and Database Server", fontsize="14", labelloc="t")

# Render the diagram
file_path = "Memory_Based_Scheduling_Framework"
dot.render(file_path, format="png", cleanup=True)

print(f"Figure saved to: {file_path}.png")
