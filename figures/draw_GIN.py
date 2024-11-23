import matplotlib.pyplot as plt
import networkx as nx

def draw_gin_model():
    # Create a directed graph
    G = nx.DiGraph()

    # Add nodes for each layer in the GIN model
    G.add_node("Input Features", pos=(0, 6))
    G.add_node("GINConv1", pos=(0, 5))
    G.add_node("ReLU1", pos=(0, 4))
    G.add_node("GINConv2", pos=(0, 3))
    G.add_node("ReLU2", pos=(0, 2))
    G.add_node("Global Mean Pool", pos=(0, 1))
    G.add_node("Linear (Mem)", pos=(-1, 0))
    G.add_node("Linear (Time)", pos=(1, 0))
    G.add_node("Outputs: Mem, Time", pos=(0, -1))

    # Add edges to represent data flow
    G.add_edges_from([
        ("Input Features", "GINConv1"),
        ("GINConv1", "ReLU1"),
        ("ReLU1", "GINConv2"),
        ("GINConv2", "ReLU2"),
        ("ReLU2", "Global Mean Pool"),
        ("Global Mean Pool", "Linear (Mem)"),
        ("Global Mean Pool", "Linear (Time)"),
        ("Linear (Mem)", "Outputs: Mem, Time"),
        ("Linear (Time)", "Outputs: Mem, Time")
    ])

    # Get node positions
    pos = nx.get_node_attributes(G, 'pos')

    # Draw the graph
    plt.figure(figsize=(8, 8))
    nx.draw(G, pos, with_labels=True, node_size=3000, node_color="skyblue", font_size=10, font_weight="bold", arrowsize=20)
    plt.title("GIN Model Structure", fontsize=14)
    plt.savefig("GIN_Model_Structure.png", dpi=300)

# Call the function to draw the GIN model
draw_gin_model()
