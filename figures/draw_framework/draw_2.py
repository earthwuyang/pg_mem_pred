import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# Initialize the figure and axes
fig, ax = plt.subplots(figsize=(12, 8))

# Define colors and positions
box_color = "#AEC6CF"
arrow_color = "#555555"
text_color = "#000000"

# Function to draw a rectangle with text
def draw_box(x, y, width, height, text, fontsize=12, color=box_color):
    rect = mpatches.FancyBboxPatch((x, y), width, height, boxstyle="round,pad=0.2", 
                                   edgecolor='black', facecolor=color)
    ax.add_patch(rect)
    ax.text(x + width / 2, y + height / 2, text, color=text_color, 
            fontsize=fontsize, ha='center', va='center', wrap=True)

# Function to draw an arrow
def draw_arrow(x1, y1, x2, y2, text=None, fontsize=10):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(facecolor=arrow_color, edgecolor=arrow_color, arrowstyle="->"))
    if text:
        ax.text((x1 + x2) / 2, (y1 + y2) / 2, text, fontsize=fontsize, ha='center', va='center')

# Memory Predictor Block
draw_box(0.2, 7, 1.5, 0.8, "Memory Predictor", fontsize=14)

# Query Plan Block
draw_box(-1, 5, 1.5, 0.8, "Query Plan", fontsize=12)
draw_arrow(-0.25, 5.8, 0.2, 7.4, text="Predicted Peak Memory", fontsize=10)

# Nested Loop Plan
draw_box(-1.5, 4, 1, 0.6, "Nested Loop", fontsize=10)
draw_arrow(-0.25, 5, -1, 4.6)

# Plan Operators
draw_box(-3, 3, 1, 0.6, "Hash Join", fontsize=10)
draw_box(-1.5, 3, 1, 0.6, "Index Scan", fontsize=10)
draw_arrow(-2, 3.6, -2.5, 3)
draw_arrow(-2, 3.6, -1.5, 3.3)

# SQL Query
draw_box(-4, 2, 1, 0.6, "SQL Query", fontsize=10)
draw_arrow(-3.5, 2.6, -3, 3)

# Scheduler
draw_box(3, 6, 1.5, 0.8, "Scheduler", fontsize=14)
draw_arrow(1.7, 7.4, 3, 6.8, text="Assign Priority (FFD)", fontsize=10)
draw_arrow(3.75, 6, 4.5, 4.5, text="Execute", fontsize=10)

# Monitor
draw_box(4.5, 3.5, 1.5, 0.8, "Monitor", fontsize=12)
draw_arrow(5.25, 3.5, 5.25, 2.5, text="Available Memory", fontsize=10)

# Database
draw_box(4.5, 1.5, 1.5, 0.8, "Database", fontsize=12)
draw_arrow(5.25, 2.3, 5.25, 1.8, text="Execute Query", fontsize=10)

# Final Layout Adjustments
ax.set_xlim(-5, 7)
ax.set_ylim(0, 9)
ax.axis('off')  # Turn off axes

# Save and show the plot
plt.tight_layout()
plt.savefig("memory_prediction_scheduling.png", dpi=300)
# plt.show()
