import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# Paths to your individual images
images = [
    "figures/memory_usage_98973d2500944ac1-bf4d0d2009caa722.png",
    "figures/memory_usage_bb3cc3628e564ab9-9c194fd98eb655a0.png",
    "figures/memory_usage_bb8adb9737f749bb-b9b4812bebed62f5.png",
    "figures/memory_usage_bb8cba7dc2ec43ce-8197944044d1f017.png",
    "figures/memory_usage_bb9d34e751464cbb-8c3e442b8676ab51.png",
    "figures/memory_usage_bc743b6247464220-9aa0caff344bb4dd.png",
]

# Labels for subfigures
subfigure_labels = ['(a)', '(b)', '(c)', '(d)', '(e)', '(f)']

# Create a 3x2 grid figure
fig, axes = plt.subplots(3, 2, figsize=(15, 15.5))

# Add each image to the grid
for i, img_path in enumerate(images):
    row = i // 2
    col = i % 2
    img = mpimg.imread(img_path)
    axes[row, col].imshow(img)
    axes[row, col].axis("off")  # Hide axis

    # Set the label below the subplot
    axes[row, col].set_title(subfigure_labels[i], fontsize=14, y=-0.15)

# Adjust layout
plt.tight_layout()
plt.subplots_adjust(hspace=0.05)  # Increase vertical space between rows

# Save the combined figure
plt.savefig("memory_usage_combined_below_labels_corrected.png")
# plt.show()


