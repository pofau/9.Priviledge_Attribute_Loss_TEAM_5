import numpy as np
import matplotlib.pyplot as plt

def plot_element(images, batch_heatmaps, attribution_maps, gradients, i):
    plt.figure(figsize=(12, 8))  # Enlarge the figure to accommodate all visualizations
    # Display the original image
    plt.subplot(2, 4, 1)
    image_to_show = images[i].permute(1, 2, 0).cpu().detach().numpy()
    plt.imshow(image_to_show)
    plt.title('Original Image')

    # Display the heatmap
    plt.subplot(2, 4, 2)
    heatmap_to_show = batch_heatmaps[i].permute(1, 2, 0).cpu().detach().numpy()
    plt.imshow(heatmap_to_show)
    plt.title('Heatmap')

    # Display each channel of the attribution map
    plt.subplot(2, 4, 3)
    attribution_to_show = attribution_maps[i].detach().permute(1, 2, 0).cpu().numpy()
    attribution_norm = (attribution_to_show - attribution_to_show.min()) / (attribution_to_show.max() - attribution_to_show.min())
    attribution_mean = np.mean(attribution_norm, axis=2)
    # Apply a 'jet' colormap to get a colored attribution map
    cmap = plt.get_cmap('bwr')
    attribution_colored = cmap(attribution_mean)

    # Remove the alpha channel returned by the colormap
    attribution_colored = attribution_colored[..., :3]
    overlayed_image = (image_to_show) * 0.2 + attribution_colored * 0.9  # Adjust the transparency here
    plt.imshow(overlayed_image)
    plt.title('Attribution Overlay on Original Image')

    # Display the output gradient over the original image
    plt.subplot(1, 4, 4)
    gradients_to_show = gradients[i].detach().permute(1, 2, 0).cpu().numpy()
    gradients_to_show = np.abs(gradients_to_show)
    gradients_to_show /= np.max(gradients_to_show)
                
    # Overlay the gradient on the original image
    overlayed_image = (gradients_to_show * 1.5 + image_to_show * 0.2)
    plt.imshow(overlayed_image)
    plt.title('Gradient Overlay on Original Image')
    
    plt.show()

if __name__ == "__main__":  # False during an import   
    pass
