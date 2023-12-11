def plot_element(images, batch_heatmaps, attribution_maps, gradients, i):
    plt.figure(figsize=(12, 8))  # Agrandir la figure pour accueillir toutes les visualisations
    # Afficher l'image original
    plt.subplot(2, 4, 1)
    image_to_show = images[i].permute(1, 2, 0).cpu().detach().numpy()
    plt.imshow(image_to_show)
    plt.title('Original Image')

    # Afficher la heatmap
    plt.subplot(2, 4, 2)
    heatmap_to_show = batch_heatmaps[i].permute(1, 2, 0).cpu().detach().numpy()
    plt.imshow(heatmap_to_show)
    plt.title('Heatmap')

    # Afficher chaque canal de la carte d'attribution
    plt.subplot(2, 4, 3)
    attribution_to_show = attribution_maps[i].detach().permute(1, 2, 0).cpu().numpy()
    attribution_norm = (attribution_to_show - attribution_to_show.min()) / (attribution_to_show.max() - attribution_to_show.min())
    attribution_mean = np.mean(attribution_norm, axis=2)
    # Appliquer une colormap 'jet' pour obtenir une carte d'attribution colorée
    cmap = plt.get_cmap('bwr')
    attribution_colored = cmap(attribution_mean)

    # Supprimer le canal alpha retourné par la colormap
    attribution_colored = attribution_colored[..., :3]
    overlayed_image = (image_to_show) * 0.2 + attribution_colored * 0.9  # Ajustez la transparence ici
    plt.imshow(overlayed_image)
    plt.title('Attribution Overlay on Original Image')

    # Afficher le gradient de sortie sur l'image originale
    plt.subplot(1, 4, 4)
    gradients_to_show = gradients[i].detach().permute(1, 2, 0).cpu().numpy()
    gradients_to_show = np.abs(gradients_to_show)
    gradients_to_show /= np.max(gradients_to_show)
                
    # Superposer le gradient sur l'image originale
    overlayed_image = (gradients_to_show * 1.5 + image_to_show * 0.2)
    plt.imshow(overlayed_image)
    plt.title('Gradient Overlay on Original Image')
    
    plt.show()