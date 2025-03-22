import cv2
import numpy as np
from graph import GraphFlow

def main():
    #img = cv2.imread("img/lego_gray.jpg", cv2.IMREAD_GRAYSCALE)
    img = cv2.imread("img/brain_ct.jpg", cv2.IMREAD_GRAYSCALE)
    if img is None:
        print("Failed to open image")
        return

    print("Image shape:", img.shape)

    graph = GraphFlow()
    rows, cols = img.shape
    num_nodes = rows * cols
    graph.set_nb_nodes(num_nodes)

    seuil = 0.38
    alpha = 2.5

    print("Setting up graph...")
    for r in range(rows):
        for c in range(cols):
            index = r * cols + c
            value = img[r, c] / 255.0

            graph.connect_source_to_node(index, value)
            graph.connect_node_to_sink(index, 1 - value)

            if r < rows - 1:
                neighbor_index = (r + 1) * cols + c
                neighbor_value = img[r + 1, c] / 255.0
                if (value >= seuil) != (neighbor_value >= seuil):
                    graph.connect_nodes(index, neighbor_index, alpha)

            if r > 0:
                neighbor_index = (r - 1) * cols + c
                neighbor_value = img[r - 1, c] / 255.0
                if (value >= seuil) != (neighbor_value >= seuil):
                    graph.connect_nodes(index, neighbor_index, alpha)

            if c < cols - 1:
                neighbor_index = r * cols + (c + 1)
                neighbor_value = img[r, c + 1] / 255.0
                if (value >= seuil) != (neighbor_value >= seuil):
                    graph.connect_nodes(index, neighbor_index, alpha)

            if c > 0:
                neighbor_index = r * cols + (c - 1)
                neighbor_value = img[r, c - 1] / 255.0
                if (value >= seuil) != (neighbor_value >= seuil):
                    graph.connect_nodes(index, neighbor_index, alpha)
    
    #graph.draw()     
    print("Running Ford-Fulkerson...")
    graph.ford_fulkerson()
    print("Finished Ford-Fulkerson")
    #graph.draw()

    vect_s, _ = graph.cut_from_source()

    seg = np.zeros_like(img, dtype=np.uint8)
    for index in vect_s:
        r = index // cols
        c = index % cols
        seg[r, c] = 255 

    cv2.imshow("Original Image", img)
    cv2.imshow("Segmented Image", seg)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()