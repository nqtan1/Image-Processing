import cv2
import numpy as np
import matplotlib.pyplot as plt
from collections import deque, defaultdict

class Graph:
    def __init__(self, vertices):
        self.V = vertices
        self.graph = defaultdict(dict)

    def add_edge(self, u, v, capacity):
        self.graph[u][v] = capacity
        self.graph[v][u] = 0

    def bfs(self, source, sink, parent):
        visited = set()
        queue = deque([source])
        visited.add(source)

        while queue:
            u = queue.popleft()
            for v in self.graph[u]:
                if v not in visited and self.graph[u][v] > 0:
                    queue.append(v)
                    visited.add(v)
                    parent[v] = u
                    if v == sink:
                        return True
        return False

    def ford_fulkerson(self, source, sink):
        parent = {}
        max_flow = 0

        while self.bfs(source, sink, parent):
            path_flow = float("Inf")
            v = sink
            while v != source:
                u = parent[v]
                path_flow = min(path_flow, self.graph[u][v])
                v = u

            v = sink
            while v != source:
                u = parent[v]
                self.graph[u][v] -= path_flow
                self.graph[v][u] += path_flow
                v = u

            max_flow += path_flow

        return max_flow

def segment_image(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    h, w = img.shape
    num_pixels = h * w
    source = num_pixels
    sink = num_pixels + 1

    graph = Graph(num_pixels + 2)

    for y in range(h):
        for x in range(w):
            idx = y * w + x

            if img[y, x] < 100:
                graph.add_edge(source, idx, 10)
            elif img[y, x] > 150:
                graph.add_edge(idx, sink, 10)

            if x < w - 1:
                neighbor_idx = y * w + (x + 1)
                graph.add_edge(idx, neighbor_idx, 2)

            if y < h - 1:
                neighbor_idx = (y + 1) * w + x
                graph.add_edge(idx, neighbor_idx, 2)

    max_flow = graph.ford_fulkerson(source, sink)

    segmented = np.zeros_like(img)
    for y in range(h):
        for x in range(w):
            idx = y * w + x
            if sink in graph.graph[idx] and graph.graph[idx][sink] > 0:
                segmented[y, x] = 255

    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(img, cmap='gray')
    plt.title("Original Image")

    plt.subplot(1, 2, 2)
    plt.imshow(segmented, cmap='gray')
    plt.title("Segmented Image")

    plt.show()

segment_image("img/lego_gray.jpg")