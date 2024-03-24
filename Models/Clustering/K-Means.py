import numpy as np
import matplotlib.pyplot as plt


def euclidean_distance(pointA, pointB):
    return np.sqrt(np.sum(np.power(np.subtract(pointA, pointB), 2)))


# https://www.youtube.com/watch?v=4b5d3muPQmA
class KMeans:
    def __init__(self, num_clusters) -> None:
        self.num_clusters = num_clusters

    def cluster(self, X):
        # Pick random points as initial centroids
        centroids = []
        for _ in range(self.num_clusters):
            random_point = X[np.random.choice(len(X), replace=False)]
            centroids.append(random_point)

        clusters = []
        previous_clusters = None
        while clusters != previous_clusters:
            # For each point find the closest centroid and add it to corresponding cluster
            clusters = [[] for _ in range(self.num_clusters)]
            for point in X:
                closest_centroid_idx = -1
                smallest_distance = float("inf")
                for centroid_idx, centroid in enumerate(centroids):
                    distance = euclidean_distance(point, centroid)
                    if distance < smallest_distance:
                        closest_centroid_idx = centroid_idx
                        smallest_distance = distance
                clusters[closest_centroid_idx].append(list(point))

            # Update centroids using mean of cluster
            for cluster_idx, cluster in enumerate(clusters):
                cluster_mean = np.mean(cluster, axis=0)
                centroids[cluster_idx] = cluster_mean

            previous_clusters = clusters

        return clusters


if __name__ == "__main__":
    # Illustrative example
    center1 = (50, 60)
    center2 = (80, 50)
    center3 = (55, -40)
    center4 = (90, -40)
    distance = 18

    x1 = np.random.uniform(center1[0], center1[0] + distance, size=(100,))
    y1 = np.random.normal(center1[1], distance, size=(100,))

    x2 = np.random.uniform(center2[0], center2[0] + distance, size=(100,))
    y2 = np.random.normal(center2[1], distance, size=(100,))

    x3 = np.random.uniform(center3[0], center3[0] + distance, size=(100,))
    y3 = np.random.normal(center3[1], distance, size=(100,))

    x4 = np.random.uniform(center4[0], center4[0] + distance, size=(100,))
    y4 = np.random.normal(center4[1], distance, size=(100,))

    cluster1 = list(zip(x1, y1))
    cluster2 = list(zip(x2, y2))
    cluster3 = list(zip(x3, y3))
    cluster4 = list(zip(x4, y4))

    all_points = []
    all_points.extend(cluster1)
    all_points.extend(cluster2)
    all_points.extend(cluster3)
    all_points.extend(cluster4)

    model = KMeans(4)
    clusters = model.cluster(np.array(all_points))
    for cluster in clusters:
        color = np.random.rand(3)
        for point in cluster:
            plt.scatter(point[0], point[1], color=color)

    plt.show()
