import numpy as np
import matplotlib.pyplot as plt


def euclidean_distance(pointA, pointB):
    return np.sqrt(np.sum(np.power(np.subtract(pointA, pointB), 2)))


"""
Concept:
https://youtu.be/JcniBqle4kw?feature=shared
https://youtu.be/7xHsRkOdVwo?feature=shared
"""


class HierarchicalClustering:
    def __init__(self, num_clusters) -> None:
        self.num_clusters = num_clusters

    def cluster(self, X):
        clusters = [[point] for point in X]
        while len(clusters) != min(self.num_clusters, len(X)):

            smallest_distance = float("inf")
            silimar_clusters_idxs = None

            # Find two the most silimar clusters
            for cluster1_idx, cluster1 in enumerate(clusters):
                for cluster2_idx, cluster2 in enumerate(clusters):
                    if cluster1 == cluster2:
                        continue

                    # Check for similarity
                    distance = euclidean_distance(
                        np.mean(cluster1, axis=0), np.mean(cluster2, axis=0)
                    )

                    if distance < smallest_distance:
                        smallest_distance = distance
                        silimar_clusters_idxs = [cluster1_idx, cluster2_idx]

            # Merge the most similar clusters
            updated_cluster = (
                clusters[silimar_clusters_idxs[0]] + clusters[silimar_clusters_idxs[1]]
            )

            del clusters[max(silimar_clusters_idxs)]
            del clusters[min(silimar_clusters_idxs)]

            clusters.append(updated_cluster)

        return clusters


if __name__ == "__main__":
    # Illustrative example
    center1 = (50, 60)
    center2 = (80, 50)
    center3 = (55, -40)
    center4 = (90, -40)
    distance = 10

    x1 = np.random.uniform(center1[0], center1[0] + distance, size=(50,))
    y1 = np.random.normal(center1[1], distance, size=(100,))

    x2 = np.random.uniform(center2[0], center2[0] + distance, size=(50,))
    y2 = np.random.normal(center2[1], distance, size=(100,))

    x3 = np.random.uniform(center3[0], center3[0] + distance, size=(50,))
    y3 = np.random.normal(center3[1], distance, size=(100,))

    x4 = np.random.uniform(center4[0], center4[0] + distance, size=(50,))
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

    model = HierarchicalClustering(4)
    clusters = model.cluster(all_points)

    for cluster in clusters:
        color = np.random.rand(3)
        for point in cluster:
            plt.scatter(point[0], point[1], color=color)

    plt.show()
