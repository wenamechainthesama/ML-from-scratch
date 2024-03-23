import numpy as np
import matplotlib.pyplot as plt


def euclidean_distance(pointA, pointB):
    return np.sqrt(np.sum(np.power(np.subtract(pointA, pointB), 2)))


class DBSCAN:
    def __init__(self, radius, num_samples_for_core):
        self.radius = radius
        self.num_samples_for_core = num_samples_for_core

    def cluster(self, X):
        # Find core points
        core_points_idxs = []
        for sample_idx, sample in enumerate(X):
            samples_around = 0
            for other_sample in X:
                distance = euclidean_distance(sample, other_sample)
                if distance <= self.radius:
                    samples_around += 1

            # Substruct 1 because we don't need to count current point itself
            samples_around -= 1
            if self.num_samples_for_core <= samples_around:
                core_points_idxs.append(sample_idx)

        # Form clusters
        clusters = []
        while len(core_points_idxs) != 0:
            cluster_points_idxs = self._form_cluster(X, core_points_idxs)
            cluster = [list(point) for point in X[cluster_points_idxs]]
            clusters.append(cluster)

        return clusters

    def _form_cluster(self, X, core_points_idxs: list):
        # Pick random core point
        cluster_points_idxs = []
        cluster_initial_point = np.random.choice(core_points_idxs, replace=False)
        cluster_points_idxs.append(cluster_initial_point)
        core_points_idxs.remove(cluster_initial_point)

        # Infect other near core points
        all_core_points_added = False
        while not all_core_points_added:
            all_core_points_added = not self._infect_core_points(
                X, core_points_idxs, cluster_points_idxs
            )

        # Infect non-core points
        for cluster_point_idx in cluster_points_idxs:
            for other_point_idx, other_point in enumerate(X):
                if other_point_idx in cluster_points_idxs:
                    continue
                distance = euclidean_distance(X[cluster_point_idx, :], other_point)
                if distance <= self.radius:
                    cluster_points_idxs.append(other_point_idx)

        return cluster_points_idxs

    def _infect_core_points(self, X, core_points_idxs: list, cluster_points_idxs: list):
        has_points_added = False
        for chosen_point in cluster_points_idxs:
            for other_core_point_idx in core_points_idxs:
                if other_core_point_idx in cluster_points_idxs:
                    continue
                distance = euclidean_distance(
                    X[chosen_point, :], X[other_core_point_idx, :]
                )
                if distance <= self.radius:
                    has_points_added = True
                    cluster_points_idxs.append(other_core_point_idx)
                    core_points_idxs.remove(other_core_point_idx)

        return has_points_added


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

    model = DBSCAN(7, 5)
    clusters = model.cluster(np.array(all_points))

    # Plot outliers
    s = [x for xs in clusters for x in xs]
    for point in all_points:
        if list(point) not in s:
            plt.scatter(point[0], point[1], c="black")

    color1 = "red"
    color2 = "green"
    color3 = "blue"
    color4 = "orange"

    # Plot clusters
    for cluster, color in list(zip(clusters, [color1, color2, color3, color4])):
        for point in cluster:
            plt.scatter(point[0], point[1], color=color)

    plt.show()
