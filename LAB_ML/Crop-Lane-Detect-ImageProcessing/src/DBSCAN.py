# DBSCAN.py

import numpy as np

class DBSCAN:
    def __init__(self, eps, minPts):
        self.dbscan_dict = {}
        self.points = []
        self.minPts = minPts
        self.eps = eps

    def update(self, line):
        line = line.squeeze()
        x1 = line[0]
        y1 = line[1]
        x2 = line[2]
        y2 = line[3]
        mid_x = (x1 + x2) / 2
        mid_y = (y1 + y2) / 2
        self.dbscan_dict[(mid_x, mid_y)] = line
        self.points.append(np.array([mid_x, mid_y]))

    def scan(self):
        labels = [0]*len(self.points)
        C = 0

        for P in range(0, len(self.points)):
            if not (labels[P] == 0):
                continue
            NeighborPts = self.region_query(P)
            if len(NeighborPts) < self.minPts:
                labels[P] = -1
            else:
                C += 1
                self.grow_cluster(labels, P, NeighborPts, C)
        return labels

    def grow_cluster(self, labels, P, NeighborPts, C):
        labels[P] = C
        i = 0
        while i < len(NeighborPts):
            Pn = NeighborPts[i]
            if labels[Pn] == -1:
                labels[Pn] = C
            elif labels[Pn] == 0:
                labels[Pn] = C
                PnNeighborPts = self.region_query(Pn)
                if len(PnNeighborPts) >= self.minPts:
                    NeighborPts = NeighborPts + PnNeighborPts
            i += 1

    def region_query(self, P):
        neighbors = []
        for Pn in range(0, len(self.points)):
            if np.linalg.norm(self.points[P] - self.points[Pn]) < self.eps:
                neighbors.append(Pn)
        return neighbors

    def return_max(self, labels):
        values, counts = np.unique(labels, return_counts=True)
        if len(values) == 0 or (values[0] == -1 and len(values) == 1):
            return []
        idx = np.argmax(counts)
        if values[idx] == -1:
            if len(counts) > 1:
                idx = np.argmax(counts[1:]) + 1
            else:
                return []
        lines = []
        for i in range(len(labels)):
            if labels[i] == values[idx]:
                key = (self.points[i][0], self.points[i][1])
                lines.append(self.dbscan_dict[key])
        return lines
