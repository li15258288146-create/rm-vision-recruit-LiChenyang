"""八邻域手写 Dijkstra/A*；不依赖现成路径规划库。"""
import heapq
import math
import time
import cv2
import numpy as np


class GridMap:
    def __init__(self, image, resolution, origin, free_thresh, radius=.30, negate=0):
        self.image, self.resolution = image, float(resolution)
        self.origin = np.array(origin[:2], float)
        yaw = origin[2]
        self.rotation = np.array([[math.cos(yaw), -math.sin(yaw)], [math.sin(yaw), math.cos(yaw)]])
        self.h, self.w = image.shape
        probability = image / 255. if negate else (255. - image) / 255.
        self.raw_free = probability < free_thresh
        # 与题目检查器一致：距离按占据栅格中心计算；外侧补障碍边界。
        padded = np.pad(self.raw_free.astype(np.uint8), 1)
        clearance = cv2.distanceTransform(padded, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)[1:-1, 1:-1] * resolution
        self.free = self.raw_free & (clearance >= radius + 1e-7)

    def world_to_cell(self, xy):
        local = self.rotation.T @ (np.array(xy) - self.origin)
        col, bottom_row = np.floor(local / self.resolution).astype(int)
        return self.h - 1 - int(bottom_row), int(col)

    def cell_to_world(self, cell):
        row, col = cell
        local = np.array([col + .5, self.h - row - .5]) * self.resolution
        return tuple(self.rotation @ local + self.origin)

    def reason(self, cell):
        r, c = cell
        if not (0 <= r < self.h and 0 <= c < self.w):
            return 'outside_map'
        if not self.raw_free[r, c]:
            return 'occupied_or_unknown'
        if not self.free[r, c]:
            return 'insufficient_robot_clearance'
        return None

    def verify(self, points):
        """对所有线段以不大于 1/5 格间距采样，检查安全栅格。"""
        for a, b in zip(points, points[1:]):
            n = max(1, math.ceil(math.dist(a, b) / (self.resolution / 5)))
            for t in np.linspace(0, 1, n + 1):
                if self.reason(self.world_to_cell(np.array(a) * (1-t) + np.array(b) * t)):
                    return False
        return True


def search(free, start, goal, algorithm='astar', heuristic='octile', weight=1., traversal_cost=None):
    # Symmetric edge weights >= geometric distance keep octile admissible.
    if traversal_cost is not None:
        if traversal_cost.shape != free.shape or not np.all(np.isfinite(traversal_cost[free])) or np.any(traversal_cost[free] < 1):
            raise ValueError('traversal_cost must be finite, shape matched and >= 1 on free cells')
    began = time.perf_counter()
    def h(cell):
        if algorithm == 'dijkstra':
            return 0.
        dy, dx = abs(cell[0]-goal[0]), abs(cell[1]-goal[1])
        return weight * (dy + dx if heuristic == 'manhattan' else max(dx,dy)+(math.sqrt(2)-1)*min(dx,dy))
    queue = [(h(start), 0., start)]
    best, parent, expanded = {start: 0.}, {}, []
    moves = [(dr,dc,math.hypot(dr,dc)) for dr in (-1,0,1) for dc in (-1,0,1) if dr or dc]
    found = None
    while queue:
        _, cost, cell = heapq.heappop(queue)
        if cost > best[cell] + 1e-10:
            continue
        expanded.append(cell)
        if cell == goal:
            found = [cell]
            while cell != start:
                cell = parent[cell]
                found.append(cell)
            found.reverse()
            break
        r,c = cell
        for dr,dc,length in moves:
            nr,nc = r+dr,c+dc
            if not (0 <= nr < free.shape[0] and 0 <= nc < free.shape[1]) or not free[nr,nc]:
                continue
            if dr and dc and (not free[r+dr,c] or not free[r,c+dc]):
                continue
            nxt = (nr,nc)
            factor = 1. if traversal_cost is None else (traversal_cost[r,c] + traversal_cost[nr,nc]) / 2
            candidate = cost + length * factor
            if candidate + 1e-10 < best.get(nxt, math.inf):
                best[nxt],parent[nxt] = candidate,cell
                heapq.heappush(queue,(candidate+h(nxt),candidate,nxt))
    return dict(path=found, cost_cells=best.get(goal) if found else None,
                expanded=expanded, seconds=time.perf_counter()-began)
