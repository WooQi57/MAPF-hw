import heapq

class AStarPlanner:
    class Node:
        def __init__(self, x, y):
            self.x = x
            self.y = y
            self.g = float('inf')  # Initialize with infinity
            self.h = 0
            self.parent = None

        def __lt__(self, other):
            return (self.g + self.h) < (other.g + other.h)

    def __init__(self, width, height, obstacles):
        self.width = width
        self.height = height
        self.obstacles = obstacles
        self.open_set = []
        self.closed_set = set()

    def heuristic(self, current, goal):
        return abs(current.x - goal.x) + abs(current.y - goal.y)

    def within_bounds(self, x, y):
        return 0 <= x < self.width and 0 <= y < self.height

    def get_neighbors(self, node):
        neighbors = []
        for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
            nx, ny = node.x + dx, node.y + dy
            if self.within_bounds(nx, ny) and (nx, ny) not in self.obstacles:
                neighbors.append(self.Node(nx, ny))
        return neighbors

    def reconstruct_path(self, current):
        path = [(current.x, current.y)]
        while current.parent:
            current = current.parent
            path.append((current.x, current.y))
        return list(reversed(path))

    def astar(self, start_x, start_y, goal_x, goal_y):
        start_node = self.Node(start_x, start_y)
        goal_node = self.Node(goal_x, goal_y)
        start_node.g = 0
        start_node.h = self.heuristic(start_node, goal_node)
        heapq.heappush(self.open_set, start_node)

        while self.open_set:
            current = heapq.heappop(self.open_set)

            if current.x == goal_node.x and current.y == goal_node.y:
                return self.reconstruct_path(current)

            self.closed_set.add((current.x, current.y))

            for neighbor in self.get_neighbors(current):
                if (neighbor.x, neighbor.y) in self.closed_set:
                    continue

                tentative_g = current.g + 1  # Assuming uniform cost for each step

                if tentative_g < neighbor.g:
                    neighbor.parent = current
                    neighbor.g = tentative_g
                    neighbor.h = self.heuristic(neighbor, goal_node)
                    heapq.heappush(self.open_set, neighbor)

        return None  # No path found

if __name__ == "__main__":
    # obstacles = {(2, 2), (3, 3), (4, 4)}  # Example obstacle positions

    # # Create an instance of AStarPlanner
    # astar = AStarPlanner(obstacles)

    # # Call the A* algorithm
    # path = astar.astar(1, 1, 8, 8)
    # print("Path:", path)

    width = 10
    height = 10
    obstacles = {(2, 2), (3, 3), (4, 4)}  # Example obstacle positions

    astar = AStarPlanner(width, height, obstacles)

    # Call the A* algorithm
    path = astar.astar(1, 1, 8, 8)
    print("Path:", path)

