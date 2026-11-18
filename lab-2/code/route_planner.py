from math import inf


def shortest_route(matrix: list[list[int]], start: int, finish: int) -> tuple[int, list[int]]:
    """Return the shortest distance and route for a graph with non-negative weights."""
    vertex_count = len(matrix)
    distance = [inf] * vertex_count
    previous = [-1] * vertex_count
    visited = [False] * vertex_count
    distance[start] = 0

    for _ in range(vertex_count):
        current = -1
        for vertex in range(vertex_count):
            if not visited[vertex] and (current == -1 or distance[vertex] < distance[current]):
                current = vertex

        if current == -1 or distance[current] == inf:
            break
        if current == finish:
            break

        visited[current] = True
        for neighbor, weight in enumerate(matrix[current]):
            if weight > 0 and not visited[neighbor]:
                candidate = distance[current] + weight
                if candidate < distance[neighbor]:
                    distance[neighbor] = candidate
                    previous[neighbor] = current

    if distance[finish] == inf:
        return -1, []

    route = []
    current = finish
    while current != -1:
        route.append(current)
        current = previous[current]
    route.reverse()
    return int(distance[finish]), route


def main() -> None:
    vertex_count = int(input("Number of vertices: "))
    matrix = [list(map(int, input().split())) for _ in range(vertex_count)]
    start, finish = map(int, input("Start and finish: ").split())

    if not (0 <= start < vertex_count and 0 <= finish < vertex_count):
        print("Invalid vertices")
        return

    distance, route = shortest_route(matrix, start, finish)
    if distance == -1:
        print("Route not found")
    else:
        print("Distance:", distance)
        print("Route:", " -> ".join(map(str, route)))


if __name__ == "__main__":
    main()
