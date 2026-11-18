#include <algorithm>
#include <iostream>
#include <limits>
#include <utility>
#include <vector>

using namespace std;

pair<int, vector<int>> shortestRoute(const vector<vector<int>>& matrix, int start, int finish) {
    const int vertexCount = static_cast<int>(matrix.size());
    const int INF = numeric_limits<int>::max();
    vector<int> distance(vertexCount, INF);
    vector<int> previous(vertexCount, -1);
    vector<bool> visited(vertexCount, false);
    distance[start] = 0;

    for (int step = 0; step < vertexCount; ++step) {
        int current = -1;
        for (int vertex = 0; vertex < vertexCount; ++vertex) {
            if (!visited[vertex] &&
                (current == -1 || distance[vertex] < distance[current])) {
                current = vertex;
            }
        }

        if (current == -1 || distance[current] == INF) break;
        if (current == finish) break;

        visited[current] = true;
        for (int neighbor = 0; neighbor < vertexCount; ++neighbor) {
            const int weight = matrix[current][neighbor];
            if (weight > 0 && !visited[neighbor] &&
                distance[current] <= INF - weight) {
                const int candidate = distance[current] + weight;
                if (candidate < distance[neighbor]) {
                    distance[neighbor] = candidate;
                    previous[neighbor] = current;
                }
            }
        }
    }

    if (distance[finish] == INF) return {-1, {}};

    vector<int> route;
    for (int current = finish; current != -1; current = previous[current]) {
        route.push_back(current);
    }
    reverse(route.begin(), route.end());
    return {distance[finish], route};
}

int main() {
    int vertexCount;
    cout << "Number of vertices: ";
    cin >> vertexCount;

    vector<vector<int>> matrix(vertexCount, vector<int>(vertexCount));
    for (auto& row : matrix) {
        for (int& weight : row) cin >> weight;
    }

    int start, finish;
    cout << "Start and finish: ";
    cin >> start >> finish;

    if (start < 0 || start >= vertexCount || finish < 0 || finish >= vertexCount) {
        cout << "Invalid vertices\n";
        return 0;
    }

    const auto [distance, route] = shortestRoute(matrix, start, finish);
    if (distance == -1) {
        cout << "Route not found\n";
    } else {
        cout << "Distance: " << distance << '\n';
        cout << "Route: ";
        for (size_t index = 0; index < route.size(); ++index) {
            if (index > 0) cout << " -> ";
            cout << route[index];
        }
        cout << '\n';
    }
    return 0;
}
