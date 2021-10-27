
# Topologically sorts the graph
def topologicalSort(vertices, edges):
    # Mark all the vertices as not visited
    visited = [False]*len(vertices)
    ret = []

    # Define helper functions
    def getVertexNumber(v, vertices):
        for (i, vert) in enumerate(vertices):
            if v == vert:
                return i

        raise Exception(f"Vertex {v} not in graph")

    def topoSortHelper(i, visited, ret, vertices, edges):
        # We are currently visiting vertex i
        visited[i] = True

        # We now visit all adjacent Vertices that have yet to be visited
        try:
            adjacent = edges[vertices[i]]
        except:
            adjacent = []

        for v in adjacent:
            if not visited[getVertexNumber(v, vertices)]:
                topoSortHelper(getVertexNumber(v, vertices), visited, ret, vertices, edges)

        ret.insert(0, vertices[i])

    # Perform Sort using helper function
    for i in range(len(vertices)):
        if not visited[i]:
            topoSortHelper(i, visited, ret, vertices, edges)

    return ret
