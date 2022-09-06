# Dijkstra's shortest path algorithm adapted for directional and weighted vertices

from collections import defaultdict
import sys


class Heap():
    def __init__(self):
        self.array = []
        self.size = 0
        self.pos = []

    def newMinHeapNode(self, v, dist):
        minHeapNode = [v, dist]
        return minHeapNode

    # A utility function to swap two nodes
    # of min heap. Needed for min heapify
    def swapMinHeapNode(self, a, b):
        t = self.array[a]
        self.array[a] = self.array[b]
        self.array[b] = t

    # A standard function to heapify at given idx
    # This function also updates position of nodes
    # when they are swapped.Position is needed
    # for decreaseKey()
    def minHeapify(self, idx):
        smallest = idx
        left = 2 * idx + 1
        right = 2 * idx + 2

        if left < self.size and self.array[left][1] < self.array[smallest][1]:
            smallest = left

        if right < self.size and self.array[right][1] < self.array[smallest][1]:
            smallest = right

        # The nodes to be swapped in min
        # heap if idx is not smallest
        if smallest != idx:
            # Swap positions
            self.pos[self.array[smallest][0]] = idx
            self.pos[self.array[idx][0]] = smallest

            # Swap nodes
            self.swapMinHeapNode(smallest, idx)

            self.minHeapify(smallest)

    # Standard function to extract minimum
    # node from heap
    def extractMin(self):

        # Return NULL wif heap is empty
        if self.isEmpty() == True:
            return

        # Store the root node
        root = self.array[0]

        # Replace root node with last node
        lastNode = self.array[self.size - 1]
        self.array[0] = lastNode

        # Update position of last node
        self.pos[lastNode[0]] = 0
        self.pos[root[0]] = self.size - 1

        # Reduce heap size and heapify root
        self.size -= 1
        self.minHeapify(0)

        return root

    def isEmpty(self):
        return True if self.size == 0 else False

    def decreaseKey(self, v, dist):

        # Get the index of v in heap array

        i = self.pos[v]

        # Get the node and update its dist value
        self.array[i][1] = dist

        # Travel up while the complete tree is
        # not hepified. This is a O(Logn) loop
        while i > 0 and self.array[i][1] < self.array[self.python2round(((i - 1) / 2))][1]:
            # Swap this node with its parent
            self.pos[self.array[i][0]] = self.python2round(((i - 1) / 2))
            self.pos[self.array[self.python2round(((i - 1) / 2))][0]] = i
            self.swapMinHeapNode(i, self.python2round(((i - 1) / 2)))

            # move to parent index
            i = self.python2round(((i - 1) / 2));

    # A utility function to check if a given
    # vertex 'v' is in min heap or not
    def isInMinHeap(self, v):

        if self.pos[v] < self.size:
            return True
        return False

    def python2round(self, f):
        if round(f + 1) - round(f) != 1:
            return int(f + abs(f) / f * 0.5)
        return int(round(f))

class Graph():
    def __init__(self, V):
        self.V = V
        self.graph = defaultdict(list)

    # Adds an edge to an undirected graph
    def addEdge(self, src, dest, weight):

        # Add an edge from src to dest. A new node
        # is added to the adjacency list of src. The
        # node is added at the beginning. The first
        # element of the node has the destination
        # and the second elements has the weight
        newNode = [dest, weight]
        self.graph[src].insert(0, newNode)

        # Since graph is undirected, add an edge
        # from dest to src also
        # newNode = [src, weight]
        # self.graph[dest].insert(0, newNode)

    # The main function that calculates distances
    # of shortest paths from src to all vertices.
    # It is a O(ELogV) function

    def printPath(self, parent, j):

        # Base Case : If j is source
        if parent[j] == -1:
            print(j, end = " ")
            return
        self.printPath(parent, parent[j])
        print(j, end = " ")

    # A utility function to print
    # the constructed distance
    # array
    def printSolution(self, src, dist, parent):
        print("Vertex \t\tDistance from Source\tPath")
        for i in range(1, len(dist)):
            print("\n%d --> %d \t%d \t\t" % (src, i, dist[i]), end=' '),
            self.printPath(parent, i)
        print('')

    def trace_route(self, parent, j, route=[]):
        # Base Case : If j is source
        if parent[j] == -1:
            route.append(j)
            return route[::-1]
        route.append(j)
        return self.trace_route(parent, parent[j], route)

    def tree_climb(self, src, dist, parent):
        tree = []
        for i in range(1, len(dist)):
            route = self.trace_route(parent, i, [])
            tree.append([src, i, route, dist[i]])
        self.tree = tree


    def dijkstra(self, src):

        V = self.V  # Get the number of vertices in graph
        dist = []  # dist values used to pick minimum
        # weight edge in cut

        # minHeap represents set E
        minHeap = Heap()

        # Initialize min heap with all vertices.
        # dist value of all vertices
        for v in range(V):
            dist.append(sys.maxsize)
            minHeap.array.append(minHeap.newMinHeapNode(v, dist[v]))
            minHeap.pos.append(v)

        # Make dist value of src vertex as 0 so
        # that it is extracted first
        minHeap.pos[src] = src
        dist[src] = 0
        minHeap.decreaseKey(src, dist[src])

        # Initially size of min heap is equal to V
        minHeap.size = V

        parent = [-1] * V

        # In the following loop,
        # min heap contains all nodes
        # whose shortest distance is not yet finalized.
        while minHeap.isEmpty() == False:

            # Extract the vertex
            # with minimum distance value
            newHeapNode = minHeap.extractMin()
            u = newHeapNode[0]

            # Traverse through all adjacent vertices of
            # u (the extracted vertex) and update their
            # distance values
            for pCrawl in self.graph[u]:
                v = pCrawl[0]
                if minHeap.isInMinHeap(v) and dist[u] != sys.maxsize and pCrawl[1] + dist[u] < dist[v]:
                    dist[v] = pCrawl[1] + dist[u]
                    #print("The vertex is: %s" % v)
                    # update distance value
                    # in min heap also
                    minHeap.decreaseKey(v, dist[v])
                    parent[v] = u

        #self.printSolution(src, dist, parent)
        self.tree_climb(src, dist, parent)