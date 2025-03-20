class Arc:
    def __init__(self, start_node, end_node, capacity):
        self.start_node = start_node
        self.end_node = end_node
        self.capacity = capacity
        self.flow = 0.0

class Node:
    def __init__(self):
        self.arcs_out = []
        self.arcs_in = []
        self.reached = False
        self.p_aug_path_arc = None

class GraphFlow:
    def __init__(self):
        self.nodes = []
        self.arcs = []

    def set_nb_nodes(self, nb_nodes):
        self.nodes = [Node() for _ in range(nb_nodes + 2)]
        self.arcs.clear()

    def get_nb_nodes(self):
        return len(self.nodes) - 2

    def get_nb_arcs(self):
        return len(self.arcs)

    def reset_flow(self):
        for arc in self.arcs:
            arc.flow = 0.0

    def connect_nodes_private(self, node1, node2, cap):
        p_arc_out = None
        p_arc_in = None

        for arc in node1.arcs_out:
            if arc.end_node == node2:
                p_arc_out = arc
        for arc in node2.arcs_in:
            if arc.start_node == node1:
                p_arc_in = arc

        if p_arc_out == p_arc_in:
            if p_arc_out:
                p_arc_out.capacity = cap
            else:
                new_arc = Arc(node1, node2, cap)
                self.arcs.append(new_arc)
                node1.arcs_out.append(new_arc)
                node2.arcs_in.append(new_arc)

    def connect_nodes(self, idx_node1, idx_node2, cap):
        node1 = self.nodes[idx_node1 + 2]
        node2 = self.nodes[idx_node2 + 2]
        self.connect_nodes_private(node1, node2, cap)

    def connect_source_to_node(self, idx_node, cap):
        node1 = self.nodes[0]
        node2 = self.nodes[idx_node + 2]
        self.connect_nodes_private(node1, node2, cap)

    def connect_node_to_sink(self, idx_node, cap):
        node1 = self.nodes[idx_node + 2]
        node2 = self.nodes[1]
        self.connect_nodes_private(node1, node2, cap)

    def find_augmenting_path(self):
        Q = [self.nodes[0]]
        current = 0
        for node in self.nodes:
            node.reached = False
            node.p_aug_path_arc = None

        self.nodes[0].reached = True
        while current < len(Q) and not self.nodes[1].reached:
            p_current_node = Q[current]
            for arc in p_current_node.arcs_out:
                if arc.flow < arc.capacity and not arc.end_node.reached:
                    Q.append(arc.end_node)
                    arc.end_node.reached = True
                    arc.end_node.p_aug_path_arc = arc
            current += 1

    def ford_fulkerson(self):
        b_path_found = False
        while True:
            self.find_augmenting_path()
            b_path_found = self.nodes[1].reached

            if b_path_found:
                min_residual_capacity = float('inf')
                p_arc = self.nodes[1].p_aug_path_arc
                p_backtrack_node = self.nodes[1]

                while p_backtrack_node.p_aug_path_arc:
                    p_arc = p_backtrack_node.p_aug_path_arc
                    min_residual_capacity = min(min_residual_capacity, p_arc.capacity - p_arc.flow)
                    p_backtrack_node = p_arc.start_node

                p_backtrack_node = self.nodes[1]
                while p_backtrack_node.p_aug_path_arc:
                    p_arc = p_backtrack_node.p_aug_path_arc
                    p_arc.flow += min_residual_capacity
                    p_backtrack_node = p_arc.start_node

                for node in self.nodes:
                    node.reached = False
            else:
                break

    def print_graph(self):
        for node in self.nodes:
            print(f"Node {self.node_name(node)}")
            for arc in node.arcs_out:
                print(f"Out  -> {self.node_name(arc.end_node)} : {arc.flow}/{arc.capacity}")
            for arc in node.arcs_in:
                print(f"In ->  {self.node_name(arc.start_node)} -> : {arc.flow}/{arc.capacity}")

    def node_name(self, node):
        if node == self.nodes[0]:
            return "s"
        elif node == self.nodes[1]:
            return "t"
        else:
            return str(self.nodes.index(node) - 2)

# Test the GraphFlow class
def test_graph_flow():
    graph = GraphFlow()
    graph.set_nb_nodes(6)  # Set 6 nodes in the graph

    # Connect nodes with capacities
    graph.connect_nodes(0, 1, 10)
    graph.connect_nodes(0, 2, 5)
    graph.connect_nodes(1, 3, 15)
    graph.connect_nodes(2, 3, 10)
    graph.connect_nodes(3, 4, 10)
    graph.connect_nodes(4, 1, 5)
    graph.connect_source_to_node(0, 10)
    graph.connect_node_to_sink(4, 10)

    print("Before running Ford-Fulkerson:")
    graph.print_graph()

    graph.ford_fulkerson()

    print("\nAfter running Ford-Fulkerson:")
    graph.print_graph()

# Run the test
test_graph_flow()
