from typing import List
import matplotlib.pyplot as plt
import numpy as np

class Arc: 
    def __init__(self,start_node: 'Node' = None, end_node: 'Node' = None, capacity: float = 0.0):
        self.start_node = start_node
        self.end_node = end_node
        self.capacity = capacity
        self.flow = 0.0

class Node:
    def __init__(self):
        self.arcs_in:List[Arc] = []
        self.arcs_out:List[Arc] = []
        self.reached: bool = False
        self.aug_path:List[Arc] = None
'''
Source and Sink are 2 special nodes
Source is the first node
Sink is the last node in the list of nodes
'''
class GraphFlow:
    def __init__(self):
        self.nodes:List[Node] = []
        self.arcs:List[Arc] = []  
    
    def set_nb_nodes(self, nb_nodes: int) -> None:
        '''
        Add 2 nodes for source and sink
        '''
        self.nodes = [Node() for _ in range(nb_nodes + 2)]
        self.arcs.clear()  
    
    def get_nb_nodes(self) -> int:
        return len(self.nodes) - 2
    
    def get_nb_arcs(self) -> int:
        return len(self.arcs)
    
    def reset_flow(self) -> None:
        for arc in self.arcs:
            arc.flow = 0.0

    def connect_nodes_private(self, node1: 'Node' = None, node2: 'Node' = None, cap: float = 0) -> None: 
        '''
        Connect 2 nodes with a capacity
        '''
        p_arc_out = None 
        p_arc_in = None

        # Consider all arcs out from node1
        for arc in node1.arcs_out:
            if arc.end_node == node2:
                p_arc_out = arc
        
        # Consider all arcs in to node2
        for arc in node2.arcs_in:
            if arc.start_node == node1:
                p_arc_in = arc
        
        # Update or create new arc
        if p_arc_out == p_arc_in: 
            if p_arc_out:
                p_arc_out.capacity = cap
            else:
                new_arc = Arc(node1, node2, cap)
                self.arcs.append(new_arc)
                node1.arcs_out.append(new_arc)
                node2.arcs_in.append(new_arc)

    def connect_nodes(self, idx_node1: int, idx_node2: int, cap: float) -> None:
        '''
        Connect 2 nodes with a capacity
        '''
        node1 = self.nodes[idx_node1 + 2]
        node2 = self.nodes[idx_node2 + 2]
        self.connect_nodes_private(node1, node2, cap)   

    def connect_source_to_node(self, idx_node: int, cap: float) -> None:
        '''
        Connect source to a node with a capacity
        '''
        node1 = self.nodes[0]
        node2 = self.nodes[idx_node + 2]
        self.connect_nodes_private(node1, node2, cap)

    def connect_node_to_sink(self, idx_node: int, cap: float) -> None:
        '''
        Connect a node to sink with a capacity
        '''
        node1 = self.nodes[idx_node + 2]
        node2 = self.nodes[1]
        self.connect_nodes_private(node1, node2, cap)

    def node_name(self,node: 'Node') -> str:
        '''
        Get name of node
        '''
        if node == self.nodes[0]:
            return "Source"
        elif node == self.nodes[1]:
            return "Sink"
        else:
            return f"Node {self.nodes.index(node) - 2 }"

    def display_graph(self) -> None:
        '''
        Display information of graph under format:
        Node <node_name>:
        Out: ->  <node_name>: <flow> / <capacity>
        In:  <node_name> -> ... : <flow> / <capacity>   
        '''
        for node in self.nodes:
            print(f"{self.node_name(node)}:")
            for arc in node.arcs_out:
                print(f"Out: ->  {self.node_name(arc.end_node)}: {arc.flow} / {arc.capacity}")
            for arc in node.arcs_in:
                print(f"In:  {self.node_name(arc.start_node)} -> ... : {arc.flow} / {arc.capacity}")
        
    def draw(self) -> None:
        '''
        Draw the graph with matplotlib
        '''
        plt.figure(figsize=(12, 6))

        num_nodes = self.get_nb_nodes()
        positions = {}

        positions[0] = (0.05, 0.5)
        positions[1] = (0.9, 0.5)

        grid_size = int(np.ceil(np.sqrt(num_nodes)))
        x_step = 0.8 / grid_size
        y_step = 0.8 / grid_size

        for i in range(num_nodes):
            row = i // grid_size
            col = i % grid_size
            positions[i + 2] = (0.2 + col * x_step, 0.1 + row * y_step)

        for node_idx, (x, y) in positions.items():
            color = 'lightgreen' if node_idx == 0 else 'lightcoral' if node_idx == 1 else 'skyblue'
            plt.scatter(x, y, s=1000, color=color, edgecolors='black', zorder=3)
            plt.text(x, y, self.node_name(self.nodes[node_idx]), fontsize=8,
                    ha='center', va='center', zorder=4)

        for arc in self.arcs:
            start_x, start_y = positions[self.nodes.index(arc.start_node)]
            end_x, end_y = positions[self.nodes.index(arc.end_node)]

            if abs(start_x - end_x) > 0.1 or abs(start_y - end_y) > 0.1:
                control_x = (start_x + end_x) / 2
                control_y = (start_y + end_y) / 2 + 0.1
                t = np.linspace(0, 1, 100)
                curve_x = (1 - t) ** 2 * start_x + 2 * (1 - t) * t * control_x + t ** 2 * end_x
                curve_y = (1 - t) ** 2 * start_y + 2 * (1 - t) * t * control_y + t ** 2 * end_y
                plt.plot(curve_x, curve_y, color='gray', alpha=0.7)

                arrow_t = 0.7
                arrow_x = (1 - arrow_t) ** 2 * start_x + 2 * (1 - arrow_t) * arrow_t * control_x + arrow_t ** 2 * end_x
                arrow_y = (1 - arrow_t) ** 2 * start_y + 2 * (1 - arrow_t) * arrow_t * control_y + arrow_t ** 2 * end_y
                dx = 2 * (1 - arrow_t) * (control_x - start_x) + 2 * arrow_t * (end_x - control_x)
                dy = 2 * (1 - arrow_t) * (control_y - start_y) + 2 * arrow_t * (end_y - control_y)
                plt.arrow(arrow_x, arrow_y, dx * 0.01, dy * 0.01, head_width=0.02, head_length=0.03, fc='gray', ec='gray')

                label_t = 0.5
                label_x = (1 - label_t) ** 2 * start_x + 2 * (1 - label_t) * label_t * control_x + label_t ** 2 * end_x
                label_y = (1 - label_t) ** 2 * start_y + 2 * (1 - label_t) * label_t * control_y + label_t ** 2 * end_y
                plt.text(label_x, label_y, f"{arc.flow}/{arc.capacity}", fontsize=8, color='red',
                        ha='center', va='center', backgroundcolor='white')
            else:
                plt.arrow(start_x, start_y, (end_x - start_x) * 0.8, (end_y - start_y) * 0.8,
                        head_width=0.005, head_length=0.005, fc='gray', ec='gray', alpha=0.7, length_includes_head=True)

                mid_x, mid_y = (start_x + end_x) / 2, (start_y + end_y) / 2
                plt.text(mid_x, mid_y, f"{arc.flow}/{arc.capacity}", fontsize=8, color='red',
                        ha='center', va='center', backgroundcolor='white')

        plt.axis('off')
        plt.title("Graph Visualization", fontsize=16)
        plt.show()

    
    def augmenting_path(self) -> None:
        '''
        Find an augmenting path from source to sink by using BFS
        '''
        queue = [self.nodes[0]]
        current = 0 

        for node in self.nodes:
            node.reached = False 
            node.aug_path = None

        self.nodes[0].reached = True
        
        while current < len(queue) and not self.nodes[1].reached:
            current_node = queue[current]

            for arc in current_node.arcs_out:
                if arc.flow < arc.capacity and not arc.end_node.reached:
                    queue.append(arc.end_node)
                    arc.end_node.reached = True
                    arc.end_node.aug_path = arc
            current += 1
    
    def ford_fulkerson(self):
        '''
        Implement Ford-Fulkerson algorithm to find the maximum flow
        '''
        while True:
            self.augmenting_path()
            if not self.nodes[1].reached:
                break
                
            min_residual_capacity = float('inf')
            backtrace_node = self.nodes[1]

            while backtrace_node.aug_path:
                p_arc = backtrace_node.aug_path
                min_residual_capacity = min(min_residual_capacity, p_arc.capacity - p_arc.flow)
                backtrace_node = p_arc.start_node

            backtrace_node = self.nodes[1]
            while backtrace_node.aug_path:
                p_arc = backtrace_node.aug_path
                p_arc.flow += min_residual_capacity
                backtrace_node = p_arc.start_node
            
            for node in self.nodes:
                node.reached = False

    def cut_from_source(self):
        '''
        Find the set of nodes reachable from source
        '''
        queue = [self.nodes[0]]
        current = 0 

        for node in self.nodes:
            node.reached = False
        
        self.nodes[0].reached = True

        #print("\nStarting BFS from Source:")
        while current < len(queue):
            current_node = queue[current]
            #print(f"Visiting node: {self.node_name(current_node)}")
            for arc in current_node.arcs_out:
                if arc.capacity - arc.flow > 0 and not arc.end_node.reached:
                    #print(f"  Adding node {self.node_name(arc.end_node)} to queue "
                    #    f"(residual capacity = {arc.capacity - arc.flow})")
                    queue.append(arc.end_node)
                    arc.end_node.reached = True
            current += 1
        
        vect_S = []
        vect_T = []

        for idx in range(2, len(self.nodes)):
            if self.nodes[idx].reached:
                vect_S.append(idx - 2)
            else:
                vect_T.append(idx - 2) 
        
        return vect_S, vect_T

    def verify_min_cut(self, vect_S) -> float:
        '''
        Find the minimum cut capacity
        '''
        min_cut_capacity = 0
        for node_idx in vect_S:
            node = self.nodes[node_idx + 2]
            for arc in node.arcs_out:
                if not arc.end_node.reached:  
                    min_cut_capacity += arc.capacity
        return min_cut_capacity

def main() -> None:
    graph = GraphFlow()
    graph.set_nb_nodes(3)

    graph.connect_nodes(0, 1, 3)
    graph.connect_nodes(0, 2, 1)
    graph.connect_nodes(1, 2, 3)
    graph.connect_source_to_node(0, 4)
    graph.connect_source_to_node(2,3)
    graph.connect_node_to_sink(1, 2)
    graph.connect_node_to_sink(2, 4)

    print("Before running Ford-Fulkerson:")
    graph.display_graph()
    #graph.draw()

    graph.ford_fulkerson()

    print("\nAfter running Ford-Fulkerson:")
    graph.display_graph()
    #graph.draw()

    print("\nResidual graph:")
    for node in graph.nodes:
        for arc in node.arcs_out:
            print(f"Arc from {graph.node_name(arc.start_node)} to {graph.node_name(arc.end_node)}: flow = {arc.flow}, capacity = {arc.capacity}, residual capacity = {arc.capacity - arc.flow}")

    vect_s, vect_t = graph.cut_from_source()
    print("\nNodes in set S (reachable from source):", vect_s)

    min_cut_capacity = graph.verify_min_cut(vect_s)
    print(f"\nMin-cut capacity: {min_cut_capacity}")

if __name__ == "__main__":  
    main()
