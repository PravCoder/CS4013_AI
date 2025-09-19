# Problem: Implement the Breadth-First Search (BFS), Depth-First Search (DFS) 
# and Greedy Best-First Search (GBFS) algorithms on the graph from Figure 1 in hw1.pdf.


# Instructions:
# 1. Represent the graph from Figure 1 in any format (e.g. adjacency matrix, adjacency list).
# 2. Each function should take in the starting node as a string. Assume the search is being performed on
#    the graph from Figure 1.
#    It should return a list of all node labels (strings) that were expanded in the order they where expanded.
#    If there is a tie for which node is expanded next, expand the one that comes first in the alphabet.
# 3. You should only modify the graph representation and the function body below where indicated.
# 4. Do not modify the function signature or provided test cases. You may add helper functions. 
# 5. Upload the completed homework to Gradescope, it must be named 'hw1.py'.

# Examples:
#     The test cases below call each search function on node 'S' and node 'A'
# -----------------------------
import heapq
from collections import deque
from queue import PriorityQueue

# defines the connections of each notes where the values of the dict are a list of nodes that are conencted to the key-node
# hardcode this representation of our graph to iterate
# used globally across algorithms
adjacency_list = {
    "A": ["B", "E"],
    "B": ["A", "C", "F"],
    "C": ["B", "S", "H"],
    "D": ["S", "L"],
    "E": ["A", "F", "I"],
    "F": ["B", "E", "J", "K"],
    "G": ["M", "N", "Q"],
    "H": ["C", "K", "L"],
    "I": ["E", "J", "M"],
    "J": ["I", "K", "F", "N"],
    "K": ["J", "F", "H", "L", "N", "P"],
    "L": ["D", "K", "H", "Q"],
    "M": ["I", "G"],
    "N": ["J", "K", "P", "G"],
    "P": ["K", "N"],
    "Q": ["G", "L"],
    "S": ["C", "D"]
}

# this gives the estimated cost of the cheapest path from the node to the goal given in problem description
# our goal is always G destination
heuristic = {
    "S": 17, "A": 10, "B": 9, "C": 16, "D": 21, "E": 13, "F": 9,
    "G": 0, "H": 12, "I": 9, "J": 5, "K": 8, "L": 18,
    "M": 3, "N": 4, "P": 6, "Q": 9
}
# for dynamicallty create constatn destiantion-node our goal
DESTINATION_NODE = "G"



def BFS(start: str) -> list:
    # START: Your code here

    # inti a list to keep track of nodes that we have visited, each element is the id-label of the node
    visited_nodes = []
    # inti a queu first in first out, with the starting node passed in func arg arbitary
    queue = [start]

    # keep iterating until there are no more nodes to explore, until the queue is empty
    while queue:
        # dequeue the first node from the queue
        cur_node = queue.pop(0)
        # only process this node if it hasnt been visited yet
        if cur_node not in visited_nodes:
            # mark the node as visited by adding it to list
            visited_nodes.append(cur_node)
            # iterate all neighbor-nodes of cur-node, sorted alphabetically
            for cur_neighbor_node in sorted(adjacency_list[cur_node]):
                # if the neighbor is the destination mode add it to visited nodes and stop the algorithm
                if cur_neighbor_node == DESTINATION_NODE:
                    visited_nodes.append(DESTINATION_NODE)
                    return visited_nodes
                # if the neighbor-node hasnt been visited yet and it hasnt been in the queue, add it to the queue
                if cur_neighbor_node not in visited_nodes and cur_neighbor_node not in queue:
                    queue.append(cur_neighbor_node)
    # return the list of visited notes in the order they were visited and explored
    return visited_nodes
    # END: Your code her


def DFS(start: str) -> list:
    # START: Your code here

    # init a list to keep track of nodes that have been visited
    visited_nodes = []
    # init a stack last in first out with the starting node arbitary given to function
    stack_dfs = [start]

    # keep iterate while there are no more nodes to proces, keep iterating as long as there are nodes in the stack
    while stack_dfs:
        # pop the last node from the stack
        cur_node = stack_dfs.pop()
        # if that cur-node has not been visited yet, we only process nodes that havent been visited
        if cur_node not in visited_nodes:
            # mark it as visited
            visited_nodes.append(cur_node)
            
            # use adjacency_list to get teh neighbors of cur-node, and reverse and sort it so we can push
            # push the neighbors onto the stack in alpahbetical order smallest neighbor is visisted first
            temp = sorted(adjacency_list[cur_node], reverse=True)
            for cur_neighbor_node in temp: 
                # we only push neighbors that haven't been visited
                if cur_neighbor_node not in visited_nodes:

                    stack_dfs.append(cur_neighbor_node)
            # stop the algorithms if the gaol node is reached
            if cur_node == DESTINATION_NODE:
                break
    # reutrn list of visited nodes in correct order of dfs
    return visited_nodes 
    # END: Your code here




def GBFS(start: str) -> list:
    # START: Your code here

    # init a list to keep track of nodes we have visisted
    visited_nodes = []

    # inti a priority queue witht the given arbitary start-node and its heuristic value of start node
    priotity_queue = PriorityQueue()
    priotity_queue.put((heuristic[start], start))

    # keep iterating while there are no more nodes in the priority queue
    while not priotity_queue.empty():
        # pop the node with the smallest heuristic value priority queue
        _, cur_node = priotity_queue.get()
        # only process cur-node if it has not been visited
        if cur_node not in visited_nodes:
            # if not visited now mark it ahs visited
            visited_nodes.append(cur_node)
            if cur_node == DESTINATION_NODE:
                break
                
            # use adjacency_list to get the neighbors of cur-node
            temp = sorted(adjacency_list[cur_node])
            # iterate all neighbors of cur-node in alphbetical order
            for cur_neighbor_node in temp:
                # if the neighbor-node hasnt been visited yet push it to the priaorty queue has a tuple with its heuristic value
                if cur_neighbor_node not in visited_nodes:
                    priotity_queue.put((heuristic[cur_neighbor_node], cur_neighbor_node))

    return visited_nodes
    # END: Your code here



# test cases - DO NOT MODIFY THESE
def run_tests():
    # Test case 1: BFS starting from node 'A'
    assert BFS('A') == ['A', 'B', 'E', 'C', 'F', 'I', 'H', 'S', 'J', 'K', 'M', 'G'], "Test case 1 failed"
    
    # Test case 2: BFS starting from node 'S'
    assert BFS('S') == ['S', 'C', 'D', 'B', 'H', 'L', 'A', 'F', 'K', 'Q', 'G'], "Test case 2 failed"

    # Test case 3: DFS starting from node 'A'
    assert DFS('A') == ['A', 'B', 'C', 'H', 'K', 'F', 'E', 'I', 'J', 'N', 'G'], "Test case 3 failed"
    
    # Test case 4: DFS starting from node 'S'
    assert DFS('S') == ['S', 'C', 'B', 'A', 'E', 'F', 'J', 'I', 'M', 'G'], "Test case 4 failed"

    # Test case 5: GBFS starting from node 'A'
    assert GBFS('A') == ['A', 'B', 'F', 'J', 'N', 'G'], "Test case 5 failed"
    
    # Test case 6: GBFS starting from node 'S'
    assert GBFS('S') == ['S', 'C', 'B', 'F', 'J', 'N', 'G'], "Test case 6 failed"

    print("All test cases passed!")

if __name__ == '__main__':
    run_tests()