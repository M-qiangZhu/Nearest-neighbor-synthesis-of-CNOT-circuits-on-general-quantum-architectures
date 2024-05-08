class Node:

    def __init__(self, val, is_steiner_point=False, left_child=None, right_child=None, parent=None):
        self.val = val
        self.is_steiner_point = is_steiner_point
        self.left_child = left_child
        self.right_child = right_child
        self.parent = parent

    def insert_left_child(self, val, is_steiner_point, **kwargs):
        if self.left_child is None:
            tmp = Node(val, is_steiner_point, **kwargs)
            tmp.parent = self
            self.left_child = tmp
        else:
            tmp = Node(val, is_steiner_point, **kwargs)
            self.left_child.parent = tmp
            tmp.left_child = self.left_child
            tmp.parent = self
            self.left_child = tmp

    def insert_right_child(self, val, is_steiner_point, **kwargs):
        if self.right_child is None:
            tmp = Node(val, is_steiner_point, **kwargs)
            tmp.parent = self
            self.right_child = tmp
        else:
            tmp = Node(val, is_steiner_point, **kwargs)
            self.right_child.parent = tmp
            tmp.right_child = self.right_child
            tmp.parent = self
            self.right_child = tmp

    # def get_root(self):
    #     return self.val

    # def set_root_val(self, val):
    #     self.val = val

    def get_value(self):
        if self is not None:
            return self.val
        return None

    def get_left_child(self):
        return self.left_child

    def get_right_child(self):
        return self.right_child


class Tree:

    def __init__(self, v: dict, e: list[tuple], r: int):
        self.vertex = v
        self.edges = e
        self.root = r

    def gen_tree(self):
        #Set value: Whether it is a Steiner point query dictionary
        global root_val, root_is_steiner_point
        v_dic = self.vertex
        edges = self.edges
        self.vertex = sorted(self.vertex.items(), key=lambda x: x[0])
        print(f"self.vertex : {self.vertex}")
        #Record Root Node
        for t in self.vertex:
            if t[0] == self.root:
                root_val, root_is_steiner_point = t[0], t[1]
                break
        # root_val, root_is_steiner_point= self.vertex[0]
        root = Node(root_val, root_is_steiner_point)


        # Traverse edge set
        generator(v_dic, root, edges)
        print(root)

        return root


def generator(v_dic, node, edges_list):
    if len(edges_list) == 0:
        return
    tar_edges = []
    for e in edges_list:
        if node.val in e:
            tar_edges.append(e)
    # Default insertion into left subtree
    if len(tar_edges) != 0:
        edge = tar_edges.pop()
        # Get the vertex values of the tree to be added
        if edge[0] == node.val:
            next_node_val = edge[1]
        else:
            next_node_val = edge[0]
        # Query whether it is a Steiner point based on the value
        is_steiner_point = v_dic.get(next_node_val)
        # 根据键值对生成节点
        node.insert_left_child(next_node_val, is_steiner_point)
        # Delete edges that have already been added to the tree from the edge set
        edges_list.remove(edge)

        generator(v_dic, node.left_child, edges_list)
    if len(tar_edges) != 0:
        edge = tar_edges.pop()
        # Get the vertex values of the tree to be added
        if edge[0] == node.val:
            next_node_val = edge[1]
        else:
            next_node_val = edge[0]
        # Query whether it is a Steiner point based on the value
        is_steiner_point = v_dic.get(next_node_val)
        node.insert_right_child(next_node_val, is_steiner_point)
        # Delete edges that have already been added to the tree from the edge set
        edges_list.remove(edge)
        generator(v_dic, node.right_child, edges_list)
    return node


# Preorder traversal
def find_node(node, val):
    # print(f"node.val : {node.val}")
    if node is None:
        return
    if node.val == val:
        return node
    if node.left_child is not None:
        return find_node(node.left_child, val)
    if node.right_child is not None:
        return find_node(node.right_child, val)
    return None



if __name__ == '__main__':
    # vertex = {0: False, 3: True, 1: False, 4: False}
    vertex = {1: False, 2: False, 3: False, 4: True}
    # vertex = {0: True, 1: False, 2: False, 3: False, 4: False}
    edges = [(1, 2), (1, 3), (3, 4)]
    tree = Tree(vertex, edges, 4)
    root = tree.gen_tree()
    print(root.get_value())  # 0
