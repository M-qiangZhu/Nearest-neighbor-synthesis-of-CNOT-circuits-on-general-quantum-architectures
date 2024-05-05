import time
from itertools import combinations

import numpy as np
from matplotlib import pyplot as plt

from qiskit import QuantumCircuit, IBMQ
from qiskit import Aer, transpile
from qiskit.providers.ibmq.runtime import IBMRuntimeService
from qiskit.visualization import plot_histogram
# from qiskit.test.mock import FakeYorktown
from qiskit.providers.fake_provider import FakeYorktown

from my_tools.graph import IbmQuito, IbmqGuadalupe, IbmqGuadalupe_new, IbmqKolkata_new, IbmqManhattan_new, IbmqLagos_new, IbmqAlmaden_new, IbmqTokyo_new, WuKong, WuKong_new
from my_tools.my_parity_maps import CNOT_tracker
from networkx.algorithms import approximation
from my_tools.my_linalg import Mat2
from my_tools.tree import Tree


def rotate(mat):
    """
    Generate numpy 2D array of center rotation
    :param mat: the matrix to be processed
    :return: rotated two-dimensional array
    """
    m = mat.data
    m = m[::-1, ::-1]
    return m


def get_circuits_to_matrix(file_name, **kwargs):
    """
    Reading quantum circuits from file
    :param file_name:
    :param kwargs:
    :return:
    """
    circuit = CNOT_tracker.from_qasm_file(file_name)
    print(f"Initial number of gates: {len(circuit.gates)}")
    mat = circuit.matrix
    print(type(mat))
    return mat


def get_center_flipped_matrix(mat):
    """
    Get the center flipped matrix of a matrix
    :param mat: original matrix
    :return: Center-flipped matrix
    """
    # Generate a new matrix to store the updated data
    fli_mat = mat.copy()
    # Get the rotated two-dimensional array
    rotate_arr = rotate(mat)
    # Assign the rotated two-dimensional array to the original matrix
    fli_mat.data = rotate_arr
    return fli_mat


def get_col(m, col_index):
    """
    Get the specified column based on the matrix
    :param m: matrix
    :param col_index: The index of the column to be obtained
    :return:
    """
    return m.data[:, col_index]


def get_row(m, row_index):
    """
    Get the specified row from the matrix
    :param m: matrix
    :param row_index: The index of the row to be obtained
    :return:
    """
    return m.data[row_index, :].tolist()


def get_ones_index_col(row_index, col_list):
    """
    Get the row number of the element with a value of 1 from the current column index row_index downwards, including row_index
    :param col_list: current column of matrix
    :param row_index: starting row index
    :return: list of indices with value 1
    """
    v_list = []
    for num in range(len(col_list)):
        if col_list[num] == 1:
            v_list.append(num)
    return v_list


def confirm_steiner_point(old_list, new_list):
    """
    Get the set of vertices used to generate the tree {0: False, 1: True}
    The element currently 1, the vertex element given by the steiner tree => generates a set of True and False
    :param old_list: The vertices whose current column value is 1 in the matrix
    :param new_list: The vertices of the steiner tree obtained by nx
    :return:
    """
    v_dict = {}
    for new_v in new_list:
        if new_v in old_list:
            v_dict[new_v] = False
        else:
            v_dict[new_v] = True
    return v_dict


def is_cut_point(g, n):
    """
    Determine whether the current point is a cutting point
    :param g: current picture
    :param n: current vertex
    :return: Whether it is a cut point
    """
    degree = g.degree[n]
    if degree > 1:
        return True
    return False


def row_add(row1, row2):
    """Add r0 to r1"""
    for i, v in enumerate(row1):
        if v:
            row2[i] = 0 if row2[i] else 1  # When a value in row1 is 1, invert the value of the corresponding position in row2
    return row2


def col_eli_set_steiner_point(m, node, col, cnot_list):
    """
    The first step of column elimination: Steiner point is set to 1
    :param m:
    :param node:
    :param col:
    :param cnot_list:
    :return:
    """
    if node is None:
        return
    if node.left_child is not None:
        m, cnot_list = col_eli_set_steiner_point(m, node.left_child, col, cnot_list)
    if node.right_child is not None:
        m, cnot_list = col_eli_set_steiner_point(m, node.right_child, col, cnot_list)
    # Get the value of the current column corresponding to the current index
    if node.parent is not None:
        j = node.val
        k = node.parent.val
        if col[j] == 1 and col[k] == 0:
            m.row_add(j, k)
            cnot_list.append((j, k))
            # print("new matrix")
            # print(matrix)
            # print(f"CNOT_list : {cnot_list}")
            # return matrix
    return m, cnot_list


def col_eli_down_elim(m, node, cnot_list):
    """
    The second step of column elimination is downward elimination
    :param m:
    :param node:
    :param cnot_list:
    :return:
    """
    if node is None:
        return
    if node.left_child is not None:
        m, cnot_list = col_eli_down_elim(m, node.left_child, cnot_list)
    if node.right_child is not None:
        m, cnot_list = col_eli_down_elim(m, node.right_child, cnot_list)
    # Add the row corresponding to the current node to the row corresponding to the child node
    parent = node.val
    if node.left_child is not None:
        left = node.left_child.val
        m.row_add(parent, left)
        cnot_list.append((parent, left))
    if node.right_child is not None:
        right = node.right_child.val
        m.row_add(parent, right)
        cnot_list.append((parent, right))
    return m, cnot_list


def col_elim(m, start_node, col, cnot_list):
    step1_m, step1_cnots = col_eli_set_steiner_point(m, start_node, col, cnot_list)
    print("column elimination step1_m :")
    print(step1_m)
    print(f"column elimination step1_cnots : {step1_cnots}")
    result_m, cnot = col_eli_down_elim(m, start_node, cnot_list)
    # tmp_cnot += cnot
    return result_m, cnot


def get_ei(m, i):
    # Generate an identity matrix of corresponding size
    n_qubits = m.rank()
    matrix = Mat2(np.identity(n_qubits))
    # Take the corresponding row from the identity matrix as ei
    ei = matrix.data[i].tolist()
    return ei


def is_row_eql(row1, row2):
    if len(row1) == len(row2):
        length = len(row1)
        if row1 == row2:
            print("Two lines are equal")
        else:
            print("Two rows are not equal")
    else:
        print("Two rows of data do not match!!!")


def find_set_j(m, tar_row_index, row_tar, ei):
    # Based on the target row, generate a list to be traversed
    length = m.rank()
    all_set = []
    print()
    for i in range(1, length):
        all_set += list(combinations([j for j in range(tar_row_index, length)], i))
    for j_set in all_set:
        # Temporarily store ei, used to restore ei
        tmp_row = ei.copy()
        for i in j_set:
            row = get_row(m, i)
            row_add(row, tmp_row)
        if tmp_row == row_tar:
            return list(j_set)


class TreeNode:
    def __init__(self, value, level, path):
        self.value = value
        self.level = level
        self.path = path  # The path of the current node, representing a list from the root to the current node
        self.left = None
        self.right = None


def build_tree(level, path, max_level):
    if level == max_level:
        return None  # Maximum level reached, stop building

    # Create a node for the current layer
    node = TreeNode(path[-1] if path else 0, level, path)

    # Recursively build left and right subtrees
    node.left = build_tree(level + 1, path + [0], max_level) if level < max_level else None
    node.right = build_tree(level + 1, path + [1], max_level) if level < max_level else None

    return node  # 返回当前节点


def print_tree(node, indent=""):
    if node:
        # Print nodes and their paths
        print(f"{indent}{node.value} (Path: {'->'.join(map(str, node.path))})")
        # Recursively print left subtree
        print_tree(node.left, indent + "  ")
        # Recursively print the right subtree
        print_tree(node.right, indent + "  ")


def dfs_search(node, target):
    if node is None:
        return None

    # If the path matches the target, the target is found
    if node.path == target:
        return node.path

    # Continue depth first search
    left_result = dfs_search(node.left, target)
    if left_result:
        return left_result

    right_result = dfs_search(node.right, target)
    if right_result:
        return right_result

    return None


def find_set_j_new(m, tar_row_index, row_tar, ei):
    """
    Implemented using pruning algorithm
    :param m:
    :param tar_row_index:
    :param row_tar:
    :param ei:
    :return:
    """
    # 1. According to the rank of matrix, generate a full binary tree with the left child node being 0 and the right child node being 1.
    root = build_tree(0, [], m + 1)  # Build the root node of the tree

    # 2. Starting from the first layer and going down, perform a deep traversal search
    target = [0, 1, 0, 1]
    result = dfs_search(root, target)
    print(result)
    # 3. Determine whether the current layer, the current column and the columns on the selected path satisfy the value of the corresponding column Ri + ei
    # 4. If satisfied, find the next level
    # 5. If not satisfied, prune, return to the previous level, and continue searching.

    # Based on the target row, generate a list to be traversed
    length = m.rank()
    all_set = []
    print()
    for i in range(1, length):
        all_set += list(combinations([j for j in range(tar_row_index, length)], i))
    for j_set in all_set:
        # Temporarily store ei, used to restore ei
        tmp_row = ei.copy()
        for i in j_set:
            row = get_row(m, i)
            row_add(row, tmp_row)
        if tmp_row == row_tar:
            return list(j_set)


# def find_set_j(m, tar_row_index, row_tar, ei):
#     length = m.rank()
#
#     def generate_combinations():
#         for i in range(1, length):
#             for j_set in combinations(range(tar_row_index, length), i):
#                 yield j_set
#
#     for j_set in generate_combinations():
#         tmp_row = ei.copy()
#         for i in j_set:
#             row = get_row(m, i)
#             row_add(row, tmp_row)
#         if tmp_row == row_tar:
#             return list(j_set)


def row_elim_step1(m, node, cnot_list):
    if node is None:
        return
    # Get the value of the current column corresponding to the current index
    if node.parent is not None and node.is_steiner_point is True:
        j = node.val
        k = node.parent.val
        m.row_add(j, k)
        cnot_list.append((j, k))
    if node.left_child is not None:
        m, cnot_list = row_elim_step1(m, node.left_child, cnot_list)
    if node.right_child is not None:
        m, cnot_list = row_elim_step1(m, node.right_child, cnot_list)
    return m, cnot_list


def row_elim_step2(m, node, cnot_list):
    if node is None:
        return
    if node.left_child is not None:
        m, cnot_list = row_elim_step2(m, node.left_child, cnot_list)
    if node.right_child is not None:
        m, cnot_list = row_elim_step2(m, node.right_child, cnot_list)
    if node.parent is not None:
        # Add the row corresponding to the current node to the row corresponding to the parent node
        parent = node.parent.val
        child = node.val
        m.row_add(child, parent)
        cnot_list.append((child, parent))
    return m, cnot_list


def row_elim(m, node, cnot_list):
    step1_m, step1_cnots = row_elim_step1(m, node, cnot_list)
    print("row elimination step1_m :")
    print(step1_m)
    print(f"row elimination step1_cnots : {step1_cnots}")
    result_m, cnot = row_elim_step2(m, node, cnot_list)
    # tmp_cnot += cnot
    return result_m, cnot


def get_node_eli_order(g):
    nodes = list(g.nodes)
    print(nodes)
    print(type(nodes))
    node_eli_order = []
    while len(nodes) != 0:
        for node in nodes:
            if is_cut_point(g, node) is False:
                node_eli_order.append(node)
                g.remove_node(node)
                nodes.remove(node)
            # else:
            #     break
    return node_eli_order


def get_gate(filepath):
    gate_list = []
    f = open(filepath, 'r')
    data = f.readlines()
    for line in data:
        xy = line.split()
        gate_list.append([eval(xy[0]), eval(xy[1])])
    return gate_list


def get_qiskit_circ(gate_list):
    in_circ = QuantumCircuit(5)
    for a in gate_list:
        in_circ.cx(a[0], a[1])
    return in_circ


def test_one_col_eli():
    # circuit_file = "./circuits/steiner/5qubits/10/Original9.qasm"  # 3
    circuit_file = "./circuits/steiner/5qubits/10/Original11.qasm"  # 1, 4
    matrix = get_circuits_to_matrix(circuit_file)
    print("matrix :")
    print(matrix)
    # 获取 ibmq_quito 架构的图
    ibmq_quito = IbmQuito()
    graph = ibmq_quito.get_graph()
    # 画图
    # ibmq_quito.draw_graph()

    # 设置当前索引
    index = 0
    # 获取当前列数据
    col_list = get_col(matrix, index)
    print(f"col_list{col_list}")
    # 获取当前列中为1的顶点
    col_ones = get_ones_index_col(index, col_list)
    print(f"col_ones : {col_ones}")
    # 如果对角线元素为0, 需要单独处理
    if col_list[index] == 0:
        # 用来生成Steiner树的顶点
        # col_ones.append(int(col_list[index]))
        v_st = col_ones + [int(index)]
        v_st = sorted(v_st)
        print(f"对角线元素为 0 时 v_st : {v_st}")
    else:
        v_st = col_ones
        print(f"对角线元素不为 0 时 v_st : {v_st}")
    # --------------------------------------------------------------------
    # 根据值为 1 的顶点集合, 生成Steiner树
    tree_from_nx = approximation.steiner_tree(graph, v_st)
    # 获取Steiner树中的顶点
    tmp_v = tree_from_nx.nodes
    print(f"tmp_v : {tmp_v}")
    # 获取用来生成树的顶点集合
    vertex = confirm_steiner_point(col_ones, tmp_v)
    # 获取用来生成树的边集合
    edges = [e for e in tree_from_nx.edges]
    print(f"vertex : {vertex}")
    print(f"edges : {edges}")
    # 生成树
    tree = Tree(vertex, edges)
    root = tree.gen_tree()
    # print(root.get_value())
    col = get_col(matrix, index)
    CNOT_list = []
    matrix, cnot = col_elim(matrix, root, col, CNOT_list)
    print(f"列消元后的矩阵 : ")
    print(matrix)
    print(f"列消元过程中使用的CNOT门: {cnot}")
    print("-" * 100)
    return matrix


def test_one_row_eli(m):
    # 获取 ibmq_quito 架构的图
    ibmq_quito = IbmQuito()
    graph = ibmq_quito.get_graph()
    index = 0
    ei = get_ei(m, index)
    print(f"ei : {ei}")
    print(f"ei类型: {type(ei)}")
    print(f"ei中数据的类型: {type(ei[0])}")
    # 获取当前被消除的行
    row_target = get_row(m, index)
    # print(f"row_{index} : {row_i}")
    # print(f"row_{index}类型: {type(row_i)}")
    # # 因为数据为引用型, row_i会被覆盖
    # row_target = row_add(ei, row_i)
    # print(f"row_target : {row_target}")
    # print(f"row_{index}: {row_i}")
    # print(ei == row_i)
    # is_row_eql(ei, row_i)
    # print(row_target == row_i)
    # is_row_eql(row_target, row_i)

    # 手动测试 row1 + row2 + row4
    # row_1 = get_row(m, 1)
    # row_2 = get_row(m, 2)
    # row_4 = get_row(m, 4)
    # print(f"row_1 : {row_1}")
    # print(f"row_2 : {row_2}")
    # print(f"row_4 : {row_4}")
    # row_add(row_1, row_2)
    # print(f"row_2更新为 : {row_2}")
    # row_add(row_2, row_4)
    # print(f"row_4更新为 : {row_4}")
    # print(row_4 == row_target)
    # 从剩余行中找到满足条件的集合{j}
    j_set = find_set_j(m, index + 1, row_target, ei)
    print(f"j_set : {j_set}")
    print(f"j_set长度为 : {len(j_set)}")

    # j_set = [1, 4, 2]
    # 根据j和i生成Steiner树
    node_set = sorted([index] + j_set)
    print(f"node_set : {node_set}")
    tree_from_nx = approximation.steiner_tree(graph, node_set)
    # 获取Steiner树中的顶点
    tmp_v = tree_from_nx.nodes
    print(f"tmp_v : {tmp_v}")
    # 获取用来生成树的顶点集合
    vertex = confirm_steiner_point(node_set, tmp_v)
    # 获取用来生成树的边集合
    edges = [e for e in tree_from_nx.edges]
    print(f"vertex : {vertex}")
    print(f"edges : {edges}")
    # 生成树
    tree = Tree(vertex, edges)
    root = tree.gen_tree()
    print(f"root.get_value() : {root.get_value()}")
    # 记录CNOT门
    CNOT_list = []
    # # 第一步: 根据j集合消元, 从根节点开始遍历树, 遇到Steiner点后, 将Steiner点对应行加到它父节点所在行
    # m, cnot = row_elim_step1(m, root, CNOT_list)
    # print(f"当前matrix : ")
    # print(m)
    # print(f"cnot : {cnot}")
    #
    # # 第二步: 从根节点开始遍历树, 将每一行加到父节点
    # m, cnot = row_elim_step2(m, root, CNOT_list)
    # print(f"当前matrix : ")
    # print(m)
    # print(f"cnot : {cnot}")

    # 执行 行消元
    m, cnot = row_elim(m, root, CNOT_list)
    print(f"行消元后的矩阵 : ")
    print(m)
    print(f"行消元过程中使用的CNOT门: {cnot}")
    print("-" * 100)
    print(f"列消元后的矩阵 : ")
    print(m)
    print(f"列消元过程中使用的CNOT门: {cnot}")
    print("-" * 100)
    return m


def test_matrix_rows_add():
    circuit_file = "./circuits/steiner/5qubits/10/Origina7.qasm"
    matrix = get_circuits_to_matrix(circuit_file)
    print("matrix :")
    print(matrix)
    matrix.row_add(1, 0)
    print("new matrix:")
    print(matrix)


def test_cut_point():
    circuit_file = "./circuits/steiner/5qubits/10/Original4.qasm"
    matrix = get_circuits_to_matrix(circuit_file)
    print("matrix :")
    print(matrix)
    col_list = get_col(matrix, 0)
    print(col_list)
    col_ones = get_ones_index_col(0, col_list)
    print(col_ones)
    ibmq_quito = IbmQuito()
    graph = ibmq_quito.get_graph()
    tree_from_nx = approximation.steiner_tree(graph, col_ones)
    tmp_v = tree_from_nx.nodes
    for v in tmp_v:
        print(v, is_cut_point(graph, v))


def test_col_eli():
    circuit_file = "./circuits/steiner/5qubits/10/Original9.qasm"
    # circuit_file = "./circuits/steiner/5qubits/10/Original11.qasm"  # 1, 4
    matrix = get_circuits_to_matrix(circuit_file)
    print("matrix :")
    print(matrix)
    ibmq_quito = IbmQuito()
    graph = ibmq_quito.get_graph()
    # ibmq_quito.draw_graph()

    for index in range(matrix.rank()):
        print(f"index : {index}")
        col_list = get_col(matrix, index)
        print(col_list)
        col_ones = get_ones_index_col(index, col_list)
        print(f"col_ones : {col_ones}")
        if col_list[index] == 0:
            # col_ones.append(int(col_list[index]))
            v_st = col_ones + [int(col_list[index])]
            v_st = sorted(v_st)
        else:
            v_st = col_ones
        # --------------------------------------------------------------------
        tree_from_nx = approximation.steiner_tree(graph, v_st)
        tmp_v = tree_from_nx.nodes
        print(f"tmp_v : {tmp_v}")
        vertex = confirm_steiner_point(col_ones, tmp_v)
        edges = [e for e in tree_from_nx.edges]
        print(f"vertex : {vertex}")
        print(f"edges : {edges}")
        tree = Tree(vertex, edges)
        root = tree.gen_tree()
        print(f"root.get_value() : {root.get_value()}")
        col = get_col(matrix, index)
        CNOT_list = []
        matrix, cnot = col_elim(matrix, root, col, CNOT_list)
        print(f"matrix : ")
        print(matrix)
        print(f"cnot : {cnot}")


def test_eli_one_cul_one_row():
    matrix = test_one_col_eli()
    print(matrix)
    matrix = test_one_row_eli(matrix)
    print(matrix)


def test_get_node_eli_order():
    ibmq_quito = IbmQuito()
    graph = ibmq_quito.get_graph()
    print(get_node_eli_order(graph))


def col_row_eli_of_ibmquatio(file_name):
    # 1. Get a diagram of the ibmq_quito architecture
    ibmq_quito = IbmQuito()
    graph = ibmq_quito.get_graph()
    # 2. Read line generation matrix
    circuit_file = file_name
    # circuit_file = "./circuits/steiner/5qubits/10/Original11.qasm"  # 1, 4
    matrix = get_circuits_to_matrix(circuit_file)

    print("matrix :")
    print(matrix)
    # 3. Depending on whether it is a cut point, generate an elimination sequence
    eli_order = get_node_eli_order(graph.copy())
    print(f"eli_order : {eli_order}")
    # 4. Record the CNOT gate used to generate the line
    CNOT = []
    # 5. enter loop
    # for index in range(rank):
    eli_order = [0, 1, 2, 3, 4]
    # eli_order = [0, 4, 3, 1, 2]
    # Row and column elimination is performed by default
    col_flag = True
    for index in eli_order:
        # column elimination
        # Get current column data
        col_list = get_col(matrix, index)
        print(f"col_list{col_list}")
        # Get the vertex with 1 in the current column
        col_ones = get_ones_index_col(index, col_list)
        print(f"col_ones : {col_ones}")
        # If the diagonal element is 0, it needs to be processed separately
        if col_list[index] == 0:
            # Vertices used to generate Steiner trees
            # col_ones.append(int(col_list[index]))
            v_st = col_ones + [index]
            v_st = sorted(v_st)
        else:
            v_st = col_ones
        # --------------------------------------------------------------------
        # Generate a Steiner tree based on the vertex set with value 1
        if len(v_st) > 1:
            tree_from_nx = approximation.steiner_tree(graph, v_st)
            # Get vertices in Steiner tree
            tmp_v = tree_from_nx.nodes
            print(f"tmp_v : {tmp_v}")
            if len(tmp_v) == 0:
                # Whether to perform column elimination
                col_flag = False
            if col_flag:
                # Get the set of vertices used to build the tree
                vertex = confirm_steiner_point(col_ones, tmp_v)
                # Get the set of edges used to build the tree
                edges = [e for e in tree_from_nx.edges]
                print(f"vertex : {vertex}")
                print(f"edges : {edges}")
                # Specify root node
                root_node = index
                # build tree
                tree = Tree(vertex, edges, root_node)
                root = tree.gen_tree()
                col = get_col(matrix, index)
                CNOT_list = []
                matrix, cnot = col_elim(matrix, root, col, CNOT_list)
                CNOT += cnot
                print(f"Matrix after column elimination : ")
                print(matrix)
                print(f"CNOT gate used in column elimination process: {cnot}")
                print("-" * 60)
        # row elimination
        ei = get_ei(matrix, index)
        print(f"ei : {ei}")
        # Get the currently eliminated row
        row_target = get_row(matrix, index)
        j_set = find_set_j(matrix, index + 1, row_target, ei)
        print(f"j_set : {j_set}")
        if j_set is not None:
            # Generate a Steiner tree based on j and i
            node_set = sorted([index] + j_set)
            print(f"node_set : {node_set}")
            tree_from_nx = approximation.steiner_tree(graph, node_set)
            # Get vertices in Steiner tree
            tmp_v = tree_from_nx.nodes
            print(f"tmp_v : {tmp_v}")
            # Get the set of vertices used to build the tree
            vertex = confirm_steiner_point(node_set, tmp_v)
            # Get the set of edges used to build the tree
            edges = [e for e in tree_from_nx.edges]
            print(f"vertex : {vertex}")
            print(f"edges : {edges}")
            tree = Tree(vertex, edges, index)
            root = tree.gen_tree()
            print(f"root.get_value() : {root.get_value()}")
            CNOT_list = []
            # Execute row elimination
            m, cnot = row_elim(matrix, root, CNOT_list)
            CNOT += cnot
            print(f"Matrix after row elimination : ")
            print(m)
            print(f"CNOT gate used in row elimination process: {cnot}")
            print("Delete current vertex")
        graph.remove_node(index)
        # Restore column elimination flag
        col_flag = True
    print(f"All CNOT doors: {CNOT}")
    # Convert CNOT according to mapping
    map_dict = {0: 0, 1: 4, 2: 3, 3: 1, 4: 2}
    # map_dict = {0: 0, 1: 2, 2: 1, 3: 3, 4: 4}
    # map_dict = {0: 2, 1: 0, 2: 1, 3: 3, 4: 4}
    # map_dict = {0: 2, 1: 4, 2: 3, 3: 1, 4: 0}
    # map_dict = {0: 4, 1: 3, 2: 0, 3: 1, 4: 2}
    # map_dict = {0: 4, 1: 3, 2: 2, 3: 1, 4: 0}
    new_CNOT = []
    for cnot in CNOT:
        control = map_dict.get(cnot[0])
        target = map_dict.get(cnot[1])
        new_CNOT.append((control, target))
    print(new_CNOT)
    return new_CNOT


def col_row_eli_of_ibmq_lagos(file_name):
    ibmq_lagos = IbmqLagos_new()
    graph = ibmq_lagos.get_graph()
    circuit_file = file_name
    matrix = get_circuits_to_matrix(circuit_file)
    print("matrix :")
    print(matrix)
    # eli_order = get_node_eli_order(graph.copy())
    CNOT = []
    # for index in range(rank):
    order = [0, 2, 1, 3, 4, 5, 6]
    # eli_order = [0, 4, 3, 1, 2]
    update_matrix(matrix, order)
    print(matrix)
    eli_order = [0, 1, 2, 3, 4, 5, 6]
    col_flag = True
    for index in eli_order:
        col_list = get_col(matrix, index)
        print(f"col_list{col_list}")
        col_ones = get_ones_index_col(index, col_list)
        print(f"col_ones : {col_ones}")
        if col_list[index] == 0:
            # col_ones.append(int(col_list[index]))
            v_st = col_ones + [index]
            v_st = sorted(v_st)
        else:
            v_st = col_ones
        # --------------------------------------------------------------------
        if len(v_st) > 1:
            tree_from_nx = approximation.steiner_tree(graph, v_st)
            tmp_v = tree_from_nx.nodes
            print(f"tmp_v : {tmp_v}")
            if len(tmp_v) == 0:
                col_flag = False
            if col_flag:
                vertex = confirm_steiner_point(col_ones, tmp_v)
                edges = [e for e in tree_from_nx.edges]
                print(f"vertex : {vertex}")
                print(f"edges : {edges}")
                root_node = index
                tree = Tree(vertex, edges, root_node)
                root = tree.gen_tree()
                col = get_col(matrix, index)
                CNOT_list = []
                matrix, cnot = col_elim(matrix, root, col, CNOT_list)
                CNOT += cnot
                print(matrix)
                print("-" * 60)
        ei = get_ei(matrix, index)
        print(f"ei : {ei}")
        row_target = get_row(matrix, index)
        j_set = find_set_j(matrix, index + 1, row_target, ei)
        print(f"j_set : {j_set}")
        if j_set is not None:
            node_set = sorted([index] + j_set)
            print(f"node_set : {node_set}")
            tree_from_nx = approximation.steiner_tree(graph, node_set)
            tmp_v = tree_from_nx.nodes
            print(f"tmp_v : {tmp_v}")
            vertex = confirm_steiner_point(node_set, tmp_v)
            edges = [e for e in tree_from_nx.edges]
            print(f"vertex : {vertex}")
            print(f"edges : {edges}")
            tree = Tree(vertex, edges, index)
            root = tree.gen_tree()
            print(f"root.get_value() : {root.get_value()}")
            CNOT_list = []
            m, cnot = row_elim(matrix, root, CNOT_list)
            CNOT += cnot
            print(m)
        graph.remove_node(index)
        col_flag = True
    map_dict = {0: 0, 1: 2, 2: 1, 3: 3, 4: 4, 5: 5, 6: 6}
    new_CNOT = []
    for cnot in CNOT:
        control = map_dict.get(cnot[0])
        target = map_dict.get(cnot[1])
        new_CNOT.append((control, target))
    print(new_CNOT)
    return new_CNOT


def col_row_eli_of_ibmq_guadalupe(file_name):
    ibmq_guadalupe = IbmqGuadalupe_new()
    graph = ibmq_guadalupe.get_graph()
    circuit_file = file_name
    # circuit_file = "./circuits/steiner/5qubits/10/Original11.qasm"  # 1, 4
    matrix = get_circuits_to_matrix(circuit_file)
    print("matrix :")
    print(matrix)
    # eli_order = get_node_eli_order(graph.copy())

    # print(f"eli_order : {eli_order}")
    CNOT = []
    # for index in range(rank):
    order = [0, 1, 2, 3, 4, 5, 6, 7, 9, 8, 10, 11, 14, 13, 12, 15]
    # eli_order = [0, 4, 3, 1, 2]
    update_matrix(matrix, order)
    print(matrix)
    eli_order = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
    col_flag = True
    for index in eli_order:
        col_list = get_col(matrix, index)
        print(f"col_list{col_list}")
        col_ones = get_ones_index_col(index, col_list)
        print(f"col_ones : {col_ones}")
        if col_list[index] == 0:
            # col_ones.append(int(col_list[index]))
            v_st = col_ones + [index]
            v_st = sorted(v_st)
        else:
            v_st = col_ones
        # --------------------------------------------------------------------
        if len(v_st) > 1:
            tree_from_nx = approximation.steiner_tree(graph, v_st)
            tmp_v = tree_from_nx.nodes
            print(f"tmp_v : {tmp_v}")
            if len(tmp_v) == 0:
                col_flag = False
            if col_flag:
                vertex = confirm_steiner_point(col_ones, tmp_v)
                edges = [e for e in tree_from_nx.edges]
                print(f"vertex : {vertex}")
                print(f"edges : {edges}")
                root_node = index
                tree = Tree(vertex, edges, root_node)
                root = tree.gen_tree()
                col = get_col(matrix, index)
                CNOT_list = []
                matrix, cnot = col_elim(matrix, root, col, CNOT_list)
                CNOT += cnot
                print(matrix)
                print("-" * 60)
        ei = get_ei(matrix, index)
        print(f"ei : {ei}")
        row_target = get_row(matrix, index)
        j_set = find_set_j(matrix, index + 1, row_target, ei)
        print(f"j_set : {j_set}")
        if j_set is not None:
            node_set = sorted([index] + j_set)
            print(f"node_set : {node_set}")
            tree_from_nx = approximation.steiner_tree(graph, node_set)
            tmp_v = tree_from_nx.nodes
            print(f"tmp_v : {tmp_v}")
            vertex = confirm_steiner_point(node_set, tmp_v)
            edges = [e for e in tree_from_nx.edges]
            print(f"vertex : {vertex}")
            print(f"edges : {edges}")
            tree = Tree(vertex, edges, index)
            root = tree.gen_tree()
            print(f"root.get_value() : {root.get_value()}")
            CNOT_list = []
            m, cnot = row_elim(matrix, root, CNOT_list)
            CNOT += cnot
            print(m)
        graph.remove_node(index)
        col_flag = True
    map_dict = {0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6, 7: 7, 8: 9, 9: 8, 10: 10, 11: 11, 12: 14, 13: 13, 14: 12, 15: 15}

    new_CNOT = []
    for cnot in CNOT:
        control = map_dict.get(cnot[0])
        target = map_dict.get(cnot[1])
        new_CNOT.append((control, target))
    print(new_CNOT)
    return new_CNOT


def col_row_eli_of_ibmq_almaden(file_name):
    ibmq_almaden = IbmqAlmaden_new()
    graph = ibmq_almaden.get_graph()
    circuit_file = file_name
    matrix = get_circuits_to_matrix(circuit_file)
    print("matrix :")
    print(matrix)
    # eli_order = get_node_eli_order(graph.copy())
    CNOT = []
    # for index in range(rank):
    order = [0, 1, 2, 4, 3, 5, 6, 7, 8, 9, 10, 11, 12, 14, 13, 15, 16, 17, 18, 19]
    # eli_order = [0, 4, 3, 1, 2]
    update_matrix(matrix, order)
    print(matrix)
    eli_order = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19]
    col_flag = True
    for index in eli_order:
        col_list = get_col(matrix, index)
        print(f"col_list{col_list}")
        col_ones = get_ones_index_col(index, col_list)
        print(f"col_ones : {col_ones}")
        if col_list[index] == 0:
            # col_ones.append(int(col_list[index]))
            v_st = col_ones + [index]
            v_st = sorted(v_st)
        else:
            v_st = col_ones
        # --------------------------------------------------------------------
        if len(v_st) > 1:
            tree_from_nx = approximation.steiner_tree(graph, v_st)
            tmp_v = tree_from_nx.nodes
            print(f"tmp_v : {tmp_v}")
            if len(tmp_v) == 0:
                col_flag = False
            if col_flag:
                vertex = confirm_steiner_point(col_ones, tmp_v)
                edges = [e for e in tree_from_nx.edges]
                print(f"vertex : {vertex}")
                print(f"edges : {edges}")
                root_node = index
                tree = Tree(vertex, edges, root_node)
                root = tree.gen_tree()
                col = get_col(matrix, index)
                CNOT_list = []
                matrix, cnot = col_elim(matrix, root, col, CNOT_list)
                CNOT += cnot
                print(matrix)
                print("-" * 60)
        ei = get_ei(matrix, index)
        print(f"ei : {ei}")
        row_target = get_row(matrix, index)
        j_set = find_set_j(matrix, index + 1, row_target, ei)
        print(f"j_set : {j_set}")
        if j_set is not None:
            node_set = sorted([index] + j_set)
            print(f"node_set : {node_set}")
            tree_from_nx = approximation.steiner_tree(graph, node_set)
            tmp_v = tree_from_nx.nodes
            print(f"tmp_v : {tmp_v}")
            vertex = confirm_steiner_point(node_set, tmp_v)
            edges = [e for e in tree_from_nx.edges]
            print(f"vertex : {vertex}")
            print(f"edges : {edges}")
            tree = Tree(vertex, edges, index)
            root = tree.gen_tree()
            print(f"root.get_value() : {root.get_value()}")
            CNOT_list = []
            m, cnot = row_elim(matrix, root, CNOT_list)
            CNOT += cnot
            print(m)
        graph.remove_node(index)
        col_flag = True
    map_dict = {0: 0, 1: 1, 2: 2, 3: 4, 4: 3, 5: 5, 6: 6, 7: 7, 8: 8, 9: 9, 10: 10, 11: 11, 12: 12, 13: 14, 14: 13, 15: 15, 16: 16, 17: 17, 18: 18, 19: 19}
    new_CNOT = []
    for cnot in CNOT:
        control = map_dict.get(cnot[0])
        target = map_dict.get(cnot[1])
        new_CNOT.append((control, target))
    print(new_CNOT)
    return new_CNOT


def col_row_eli_of_ibmq_tokyo(file_name):
    ibmq_tokyo = IbmqTokyo_new()
    graph = ibmq_tokyo.get_graph()
    circuit_file = file_name
    matrix = get_circuits_to_matrix(circuit_file)
    print("matrix :")
    print(matrix)
    # eli_order = get_node_eli_order(graph.copy())
    CNOT = []
    # for index in range(rank):
    order = [0, 1, 2, 3, 4, 9, 8, 7, 6, 5, 10, 11, 12, 14, 18, 14, 15, 16, 17, 19]
    # eli_order = [0, 4, 3, 1, 2]
    update_matrix(matrix, order)
    print(matrix)
    eli_order = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19]
    col_flag = True
    for index in eli_order:
        col_list = get_col(matrix, index)
        print(f"col_list{col_list}")
        col_ones = get_ones_index_col(index, col_list)
        print(f"col_ones : {col_ones}")
        if col_list[index] == 0:
            # col_ones.append(int(col_list[index]))
            v_st = col_ones + [index]
            v_st = sorted(v_st)
        else:
            v_st = col_ones
        # --------------------------------------------------------------------
        if len(v_st) > 1:
            tree_from_nx = approximation.steiner_tree(graph, v_st)
            tmp_v = tree_from_nx.nodes
            print(f"tmp_v : {tmp_v}")
            if len(tmp_v) == 0:
                col_flag = False
            if col_flag:
                vertex = confirm_steiner_point(col_ones, tmp_v)
                edges = [e for e in tree_from_nx.edges]
                print(f"vertex : {vertex}")
                print(f"edges : {edges}")
                root_node = index
                tree = Tree(vertex, edges, root_node)
                root = tree.gen_tree()
                col = get_col(matrix, index)
                CNOT_list = []
                matrix, cnot = col_elim(matrix, root, col, CNOT_list)
                CNOT += cnot
                print(matrix)
                print("-" * 60)
        ei = get_ei(matrix, index)
        print(f"ei : {ei}")
        row_target = get_row(matrix, index)
        j_set = find_set_j(matrix, index + 1, row_target, ei)
        print(f"j_set : {j_set}")
        if j_set is not None:
            node_set = sorted([index] + j_set)
            print(f"node_set : {node_set}")
            tree_from_nx = approximation.steiner_tree(graph, node_set)
            tmp_v = tree_from_nx.nodes
            print(f"tmp_v : {tmp_v}")
            vertex = confirm_steiner_point(node_set, tmp_v)
            edges = [e for e in tree_from_nx.edges]
            print(f"vertex : {vertex}")
            print(f"edges : {edges}")
            tree = Tree(vertex, edges, index)
            root = tree.gen_tree()
            print(f"root.get_value() : {root.get_value()}")
            CNOT_list = []
            m, cnot = row_elim(matrix, root, CNOT_list)
            CNOT += cnot
            print(m)
        graph.remove_node(index)
        col_flag = True
    map_dict = {0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 9, 6: 8, 7: 7, 8: 6, 9: 5, 10: 10, 11: 11, 12: 12, 13: 13, 14: 18, 15: 14, 16: 15, 17: 16, 18: 17, 19: 19}
    new_CNOT = []
    for cnot in CNOT:
        control = map_dict.get(cnot[0])
        target = map_dict.get(cnot[1])
        new_CNOT.append((control, target))
    print(new_CNOT)
    return new_CNOT


def col_row_eli_of_ibmq_kolkata(file_name):
    ibmq_kolkata = IbmqKolkata_new()
    graph = ibmq_kolkata.get_graph()
    circuit_file = file_name
    matrix = get_circuits_to_matrix(circuit_file)
    print("matrix :")
    print(matrix)
    print(type(matrix))
    # eli_order = get_node_eli_order(graph.copy())
    CNOT = []
    # for index in range(rank):
    order = [0, 1, 2, 3, 4, 5, 6, 7, 9, 8, 10, 11, 12, 13, 14, 15, 16, 17, 18, 20, 19, 21, 22, 23, 24, 25, 26]
    # eli_order = [0, 4, 3, 1, 2]
    update_matrix(matrix, order)
    print(matrix)
    eli_order = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26]
    col_flag = True
    for index in eli_order:
        col_list = get_col(matrix, index)
        print(f"col_list{col_list}")
        col_ones = get_ones_index_col(index, col_list)
        print(f"col_ones : {col_ones}")
        if col_list[index] == 0:
            # col_ones.append(int(col_list[index]))
            v_st = col_ones + [index]
            v_st = sorted(v_st)
        else:
            v_st = col_ones
        # --------------------------------------------------------------------
        if len(v_st) > 1:
            tree_from_nx = approximation.steiner_tree(graph, v_st)
            tmp_v = tree_from_nx.nodes
            print(f"tmp_v : {tmp_v}")
            if len(tmp_v) == 0:
                col_flag = False
            if col_flag:
                vertex = confirm_steiner_point(col_ones, tmp_v)
                print(f"vertex : {vertex}")
                print(f"edges : {edges}")
                root_node = index
                tree = Tree(vertex, edges, root_node)
                root = tree.gen_tree()
                col = get_col(matrix, index)
                CNOT_list = []
                matrix, cnot = col_elim(matrix, root, col, CNOT_list)
                CNOT += cnot
                print(matrix)
                print("-" * 60)
        ei = get_ei(matrix, index)
        print(f"ei : {ei}")
        row_target = get_row(matrix, index)
        j_set = find_set_j(matrix, index + 1, row_target, ei)
        print(f"j_set : {j_set}")
        if j_set is not None:
            node_set = sorted([index] + j_set)
            print(f"node_set : {node_set}")
            tree_from_nx = approximation.steiner_tree(graph, node_set)
            tmp_v = tree_from_nx.nodes
            print(f"tmp_v : {tmp_v}")
            vertex = confirm_steiner_point(node_set, tmp_v)
            edges = [e for e in tree_from_nx.edges]
            print(f"vertex : {vertex}")
            print(f"edges : {edges}")
            tree = Tree(vertex, edges, index)
            root = tree.gen_tree()
            print(f"root.get_value() : {root.get_value()}")
            CNOT_list = []
            m, cnot = row_elim(matrix, root, CNOT_list)
            CNOT += cnot
            print(m)
        graph.remove_node(index)
        col_flag = True
    map_dict = {0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6, 7: 7, 8: 9, 9: 8, 10: 10, 11: 11, 12: 12, 13: 13,
                14: 14, 15: 15, 16: 16, 17: 17, 18: 18, 19: 20, 20: 19, 21: 21, 22: 22, 23: 23, 24: 24, 25: 25, 26: 26}

    new_CNOT = []
    for cnot in CNOT:
        control = map_dict.get(cnot[0])
        target = map_dict.get(cnot[1])
        new_CNOT.append((control, target))
    print(new_CNOT)
    return new_CNOT


def col_row_eli_of_ibmq_manhattan(file_name):
    ibmq_manhattan = IbmqManhattan_new()
    graph = ibmq_manhattan.get_graph()
    circuit_file = file_name
    matrix = get_circuits_to_matrix(circuit_file)
    print("matrix :")
    print(matrix)

    CNOT = []
    order = [0, 1, 2, 3, 4, 5, 6, 7, 9, 8, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41,
             42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64]
    update_matrix(matrix, order)
    print(matrix)
    eli_order = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41,
                 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64]
    col_flag = True
    for index in eli_order:
        col_list = get_col(matrix, index)
        print(f"col_list{col_list}")
        col_ones = get_ones_index_col(index, col_list)
        print(f"col_ones : {col_ones}")
        if col_list[index] == 0:
            v_st = col_ones + [index]
            v_st = sorted(v_st)
        else:
            v_st = col_ones
        # --------------------------------------------------------------------
        if len(v_st) > 1:
            tree_from_nx = approximation.steiner_tree(graph, v_st)
            tmp_v = tree_from_nx.nodes
            print(f"tmp_v : {tmp_v}")
            if len(tmp_v) == 0:
                col_flag = False
            if col_flag:
                vertex = confirm_steiner_point(col_ones, tmp_v)
                edges = [e for e in tree_from_nx.edges]
                print(f"vertex : {vertex}")
                print(f"edges : {edges}")
                root_node = index
                tree = Tree(vertex, edges, root_node)
                root = tree.gen_tree()
                col = get_col(matrix, index)
                CNOT_list = []
                matrix, cnot = col_elim(matrix, root, col, CNOT_list)
                CNOT += cnot
                print(matrix)
                print("-" * 60)
        ei = get_ei(matrix, index)
        print(f"ei : {ei}")
        row_target = get_row(matrix, index)
        j_set = find_set_j(matrix, index + 1, row_target, ei)
        print(f"j_set : {j_set}")
        if j_set is not None:
            node_set = sorted([index] + j_set)
            print(f"node_set : {node_set}")
            tree_from_nx = approximation.steiner_tree(graph, node_set)
            tmp_v = tree_from_nx.nodes
            print(f"tmp_v : {tmp_v}")
            vertex = confirm_steiner_point(node_set, tmp_v)
            edges = [e for e in tree_from_nx.edges]
            print(f"vertex : {vertex}")
            print(f"edges : {edges}")
            tree = Tree(vertex, edges, index)
            root = tree.gen_tree()
            print(f"root.get_value() : {root.get_value()}")
            CNOT_list = []
            m, cnot = row_elim(matrix, root, CNOT_list)
            CNOT += cnot
            print(m)
        graph.remove_node(index)
        col_flag = True
    map_dict = {0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6, 7: 7, 8: 9, 9: 8, 10: 10, 11: 11, 12: 12, 13: 13, 14: 14, 15: 15, 16: 16, 17: 17, 18: 18, 19: 19, 20: 20, 21: 21, 22: 22,
                23: 23, 24: 24, 25: 25, 26: 26, 27: 27, 28: 28, 29: 29, 30: 30, 31: 31, 32: 32, 33: 33, 34: 34, 35: 35, 36: 36, 37: 37, 38: 38, 39: 39, 40: 40, 41: 41, 42: 42,
                43: 43, 44: 44, 45: 45, 46: 46, 47: 47, 48: 48, 49: 49, 50: 50, 51: 51, 52: 52, 53: 53, 54: 54, 55: 55, 56: 56, 57: 57, 58: 58, 59: 59, 60: 60, 61: 61, 62: 62,
                63: 63, 64: 64}
    new_CNOT = []
    for cnot in CNOT:
        control = map_dict.get(cnot[0])
        target = map_dict.get(cnot[1])
        new_CNOT.append((control, target))
    print(new_CNOT)
    return new_CNOT


def col_row_eli_of_wukong(file_name):
    wukong = WuKong_new()
    graph = wukong.get_graph()
    circuit_file = file_name
    matrix = get_circuits_to_matrix(circuit_file)
    print("matrix :")
    print(matrix)
    CNOT = []
    order = [0, 1, 2, 3, 4, 5, 6, 7, 8, 10, 9, 11]
    update_matrix(matrix, order)
    print(matrix)
    eli_order = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
    col_flag = True
    for index in eli_order:
        # column elimination
        col_list = get_col(matrix, index)
        print(f"col_list{col_list}")
        col_ones = get_ones_index_col(index, col_list)
        print(f"col_ones : {col_ones}")
        if col_list[index] == 0:
            # col_ones.append(int(col_list[index]))
            v_st = col_ones + [index]
            v_st = sorted(v_st)
        else:
            v_st = col_ones
        # --------------------------------------------------------------------
        if len(v_st) > 1:
            tree_from_nx = approximation.steiner_tree(graph, v_st)
            tmp_v = tree_from_nx.nodes
            print(f"tmp_v : {tmp_v}")
            if len(tmp_v) == 0:
                col_flag = False
            if col_flag:
                vertex = confirm_steiner_point(col_ones, tmp_v)
                edges = [e for e in tree_from_nx.edges]
                print(f"vertex : {vertex}")
                print(f"edges : {edges}")
                root_node = index
                tree = Tree(vertex, edges, root_node)
                root = tree.gen_tree()
                col = get_col(matrix, index)
                CNOT_list = []
                matrix, cnot = col_elim(matrix, root, col, CNOT_list)
                CNOT += cnot
                print(matrix)
                print("-" * 60)
        # row elimination
        ei = get_ei(matrix, index)
        print(f"ei : {ei}")
        row_target = get_row(matrix, index)
        j_set = find_set_j(matrix, index + 1, row_target, ei)
        print(f"j_set : {j_set}")
        if j_set is not None:
            node_set = sorted([index] + j_set)
            print(f"node_set : {node_set}")
            tree_from_nx = approximation.steiner_tree(graph, node_set)
            tmp_v = tree_from_nx.nodes
            print(f"tmp_v : {tmp_v}")
            vertex = confirm_steiner_point(node_set, tmp_v)
            edges = [e for e in tree_from_nx.edges]
            print(f"vertex : {vertex}")
            print(f"edges : {edges}")
            tree = Tree(vertex, edges, index)
            root = tree.gen_tree()
            print(f"root.get_value() : {root.get_value()}")
            CNOT_list = []
            m, cnot = row_elim(matrix, root, CNOT_list)
            CNOT += cnot
            print(m)
        graph.remove_node(index)
        col_flag = True
    map_dict = {0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6, 7: 7, 8: 8, 9: 10, 10: 9, 11: 11}
    new_CNOT = []
    for cnot in CNOT:
        control = map_dict.get(cnot[0])
        target = map_dict.get(cnot[1])
        new_CNOT.append((control, target))
    print(new_CNOT)
    return new_CNOT


def test_gen_circuit_old(qubits, file):
    cnot = col_row_eli_of_ibmq_guadalupe(file)
    circuit = QuantumCircuit(qubits)
    for cnot_gate in cnot:
        control = cnot_gate[0]
        target = cnot_gate[1]
        circuit.cx(control, target)
    circuit.measure_all()
    circuit.draw("mpl")
    print(circuit)
    circuit.qasm(filename=f"add-exam/result-circuits/16qubits/hwb_12.qasm")




def test_gen_circuit_new(qubits, cnots, file_name):
    circuit = QuantumCircuit(qubits)
    for cnot_gate in cnots:
        control = cnot_gate[0]
        target = cnot_gate[1]
        circuit.cx(control, target)
    circuit.measure_all()
    circuit.draw("mpl")
    print(circuit)
    circuit.qasm(filename=f"add-exam/result-circuits/B&D_circuits_synthesize/{file_name}-{qubits}qubits_synthesis.qasm")




def test_read_cir():
    circuit = QuantumCircuit(16)
    circuit = circuit.from_qasm_file("./circuits/benchmark/16/cnt3-5_179.qasm")
    print(circuit)


def execute_circuit():
    gates_list = [2, 4, 5, 8, 10, 15, 20, 30, 40, 80, 100, 200]
    for gate in gates_list:
        for i in range(20):
            file_name = f"./circuits/steiner/5qubits/{gate}/Original{i}.qasm"
            origin_cnot_list = col_row_eli_of_ibmquatio(file_name)
            circuit = QuantumCircuit(5)
            for cnot_gate in origin_cnot_list:
                control = cnot_gate[0]
                target = cnot_gate[1]
                circuit.cx(control, target)
            circuit.measure_all()
            circuit.draw("mpl")
            print(circuit)
            circuit.qasm(filename=f"./result/01234-02134/{gate}/circuit{i}.qasm")


def execute_benchmark():
    file_list = ['4gt5_75', '4gt13_90', '4gt13_91', '4gt13_92', '4mod5-v1_22', '4mod5-v1_23', '4mod5-v1_24', 'alu-v0_27', 'alu-v3_35', 'alu-v4_36', 'alu-v4_37', 'decod24-v2_43',
                 'hwb4_49', 'mod5mils_65', 'mod10_171']
    for file in file_list:
        file_name = f"./circuits/benchmark/5qubits/qasm/{file}.qasm"
        origin_cnot_list = col_row_eli_of_ibmquatio(file_name)
        circuit = QuantumCircuit(5)
        for cnot_gate in origin_cnot_list:
            control = cnot_gate[0]
            target = cnot_gate[1]
            circuit.cx(control, target)
        circuit.measure_all()
        circuit.draw("mpl")
        print(circuit)
        circuit.qasm(filename=f"./result/01234-43210/qasm-trans/{file}_eli.qasm")


def update_matrix(matrix, order):
    """
   Update matrix based on elimination path
    :param order:
    :return:
    """
    matrix_rank = matrix.rank()
    print(matrix_rank)
    new_matrix = Mat2.id(matrix_rank)
    print(new_matrix)
    print(new_matrix.data[1][1])
    for i in range(matrix_rank):
        for j in range(matrix_rank):
            new_matrix.data[i][j] = matrix.data[order[i]][order[j]]
    print(new_matrix)
    return new_matrix


if __name__ == '__main__':
    execute_benchmark()

    col_row_eli_of_ibmq_guadalupe("./circuits/benchmark/15_and_16_qubits_test/16qubit_circuit/cnt3-5_179.qasm")
    circuit_file = "./circuits/benchmark/15_and_16_qubits_test/16qubit_circuit/cnt3-5_179.qasm"
    circuit_file = "./circuits/steiner/5qubits/10/Original11.qasm"  # 1, 4
    matrix = get_circuits_to_matrix(circuit_file)
    order = [0, 1, 2, 3, 4, 5, 6, 7, 9, 8, 10, 11, 14, 13, 12, 15]
    update_matrix(matrix, order)


    cir = "Bernstein-Vazirani"
    qubits = 27
    start_time = time.time()
    cnots = col_row_eli_of_ibmq_kolkata(f'./circuits/benchmark/B&D/B&D_circuits/{cir}-{qubits}qubits-delete-singlegate.qasm')
    test_gen_circuit_new(qubits, cnots, cir)
    end_time = time.time()
    execution_time = end_time - start_time
    print("Execution Time:", execution_time, "seconds")
