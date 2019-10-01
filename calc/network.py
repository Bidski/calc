# TODO: consider changing w -> w_k and width -> w_map
import networkx as nx
import numpy as np


class Layer:
    def __init__(self, name, m, width):
        """
        :param name:
        :param m: number of feature maps
        :param width: layer width in pixels
        """
        self.name = name
        self.m = m
        self.width = width


class Connection:
    def __init__(self, pre, post, c, s, w, sigma):
        """
        :param pre: input layer
        :param post: output layer
        :param c: fraction of presynaptic feature maps that contribute to connection
        :param s: stride
        :param w: width of square kernel
        :param sigma: pixel-wise connection sparsity
        """
        self.pre = pre
        self.post = post
        self.c = c
        self.s = s
        self.w = w
        self.sigma = sigma

    def get_name(self):
        return "{}->{}".format(self.pre.name, self.post.name)


class Network:
    def __init__(self):
        self.layers = []
        self.connections = []

    def add(self, name, m, width):
        result = Layer(name, m, width)
        self.layers.append(result)
        return result

    def connect(self, pre, post, c, s, w, sigma):
        if isinstance(pre, str):
            pre = self.find_layer(pre)
        if isinstance(post, str):
            post = self.find_layer(post)
        result = Connection(pre, post, c, s, w, sigma)
        self.connections.append(result)
        return result

    def find_layer(self, name):
        result = None
        for layer in self.layers:
            if layer.name == name:
                result = layer
                break
        return result

    def find_layer_index(self, name):
        result = None
        for i in range(len(self.layers)):
            if self.layers[i].name == name:
                result = i
                break
        return result

    def find_connection_index(self, pre_name, post_name):
        result = None
        for i in range(len(self.connections)):
            if self.connections[i].pre.name == pre_name and self.connections[i].post.name == post_name:
                result = i
                break
        return result

    def print(self):
        for layer in self.layers:
            print("{} (m={:2.2f} width={:2.2f})".format(layer.name, layer.m, layer.width))

        for conn in self.connections:
            print(
                "{} -> {} (c={:8.6f} s={:2.2f} w={:2.2f} sigma={:8.6f})".format(
                    conn.pre.name, conn.post.name, conn.c, conn.s, conn.w, conn.sigma
                )
            )

    def find_inbounds(self, layer_name):
        """
        :param layer_name: Name of a layer in the network
        :return: List of connections that provide input to the layer
        """
        result = []
        for connection in self.connections:
            if connection.post.name == layer_name:
                result.append(connection)
        return result

    def find_outbounds(self, layer_name):
        """
        :param layer_name: Name of a layer in the network
        :return: List of connections out of the layer
        """
        result = []
        for connection in self.connections:
            if connection.pre.name == layer_name:
                result.append(connection)
        return result

    def remove_layer(self, name):
        for connection in self.find_inbounds(name):
            self.remove_connection(connection.pre.name, connection.post.name)

        for connection in self.find_outbounds(name):
            self.remove_connection(connection.pre.name, connection.post.name)

        index = self.find_layer_index(name)
        del self.layers[index]

    def remove_connection(self, pre_name, post_name):
        index = self.find_connection_index(pre_name, post_name)
        del self.connections[index]

    def make_graph(self):
        graph = nx.DiGraph()

        for layer in self.layers:
            graph.add_node(layer.name)

        for connection in self.connections:
            graph.add_edge(connection.pre.name, connection.post.name)

        return graph

    def write_dot_graph(self, output_file, use_rank=False):
        colours = {
            "AITd_2_3": {"hex": "#aaa4e1", "count": 1},
            "AITd_4": {"hex": "#141d1b", "count": 9},
            "AITd_5": {"hex": "#a6e590", "count": 2},
            "AITd_6": {"hex": "#340a29", "count": 2},
            "AITv_2_3": {"hex": "#1cf1a3", "count": 1},
            "AITv_4": {"hex": "#1e0e76", "count": 13},
            "AITv_5": {"hex": "#a9e81a", "count": 2},
            "AITv_6": {"hex": "#6439e5", "count": 2},
            "CITd_2_3": {"hex": "#6e9f23", "count": 1},
            "CITd_4": {"hex": "#ef6ade", "count": 3},
            "CITd_5": {"hex": "#20f53d", "count": 2},
            "CITd_6": {"hex": "#fe16f4", "count": 2},
            "CITv_2_3": {"hex": "#02531d", "count": 1},
            "CITv_4": {"hex": "#f03c6d", "count": 9},
            "CITv_5": {"hex": "#65e6f9", "count": 2},
            "CITv_6": {"hex": "#8f323c", "count": 2},
            "INPUT": {"hex": "#000000", "count": 1},
            "konio_LGN": {"hex": "#459da1", "count": 1},
            "magno_LGN": {"hex": "#d87378", "count": 1},
            "parvo_LGN": {"hex": "#10558a", "count": 1},
            "PITd_2_3": {"hex": "#edd45e", "count": 1},
            "PITd_4": {"hex": "#5252b9", "count": 7},
            "PITd_5": {"hex": "#e6861f", "count": 2},
            "PITd_6": {"hex": "#a32890", "count": 2},
            "PITv_2_3": {"hex": "#f2cdb9", "count": 1},
            "PITv_4": {"hex": "#6b4c33", "count": 7},
            "PITv_5": {"hex": "#ff2a0d", "count": 2},
            "PITv_6": {"hex": "#ab8a77", "count": 2},
            "V1_2_3blob": {"hex": "#aaa4e1", "count": 3},
            "V1_2_3interblob": {"hex": "#141d1b", "count": 2},
            "V1_4B": {"hex": "#a6e590", "count": 1},
            "V1_4Calpha": {"hex": "#340a29", "count": 1},
            "V1_4Cbeta": {"hex": "#1cf1a3", "count": 1},
            "V1_5": {"hex": "#1e0e76", "count": 3},
            "V1_6": {"hex": "#a9e81a", "count": 3},
            "V2pale_2_3": {"hex": "#6439e5", "count": 1},
            "V2pale_4": {"hex": "#6e9f23", "count": 1},
            "V2pale_5": {"hex": "#ef6ade", "count": 2},
            "V2pale_6": {"hex": "#20f53d", "count": 2},
            "V2thick_2_3": {"hex": "#20f53d", "count": 1},
            "V2thick_4": {"hex": "#fe16f4", "count": 1},
            "V2thick_5": {"hex": "#02531d", "count": 2},
            "V2thick_6": {"hex": "#f03c6d", "count": 2},
            "V2thin_2_3": {"hex": "#65e6f9", "count": 1},
            "V2thin_4": {"hex": "#8f323c", "count": 2},
            "V2thin_5": {"hex": "#459da1", "count": 2},
            "V2thin_6": {"hex": "#d87378", "count": 2},
            "V4_2_3": {"hex": "#10558a", "count": 1},
            "V4_4": {"hex": "#edd45e", "count": 6},
            "V4_5": {"hex": "#5252b9", "count": 2},
            "V4_6": {"hex": "#e6861f", "count": 2},
            "VOT_2_3": {"hex": "#a32890", "count": 1},
            "VOT_4": {"hex": "#f2cdb9", "count": 6},
            "VOT_5": {"hex": "#ab8a77", "count": 2},
            "VOT_6": {"hex": "#aaa4e1", "count": 2},
        }

        with open(output_file, "w") as dot:

            dot.write("digraph MSH {\n")
            dot.write("    splines=true;\n")
            dot.write("    rankdir=LR;\n")

            dot.write("    # Node definitions\n")
            for layer in self.layers:
                layer_label = "{0} | {{ {{ m | w }} | {{ {1:.4e} | {2:.4e} }} }}".format(
                    layer.name.replace("/", "_"), layer.m, layer.width
                )
                dot.write(
                    '    {} [shape=Mrecord, style=bold, label="{}", color="{}", penwidth={}];\n'.format(
                        layer.name.replace("/", "_"),
                        layer_label,
                        colours[layer.name.replace("/", "_")]["hex"],
                        colours[layer.name.replace("/", "_")]["count"],
                    )
                )

            dot.write("\n")

            dot.write("    # Edge definitions\n")
            for conn in self.connections:
                edge_label = "{}_to_{} | {{ {{ c | s | w | {} }} | {{ {:.4e} | {:.4e} | {:.4e} | {:.4e} }} }}".format(
                    conn.pre.name.replace("/", "_"),
                    conn.post.name.replace("/", "_"),
                    "\u03c3",
                    conn.c,
                    conn.s,
                    conn.w,
                    conn.sigma,
                )
                dot.write(
                    '    {}_to_{} [shape=record, style=radial, label="{}"];\n'.format(
                        conn.pre.name.replace("/", "_"), conn.post.name.replace("/", "_"), edge_label
                    )
                )

            dot.write("    # Edges\n")
            for conn in self.connections:
                dot.write(
                    '    {0}:e -> {0}_to_{1} -> {1}:w [color="{2}", penwidth={3}];\n'.format(
                        conn.pre.name.replace("/", "_"),
                        conn.post.name.replace("/", "_"),
                        colours[conn.post.name.replace("/", "_")]["hex"],
                        colours[conn.post.name.replace("/", "_")]["count"],
                    )
                )

            if use_rank:
                dot.write("    # Rank groupings\n")
                groups = ["LGN", "V1_2", "V1_4", "V1_5", "V1_6", "V2thin", "V2thick", "V4", "PIT", "CIT", "VOT"]
                for group in groups:
                    rank = "; ".join([layer.name.replace("/", "_") for layer in self.layers if group in layer.name])
                    dot.write("    {{rank = same; {};}}\n".format(rank))

            dot.write("}\n")

    def prune_dead_ends(self, output_layers):
        graph = self.make_graph()

        # keep = []
        remove = []
        for layer in self.layers:
            path_exists_to_output = False
            for output in output_layers:
                if nx.has_path(graph, layer.name, output):
                    path_exists_to_output = True
                    break
            # keep.append(path_exists_to_output)
            if not path_exists_to_output:
                remove.append(layer.name)
                print("Pruning {}".format(layer.name))

        removed_indices = []
        for layer_name in remove:
            removed_indices.append(self.find_layer_index(layer_name))

        for layer_name in remove:
            self.remove_layer(layer_name)

        return removed_indices

    def scale_c(self, factor):
        """
        Scales all c parameters in log space.
        :param factor factor to scale by
        """
        for connection in self.connections:
            connection.c = np.exp(np.log(connection.c) * factor)

    def scale_sigma(self, factor):
        """
        Scales all sigma parameters in log space.
        :param factor factor to scale by
        """
        for connection in self.connections:
            connection.sigma = np.exp(np.log(connection.sigma) * factor)
