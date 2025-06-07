import torch


def grid_to_graph(height, width):
    """
    Converts a 2D grid into graph format for GCN.
    Nodes are pixels, and edges are connections between adjacent pixels.
    """
    edge_list = []
    for i in range(height):
        for j in range(width):
            node = i * width + j
            if i + 1 < height:  # Down
                edge_list.append([node, (i + 1) * width + j])
                edge_list.append([(i + 1) * width + j, node])  # Bidirectional
            if j + 1 < width:  # Right
                edge_list.append([node, i * width + j + 1])
                edge_list.append([i * width + j + 1, node])  # Bidirectional

    edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
    return edge_index