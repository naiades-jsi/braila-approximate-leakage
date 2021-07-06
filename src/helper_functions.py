import plotly

import src.configfile
from src.epanet.EPANETUtils import EPANETUtils


def pretty_print(node_arr):
    string_to_print = ""

    for index, node in enumerate(node_arr):
        stripped_node = node.replace(", 16.0LPS", "")
        string_to_print += "  " + stripped_node + ",\n"

    return string_to_print


def visualize_node_groups(critical_node_name, node_groups_dict, epanet_file_path, node_size=8, link_width=1,
                          figsize=[1920, 1080], round_ndigits=2, add_to_node_popup=None, filename='plotly_network.html',
                          auto_open=True):
    water_network_model = EPANETUtils(epanet_file_path, "PDD").get_original_water_network_model()
    network_graph = water_network_model.get_graph()    # Graph
    colors = ["#9FE2BF", "#6495ED", "#FF7F50", "#DE3163"]

    # create new dict without lps
    node_groups_dict_without_lps = dict()
    group_colors = dict()
    for index, group_name in enumerate(node_groups_dict):
        group_colors[group_name] = colors[index]
        node_groups_dict_without_lps[group_name] = dict()

        group = node_groups_dict[group_name]
        for node_name in group:
            new_name = node_name.replace(", 16.0LPS", "")
            new_name = new_name.replace("Node_", "")
            node_groups_dict_without_lps[group_name][new_name] = node_groups_dict[group_name][node_name]

    # Create edge trace
    edge_trace = plotly.graph_objs.Scatter(
        x=[],
        y=[],
        text=[],
        hoverinfo='text',
        mode='lines',
        line=dict(
            color='#888',
            width=link_width)
    )

    for edge in network_graph.edges():
        x0, y0 = network_graph.nodes[edge[0]]['pos']
        x1, y1 = network_graph.nodes[edge[1]]['pos']
        edge_trace['x'] += tuple([x0, x1, None])
        edge_trace['y'] += tuple([y0, y1, None])

    # Create node traces
    data = [edge_trace]
    for group in node_groups_dict_without_lps:
        node_trace = plotly.graph_objs.Scatter(
            x=[],
            y=[],
            text=[],
            name=group,
            hoverinfo='text',
            mode='markers',
            marker=dict(
                color=group_colors[group],
                size=node_size,
                line=dict(width=1),
                symbol=[],
            ))

        for node in network_graph.nodes():
            if str(node) in node_groups_dict_without_lps[group]:
                x, y = network_graph.nodes[node]['pos']
                node_trace['x'] += tuple([x])
                node_trace['y'] += tuple([y])

                # Assigning symbols
                if network_graph.nodes[node]["type"] == "Reservoir":
                    node_trace['marker']['symbol'] += tuple(["square"])
                elif "Senzor" in str(node):
                    node_trace['marker']['symbol'] += tuple(["arrow-right"])
                elif "Jonctiune" in str(node) and "Junction" not in str(node):
                    node_trace['marker']['symbol'] += tuple(["circle"])
                else:
                    node_trace['marker']['symbol'] += tuple(["bowtie"])
                node_info = water_network_model.get_node(node).node_type + ': ' + str(node) + '<br>'

                if add_to_node_popup is not None:
                    if node in add_to_node_popup.index:
                        for key, val in add_to_node_popup.loc[node].iteritems():
                            node_info = node_info + '<br>' + \
                                        key + ': ' + '{:.{prec}f}'.format(val, prec=round_ndigits)

                node_trace['text'] += tuple([node_info])
        data.append(node_trace)

    # Create figure
    layout = plotly.graph_objs.Layout(
        title=critical_node_name + " affected groups",
        titlefont=dict(size=16),
        showlegend=True,
        width=figsize[0],
        height=figsize[1],
        hovermode='closest',
        margin=dict(b=20, l=5, r=5, t=40),
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False))

    fig = plotly.graph_objs.Figure(data=data, layout=layout)
    if filename:
        plotly.offline.plot(fig, filename=filename, auto_open=auto_open)
    else:
        plotly.offline.plot(fig, auto_open=auto_open)
