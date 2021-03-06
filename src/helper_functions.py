import plotly

import src.configfile
import matplotlib.pyplot as plt
from src.epanet.EPANETUtils import EPANETUtils


def pretty_print(node_arr):
    string_to_print = ""

    for _, node in enumerate(node_arr):
        stripped_node = node.replace(", 16.0LPS", "")
        string_to_print += "  " + stripped_node + ",\n"

    return string_to_print


def visualize_node_groups(critical_node_name, node_groups_dict, epanet_file_path, leak_amount, node_size=8,
                          link_width=1, figsize=[950, 950], round_ndigits=2, add_to_node_popup=None,
                          filename='plotly_network.html',
                          auto_open=False):
    water_network_model = EPANETUtils(epanet_file_path, "PDD").get_original_water_network_model()
    network_graph = water_network_model.get_graph()    # Graph

    if len(node_groups_dict) > 10:
        raise Exception("Visualization is currently supported for only 10 groups !")

    # TODO make colors better when there are less groups. change when using more than 5 groups
    # colors_arr = ["#52f247", "#84e100", "#a4cf00", "#bdbb00", "#d0a600",
    #               "#e09000", "#ea7800", "#f15e00", "#f24123", "#ef1b3a"]
    colors_arr = ["#DE3163", "#FF7F50", "#6495ED", "#9FE2BF"]

    # create new dict without lps
    node_groups_dict_without_lps = node_groups_dict.copy()
    group_colors = dict()
    for index, group_name in enumerate(node_groups_dict):
        group_colors[group_name] = colors_arr[index]

        group = node_groups_dict[group_name]
        for arr_index, node_name in enumerate(group):
            new_name = node_name.replace("Node_", "")
            node_groups_dict_without_lps[group_name][arr_index] = new_name

    # Create edge trace
    edge_trace = plotly.graph_objs.Scatter(
        x=[],
        y=[],
        text=[],
        name="Pipes",
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
            name=f"Group {group}",
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
            # TODO implement an else that makes sense - since nodes can be in one group not two so you have duplicates
        data.append(node_trace)

    # Create figure
    layout = plotly.graph_objs.Layout(
        title=critical_node_name + " affected groups. Leak amount " + str(leak_amount),
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
