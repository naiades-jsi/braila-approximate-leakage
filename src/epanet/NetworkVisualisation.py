import plotly
import wntr
import pandas as pd
import matplotlib
import networkx as nx


def interactive_visualization(self, water_network_model=None, node_attribute_name='Value', title=None, node_size=8, node_labels=True,
                            link_width=1, figsize=[700, 450], round_ndigits=2, add_to_node_popup=None,
                            filename='plotly_network.html', auto_open=True):
    """
    TODO
    :param water_network_model:
    :param node_attribute_name:
    :param title:
    :param node_size:
    :param node_labels:
    :param link_width:
    :param figsize:
    :param round_ndigits:
    :param add_to_node_popup:
    :param filename:
    :param auto_open:
    :return:
    """
    # TODO import your svg icons, integrate it with pressure/flow - aka take the already made code
    # TODO junction are the one with no base demand usage, nodes aka circles are the ones with usage
    # TODO more viz options, pressure wiz, flowrate viz, pressure + flowrate ?
    if water_network_model is None:
        # if no water model is provided the default unmodified instance will be returned
        water_network_model = self.get_original_water_network_model()

    network_graph = water_network_model.get_graph()
    # Create edge trace - connections

    edge_trace = plotly.graph_objs.Scatter(
        name="Pipe",
        x=[],
        y=[],
        text=[],
        hoverinfo='text',
        mode='lines',
        # TODO change line shape ? line_shape='hv',
        line=dict(
            color='#888',
            width=link_width
        )
    )

    # Node trace for normal Junctions
    node_junction = plotly.graph_objs.Scatter(
        name="Junction",
        x=[],
        y=[],
        text=[],
        hoverinfo='text',
        mode='markers',
        marker=dict(
            color="#FCA103",
            size=node_size,
            opacity=1,
            line=dict(width=1),
            symbol='bowtie'))
    # Node trace for terminal junctions (only one connection)
    node_terminal_junction = plotly.graph_objs.Scatter(
        name="Terminal Junction",
        x=[],
        y=[],
        text=[],
        hoverinfo='text',
        mode='markers',
        marker=dict(
            color="#FC4E03",
            size=5,
            opacity=1,
            line=dict(width=1),
            symbol='circle'
        ))
    # Node trace for tanks
    node_tank = plotly.graph_objs.Scatter(
        name="Tank",
        x=[],
        y=[],
        text=[],
        hoverinfo='text',
        mode='markers',
        marker=dict(
            color="#33B0FF",
            size=node_size + 10,
            opacity=1,
            line=dict(width=1),
            symbol='square'
        ))

    # Node trace for sensors
    node_sensor = plotly.graph_objs.Scatter(
        name="Sensor",
        x=[],
        y=[],
        text=[],
        hoverinfo='text',
        mode='markers',
        marker=dict(
            color="#41BE51",
            size=node_size,
            opacity=1,
            line=dict(width=1),
            symbol='arrow-right'
        )
    )

    for edge in network_graph.edges():
        x0, y0 = network_graph.nodes[edge[0]]['pos']
        x1, y1 = network_graph.nodes[edge[1]]['pos']
        edge_trace['x'] += tuple([x0, x1, None])
        edge_trace['y'] += tuple([y0, y1, None])

    terminal_nodes = [node[0] for node in network_graph.degree() if node[1] == 1]
    for node in network_graph.nodes():
        curr_node = None
        x, y = network_graph.nodes[node]['pos']

        # G.nodes[node] has a property "type", this can be Junction, Reservoir
        if network_graph.nodes[node]["type"] == "Reservoir":
            curr_node = node_tank
        elif node in terminal_nodes:
            curr_node = node_terminal_junction
        elif "Senzor" in node:
            curr_node = node_sensor
        else:
            curr_node = node_junction

        curr_node['x'] += tuple([x])
        curr_node['y'] += tuple([y])
        # Add node labels
        if node_labels:
            node_info = water_network_model.get_node(node).node_type + ': ' + str(node) + '<br>' + \
                        node_attribute_name + ': ' + "No data"
            if add_to_node_popup is not None:
                if node in add_to_node_popup.index:
                    for key, val in add_to_node_popup.loc[node].iteritems():
                        node_info = node_info + '<br>' + \
                                    key + ': ' + '{:.{prec}f}'.format(val, prec=round_ndigits)

            curr_node['text'] += tuple([node_info])

    # Create figure
    data = [edge_trace, node_terminal_junction, node_junction, node_tank, node_sensor]
    layout = plotly.graph_objs.Layout(
        title=title,
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


def interactive_visualization_24_hour_simulation():
    # TODO implement simulation in which values change in a 24 hour cycle
    pass


def visualization_for_pressure_with_leaks():
    pass


def _format_node_attribute(node_attribute, wn):

    if isinstance(node_attribute, str):
        node_attribute = wn.query_node_attribute(node_attribute)
    if isinstance(node_attribute, list):
        node_attribute = dict(zip(node_attribute, [1] * len(node_attribute)))
    if isinstance(node_attribute, pd.Series):
        node_attribute = dict(node_attribute)

    return node_attribute


def plot_interactive_network(water_network_model, node_attribute=None, node_attribute_name='Value', title=None,
                             node_size=8, node_range=[None, None], node_cmap='jet', node_labels=True,
                             link_width=1, add_colorbar=True, reverse_colormap=False,
                             figsize=[700, 450], round_ndigits=2, add_to_node_popup=None,
                             filename='plotly_network.html', auto_open=True):
    if plotly is None:
        raise ImportError('plotly is required')

    # Graph
    network_graph = water_network_model.get_graph()

    # Node attribute
    if node_attribute is not None:
        if isinstance(node_attribute, list):
            node_cmap = 'Reds'
            add_colorbar = False
        node_attribute = _format_node_attribute(node_attribute, water_network_model)
    else:
        add_colorbar = False

    # Create edge trace
    edge_trace = plotly.graph_objs.Scatter(
        x=[],
        y=[],
        text=[],
        hoverinfo='text',
        mode='lines',
        line=dict(
            # colorscale=link_cmap,
            # reversescale=reverse_colormap,
            color='#888',  # [],
            width=link_width)
    )

    for edge in network_graph.edges():
        x0, y0 = network_graph.nodes[edge[0]]['pos']
        x1, y1 = network_graph.nodes[edge[1]]['pos']
        edge_trace['x'] += tuple([x0, x1, None])
        edge_trace['y'] += tuple([y0, y1, None])

    # Create node trace
    node_trace = plotly.graph_objs.Scatter(
        x=[],
        y=[],
        text=[],
        hoverinfo='text',
        mode='markers',
        marker=dict(
            showscale=add_colorbar,
            colorscale=node_cmap,
            cmin=node_range[0],  # TODO: Not sure this works
            cmax=node_range[1],  # TODO: Not sure this works
            reversescale=reverse_colormap,
            color=[],
            size=node_size,
            # opacity=0.75,
            colorbar=dict(
                thickness=15,
                xanchor='left',
                titleside='right'),
            line=dict(width=1),
            symbol=[],
        ))
    for node in network_graph.nodes():
        x, y = network_graph.nodes[node]['pos']
        node_trace['x'] += tuple([x])
        node_trace['y'] += tuple([y])

        # G.nodes[node] has a property "type", this can be Junction, Reservoir
        if network_graph.nodes[node]["type"] == "Reservoir":
            node_trace['marker']['symbol'] += tuple(["square"])
        elif "Senzor" in str(node):
            node_trace['marker']['symbol'] += tuple(["arrow-right"])
        elif "Jonctiune" in str(node) and "Junction" not in str(node):
            node_trace['marker']['symbol'] += tuple(["circle"])
        else:
            node_trace['marker']['symbol'] += tuple(["bowtie"])

        try:
            # Add node attributes
            node_trace['marker']['color'] += tuple([node_attribute[node]])
            # node_trace['marker']['size'].append(node_size)

            # Add node labels
            if node_labels:
                node_info = water_network_model.get_node(node).node_type + ': ' + str(node) + '<br>' + \
                            node_attribute_name + ': ' + str(round(node_attribute[node], round_ndigits))
                if add_to_node_popup is not None:
                    if node in add_to_node_popup.index:
                        for key, val in add_to_node_popup.loc[node].iteritems():
                            node_info = node_info + '<br>' + \
                                        key + ': ' + '{:.{prec}f}'.format(val, prec=round_ndigits)

                node_trace['text'] += tuple([node_info])
        except:
            node_trace['marker']['color'] += tuple(['#888'])
            if node_labels:
                node_info = water_network_model.get_node(node).node_type + ': ' + str(node)

                node_trace['text'] += tuple([node_info])
            # node_trace['marker']['size'] += tuple([5])
    # node_trace['marker']['colorbar']['title'] = 'Node colorbar title'

    # Create figure
    data = [edge_trace, node_trace]
    layout = plotly.graph_objs.Layout(
        title=title,
        titlefont=dict(size=16),
        showlegend=False,
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



# Outdated problems with colormap
# def plot_interactive_network(water_network_model, node_attribute=None, node_attribute_name='Value', title=None,
#                              node_size=8, node_range=[None, None], node_cmap='jet', node_labels=True,
#                              link_width=1, add_colorbar=True, reverse_colormap=False,
#                              figsize=[700, 450], round_ndigits=2, add_to_node_popup=None,
#                              filename='plotly_network.html', auto_open=True):
#     """
#
#     """
#     network_graph = water_network_model.get_graph()
#
#     # Node attribute
#     if node_attribute is not None:
#         if isinstance(node_attribute, list):
#             node_cmap = 'reds'
#             add_colorbar = False
#         node_attribute = _format_node_attribute(node_attribute, water_network_model)
#     else:
#         add_colorbar = False
#
#     # Create edge trace
#     # edge_trace = plotly.graph_objs.Scatter(
#     #     x=[],
#     #     y=[],
#     #     text=[],
#     #     hoverinfo='text',
#     #     mode='lines',
#     #     line=dict(
#     #         # colorscale=link_cmap,
#     #         # reversescale=reverse_colormap,
#     #         color='#888',  # [],
#     #         width=link_width))
#     pipe_traces = plotly.graph_objs.Scatter(
#         name="Pipe",
#         x=[],
#         y=[],
#         text=[],
#         hoverinfo='text',
#         mode='lines',
#         # TODO change line shape ? line_shape='hv',
#         line=dict(
#             color='#888',
#             width=link_width
#         )
#     )
#
#     # Node trace for normal Junctions
#     node_junction = plotly.graph_objs.Scatter(
#         name="Junction",
#         x=[],
#         y=[],
#         text=[],
#         hoverinfo='text',
#         mode='markers',
#         marker=dict(
#             color=[],
#             size=node_size,
#             reversescale=False,
#             opacity=1,
#             line=dict(width=1),
#             symbol='circle'))
#     # Node trace for valves same as epanet
#     node_valve = plotly.graph_objs.Scatter(
#         name="Valve Junction",
#         x=[],
#         y=[],
#         text=[],
#         hoverinfo='text',
#         mode='markers',
#         marker=dict(
#             color=[],
#             size=node_size,
#             opacity=1,
#             line=dict(width=1),
#             symbol='bowtie',
#         ))
#     # Node trace for tanks
#     node_tank = plotly.graph_objs.Scatter(
#         name="Tank",
#         x=[],
#         y=[],
#         text=[],
#         hoverinfo='text',
#         mode='markers',
#         marker=dict(
#             color=[],
#             size=node_size + 10,
#             opacity=1,
#             line=dict(width=1),
#             symbol='square',
#         ))
#
#     # Node trace for sensors
#     node_sensor = plotly.graph_objs.Scatter(
#         name="Sensor",
#         x=[],
#         y=[],
#         text=[],
#         hoverinfo='text',
#         mode='markers',
#         marker=dict(
#             color=[],
#             size=node_size,
#             opacity=1,
#             line=dict(width=1),
#             symbol='arrow-right',
#         )
#     )
#
#     # getting min and max values    # TODO all of this into a function
#     max_node_value = max(node_attribute.values())
#     min_node_value = min(node_attribute.values())
#     print(max_node_value, min_node_value)
#     # creating a color map
#     norm = matplotlib.colors.Normalize(vmin=min_node_value, vmax=max_node_value)
#     cmap = matplotlib.cm.get_cmap('Blues')
#     color_scaler = matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap)
#
#     for edge in network_graph.edges():
#         x0, y0 = network_graph.nodes[edge[0]]['pos']
#         x1, y1 = network_graph.nodes[edge[1]]['pos']
#         pipe_traces['x'] += tuple([x0, x1, None])
#         pipe_traces['y'] += tuple([y0, y1, None])
#
#     for node in network_graph.nodes():
#         curr_node = None
#         x, y = network_graph.nodes[node]['pos']
#
#         # G.nodes[node] has a property "type", this can be Junction, Reservoir
#         if network_graph.nodes[node]["type"] == "Reservoir":
#             curr_node = node_tank
#         elif "Senzor" in str(node):
#             curr_node = node_sensor
#         elif "Jonctiune" in str(node) and "Junction" not in str(node):
#             curr_node = node_junction
#         else:
#             # print("Uknown node type")
#             curr_node = node_valve
#
#         curr_node['x'] += tuple([x])
#         curr_node['y'] += tuple([y])
#
#         try:
#             # Add node attributes
#             # print(node_attribute[node])
#             curr_node['marker']['color'] += color_scaler.to_rgba(node_attribute[node])
#             # print(curr_node['marker']['color'])
#
#             if node_labels:
#                 node_info = water_network_model.get_node(node).node_type + ': ' + str(node) + '<br>' + \
#                             node_attribute_name + ': ' + str(round(node_attribute[node], round_ndigits))
#                 if add_to_node_popup is not None:
#                     if node in add_to_node_popup.index:
#                         for key, val in add_to_node_popup.loc[node].iteritems():
#                             node_info = node_info + '<br>' + \
#                                         key + ': ' + '{:.{prec}f}'.format(val, prec=round_ndigits)
#
#                 curr_node['text'] += tuple([node_info])
#         except:
#             # curr_node['marker']['color'] += tuple(['#888'])
#             if node_labels:
#                 node_info = water_network_model.get_node(node).node_type + ': ' + str(node)
#
#                 curr_node['text'] += tuple([node_info])
#
#         # # Add node labels TODO my changes
#         # if node_labels:
#         #     node_info = water_network_model.get_node(node).node_type + ': ' + str(node) + '<br>' + \
#         #                 node_attribute_name + ': ' + "No data"
#         #     if add_to_node_popup is not None:
#         #         if node in add_to_node_popup.index:
#         #             for key, val in add_to_node_popup.loc[node].iteritems():
#         #                 node_info = node_info + '<br>' + \
#         #                             key + ': ' + '{:.{prec}f}'.format(val, prec=round_ndigits)
#         #
#         #     curr_node['text'] += tuple([node_info])
#
#     # Create figure
#     data = [pipe_traces, node_junction, node_valve, node_tank, node_sensor]
#     layout = plotly.graph_objs.Layout(
#         title=title,
#         titlefont=dict(size=16),
#         showlegend=False,
#         width=figsize[0],
#         height=figsize[1],
#         hovermode='closest',
#         # coloraxis=plotly.graph_objects.layout.Coloraxis(
#         #     reversescale=reverse_colormap,
#         #     colorscale="Reds",
#         #     showscale=add_colorbar
#         # ),
#         margin=dict(b=20, l=5, r=5, t=40),
#         xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
#         yaxis=dict(showgrid=False, zeroline=False, showticklabels=False))
#
#     fig = plotly.graph_objs.Figure(data=data, layout=layout)
#     if filename:
#         plotly.offline.plot(fig, filename=filename, auto_open=auto_open)
#     else:
#         plotly.offline.plot(fig, auto_open=auto_open)
