import os
import wntr
import pickle as pkl
import json

import src.neo4j.neo4jdb as neo
import src.analytics.analytics as analytics

_empty_id_cnt = -1

def _neo_node_id(node_id):
    global _empty_id_cnt

    if node_id == '-':
        _empty_id_cnt += 1
        return 'empty-' + str(_empty_id_cnt)
    return node_id


def _load_from_sim(base_dir, sim_fpath, node_id, simN):
    def _load():
        print('loading network')

        with open(sim_fpath, 'rb') as f_in:
            sim_obj = pkl.load(f_in)
        node_sim_vec = sim_obj[node_id]
        sim_dict = node_sim_vec[simN]

        pressure_df = sim_dict['LPM']
        divergence_df = sim_dict['DM']
        used_leak_flows_df = sim_dict['LM']

        water_network_model = sim_dict['network']
        leak_node_id = sim_dict['Meta']['leakNode']

        if os.path.exists(os.path.join(base_dir, 'model_stats.json')):
            with open(os.path.join(base_dir, 'model_stats.json')) as f_in:
                node_stats_map = json.load(f_in)

        print('loaded')

        return water_network_model, {
            'leak_node_id': leak_node_id,

            'pressure_df': pressure_df,
            'divergence_df': divergence_df,
            'used_leak_flows_df': used_leak_flows_df,

            'node_stats_map': node_stats_map
        }
    return _load

def _load_from_file(network_fname):
    def _load():
        print('loading network')
        water_network_model = wntr.network.WaterNetworkModel(network_fname)
        return water_network_model, None
    return _load



def main(neo_host, neo_uname, neo_pass, load_network_fun, hourN):
    print('connecting to DB')
    graph_db = neo.Neo4jDatabase(neo_host, neo_uname, neo_pass)
    print('connected')

    water_network_model, stats_map = load_network_fun()
    network_graph = water_network_model.get_graph()

    node_id_vec = network_graph.nodes()
    edge_tup_vec = network_graph.edges()

    node_map = {node_id: node for node_id, node in water_network_model.nodes()}
    edge_map = {(edge[1].start_node_name, edge[1].end_node_name): edge[1] for edge in water_network_model.links()}

    # calculate distance from reservoirs
    reservoir_upstream_dist_map = analytics.dist_from_set_map(network_graph, ['RaduNegru1', 'RaduNegru2', 'Apollo'])
    reservoir_dist_map = analytics.undir_dist_from_set_map(network_graph, ['RaduNegru1', 'RaduNegru2', 'Apollo'])

    print('creating Cypher string')

    cypher_str = ''
    for node_id in node_id_vec:
        node = node_map[node_id]

        reservoir_upstream_dist = reservoir_upstream_dist_map[node_id]
        reservoir_dist = reservoir_dist_map[node_id]

        cypher_str += '\nWITH sum(1) AS _'
        cypher_str += f'\nMERGE (n:{node.node_type} {{nodeId: \'{_neo_node_id(node_id)}\'}})'

        if node.node_type == 'Junction':
            cypher_str += f'\nSET n += {{' + \
                    f'x: {node.coordinates[0]}, y: {node.coordinates[1]}' + \
            f'}}'
        elif node.node_type == 'Reservoir':
            cypher_str += f'\nSET n += {{' + \
                    f'x: {node.coordinates[0]}, y: {node.coordinates[1]}' + \
            f'}}'

        cypher_str += f'\nSET n += {{reservoirUpstreamDist: {min(reservoir_upstream_dist, 1000)}}}'
        cypher_str += f'\nSET n += {{reservoirDist: {min(reservoir_dist, 1000)}}}'

        if node_id.startswith('Sensor'):
            cypher_str += '\nSET n:Sensor'
        if node_id.startswith('PT'):
            cypher_str += '\nSET n:PT'
        if node_id.startswith('J-NR'):
            cypher_str += '\nSET n:JNR'
        if node_id == 'J-Apollo':
            cypher_str += '\nSET n:Apollo'


    for src_node_id, dst_node_id in edge_tup_vec:
        edge = edge_map[(src_node_id, dst_node_id)]
        edge_type = edge.link_type

        src_node = node_map[src_node_id]
        dst_node = node_map[dst_node_id]

        cypher_str += '\nWITH sum(1) AS _'
        cypher_str += f'\nMATCH (src:{src_node.node_type} {{nodeId: \'{_neo_node_id(src_node_id)}\'}})'
        cypher_str += f'\nMATCH (dst:{dst_node.node_type} {{nodeId: \'{_neo_node_id(dst_node_id)}\'}})'
        cypher_str += f'\nMERGE (src)-[edge:{edge.link_type.upper()}]->(dst)'
        cypher_str += f'\nSET edge += {{length: {edge.length}, diameter: {edge.diameter}, roughness: {edge.roughness}, minorLoss: {edge.minor_loss}, checkValve: {edge.cv}}}'

    if stats_map is not None:
        leak_node_id = stats_map['leak_node_id']

        pressure_df = stats_map['pressure_df']
        divergence_df = stats_map['divergence_df']
        used_leak_flows_df = stats_map['used_leak_flows_df']

        node_stats_map = stats_map['node_stats_map']

        pressure_state_map = pressure_df.iloc[hourN, :]
        divergence_state_map = divergence_df.iloc[hourN, :]
        used_leak_flows_state_map = used_leak_flows_df.iloc[hourN, :]

        node_state_map = {node_id: {
            'pressure': pressure_state_map[node_id],
            'pressureDiff': divergence_state_map[node_id]
        } for node_id in pressure_df.columns}

        cypher_str += f'\nWITH sum(1) AS _'
        cypher_str += f'\nMATCH (n) WHERE n.nodeId = \'{_neo_node_id(leak_node_id)}\''
        cypher_str += f'\nSET n:Leak'

        for node_id, node_prop_map in node_state_map.items():
            if len(node_prop_map) > 0:
                prop_str = ','.join([prop_key + ': ' + str(prop_val) for prop_key, prop_val in node_prop_map.items() if prop_val is not None])
                if len(prop_str) > 0:
                    cypher_str += f'\nWITH sum(1) AS _'
                    cypher_str += f'\nMATCH (n {{nodeId: \'{_neo_node_id(node_id)}\'}})'

                    cypher_str += '\nSET n += {' + prop_str + '}'

        for node_id, node_stats in node_stats_map.items():
            hit_perc = node_stats['hit_perc']
            mean_misclassify_dist = node_stats['mean_misclassify_dist']
            max_misclassify_dist = node_stats['max_misclassify_dist']
            is_undetectable = node_stats['is_undetectable']

            cypher_str += f'\nWITH sum(1) AS _'
            cypher_str += f'\nMATCH (n {{nodeId: \'{_neo_node_id(node_id)}\'}})'

            cypher_str += f'\nSET n += {{classifyHitPerc: {hit_perc}}}'
            cypher_str += f'\nSET n += {{misclassifyMeanDist: {mean_misclassify_dist}}}'
            cypher_str += f'\nSET n += {{misclassifyMaxDist: {max_misclassify_dist}}}'
            cypher_str += f'\nSET n += {{isUndetectable: {is_undetectable}}}'

    cypher_str += '\nRETURN sum(1) AS _'

    print('clearing database')
    graph_db.execute_query('MATCH (n) DETACH DELETE n;')

    print(cypher_str)

    graph_db.execute_query(cypher_str)
    print('done')


if __name__ == '__main__':
    neo_host = 'neo4j+s://f9af0d5f.databases.neo4j.io'
    neo_uname = 'neo4j'
    neo_pass = 'TYQj2OrUwJATh7MovgY0Z0qwnjJUqX5gtkeoj7Xa2Gc'

    hourN = 16

    # load_network_fun = _load_from_file('./data/epanet_networks/Braila_V2022_2_2.inp')
    load_network_fun = _load_from_sim('data/sim-single-leak/', '/home/lstopar/storage/data/naiadas/2022-08-09_sim-single-leak/1_leak_17.100-17.300_t_12.pkl', 'Jonctiune-2195', 0)

    main(neo_host, neo_uname, neo_pass, load_network_fun, hourN)
