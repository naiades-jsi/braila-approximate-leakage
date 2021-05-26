
def pretty_print(node_arr):
    string_to_print = ""

    for index, node in enumerate(node_arr):
        stripped_node = node.replace(", 16.0LPS", "")
        string_to_print += "  " + stripped_node + ",\n"

    return string_to_print