import json


class PlanNode:
    def __init__(self, node_id, op_name, est_startup_cost, est_cost, est_card, est_width,  workers_planned, est_children_card):
        self.node_id = node_id
        self.op_name = op_name
        self.est_startup_cost = est_startup_cost
        self.est_cost = est_cost
        self.est_card = est_card
        self.est_width = est_width
        self.workers_planned = workers_planned
        self.est_children_card = est_children_card
        self.children = []

        self.get_featurs()
    
    @staticmethod
    def normalize(feature, center, scale):
        return (feature - center) / scale

    def get_featurs(self):
        statistics_file_name = '../tpch_data_20000/statistics_workload_combined.json'
        with open(statistics_file_name) as f:
            statistics = json.load(f)

        op_name_statistics = statistics.get('op_name')
        est_startup_cost_statistics = statistics.get('est_startup_cost')
        est_cost_statistics = statistics.get('est_cost')
        est_card_statistics = statistics.get('est_card')
        est_width_statistics = statistics.get('est_width')
        workers_planned_statistics = statistics.get('workers_planned')
        est_children_card_statistics = statistics.get('est_children_card')

        op_name = op_name_statistics['value_dict'].get(self.op_name)
        est_startup_cost = self.normalize(self.est_startup_cost, est_startup_cost_statistics['center'], est_startup_cost_statistics['scale'])
        est_cost = self.normalize(self.est_cost, est_cost_statistics['center'], est_cost_statistics['scale'])
        est_card = self.normalize(self.est_card, est_card_statistics['center'], est_card_statistics['scale'])
        est_width = self.normalize(self.est_width, est_width_statistics['center'], est_width_statistics['scale'])
        workers_planned = self.normalize(self.workers_planned, workers_planned_statistics['center'], workers_planned_statistics['scale'])
        est_children_card = self.normalize(self.est_children_card, est_children_card_statistics['center'], est_children_card_statistics['scale'])

        self.features = [op_name, est_startup_cost, est_card, est_width, workers_planned, est_children_card]


    def add_child(self, child):
        self.children.append(child)

    # def __str__(self):
    #     return "PlanNode(op_name={}, est_startup_cost={}, est_cost={}, est_card={}, est_width={}, workers_planned={}, est_children_card={}, children={})".format(self.op_name, self.est_startup_cost, self.est_cost, self.est_card, self.est_width, self.workers_planned, self.est_children_card, self.children)

    def __str__(self):
        return f"PlanNode({self.node_id})"


    def traverse_tree(self, edge_src_nodes, edge_tgt_nodes, features_list):
        features_list.append(self.features)
        for child in self.children:
            edge_src_nodes.append(child.node_id)
            edge_tgt_nodes.append(self.node_id)
            edge_src_nodes, edge_tgt_nodes, features_list = child.traverse_tree(edge_src_nodes, edge_tgt_nodes, features_list)
        return edge_src_nodes, edge_tgt_nodes, features_list


def parse_postgres_plan(plan, global_node_id=[0]):
    plan_parameters = plan.get('plan_parameters')
    op_name = plan_parameters.get("op_name", "")
    est_startup_cost = plan_parameters.get('est_startup_cost', 0)
    est_cost = plan_parameters.get('est_cost', 0)
    est_card = plan_parameters.get('est_card', 0)
    est_width = plan_parameters.get('est_width', 0)
    workers_planned = plan_parameters.get('workers_planned', 0)
    est_children_card = plan_parameters.get('est_children_card', 0)

    node = PlanNode(global_node_id[0], op_name, est_startup_cost, est_cost, est_card, est_width, workers_planned, est_children_card)
    global_node_id[0] += 1

    for i,subplan in enumerate(plan.get("children", [])):
        child_node = parse_postgres_plan(subplan)
        node.add_child(child_node)

    return node


def parse_postgres_plan(plan, next_node_id=0):
    plan_parameters = plan.get('plan_parameters')
    op_name = plan_parameters.get("op_name", "")
    est_startup_cost = plan_parameters.get('est_startup_cost', 0)
    est_cost = plan_parameters.get('est_cost', 0)
    est_card = plan_parameters.get('est_card', 0)
    est_width = plan_parameters.get('est_width', 0)
    workers_planned = plan_parameters.get('workers_planned', 0)
    est_children_card = plan_parameters.get('est_children_card', 0)
    node = PlanNode(next_node_id, op_name, est_startup_cost, est_cost, est_card, est_width, workers_planned, est_children_card)
    next_node_id += 1

    for i,subplan in enumerate(plan.get("children", [])):
        child_node, next_node_id = parse_postgres_plan(subplan, next_node_id)
        node.add_child(child_node)

    return node, next_node_id

def print_tree(node, level=0):
    print("  " * level + str(node))
    for child in node.children:
        print_tree(child, level + 1)

if __name__ == '__main__':
    with open('../tpch_data_20000/val_plans.json') as f:
        plans = json.load(f)
    # print(f"json file loaded")
    
    plan = plans['parsed_plans'][1]
    print(f"plan {plan['plan_parameters']}")
    node, _  = parse_postgres_plan(plan, 0)
    print(f"node", node)
    print("tree:\n")
    print_tree(node)

    node2, _ = parse_postgres_plan(plan)
    print(f"node2", node2)
    print("tree:\n")
    print_tree(node2)
    exit()
    edge_src_nodes = []
    edge_tgt_nodes = []
    features_list = []
    edge_src_nodes, edge_tgt_nodes, features_list, num_nodes = node.traverse_tree(edge_src_nodes, edge_tgt_nodes, features_list)

    print(f"edge_src_nodes: {edge_src_nodes}, edge_tgt_nodes: {edge_tgt_nodes}, features_list: {features_list}, num_nodes: {num_nodes}, peakmem: {plan['peakmem']}")