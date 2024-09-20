import torch
import torch.nn as nn
import json


class QueryPlanNode:
    def __init__(self, plan_parameters, statistics, children=None):
        self.plan_parameters = plan_parameters
        self.children = children if children is not None else []

        self.statistics = statistics

    @staticmethod
    def normalize(feature, center, scale):
        return (feature - center) / scale
    

    def to_features(self):
        """
        Convert the plan parameters into a feature vector.
        You can choose which parameters are important for your model.
        """
        self.op_name = self.plan_parameters.get('op_name', 'None')
        self.est_startup_cost = self.plan_parameters.get('est_startup_cost', 0)
        self.est_cost = self.plan_parameters.get('est_cost', 0)
        self.est_card = self.plan_parameters.get('est_card', 0)
        self.est_width = self.plan_parameters.get('est_width', 0)
        self.workers_planned = self.plan_parameters.get('workers_planned', 0)
        self.est_children_card = self.plan_parameters.get('est_children_card', 0)


        statistics = self.statistics
        # print(f"statistics: {statistics}")
        op_name_statistics = statistics.get('op_name')
        est_startup_cost_statistics = statistics.get('est_startup_cost')
        est_cost_statistics = statistics.get('est_cost')
        est_card_statistics = statistics.get('est_card')
        est_width_statistics = statistics.get('est_width')
        workers_planned_statistics = statistics.get('workers_planned')
        est_children_card_statistics = statistics.get('est_children_card')

        # Normalize the features
        # print(f"op_name_statistics {op_name_statistics}")
        op_name = op_name_statistics['value_dict'].get(self.op_name)
        est_startup_cost = self.normalize(self.est_startup_cost, est_startup_cost_statistics['center'], est_startup_cost_statistics['scale'])
        est_cost = self.normalize(self.est_cost, est_cost_statistics['center'], est_cost_statistics['scale'])
        est_card = self.normalize(self.est_card, est_card_statistics['center'], est_card_statistics['scale'])
        est_width = self.normalize(self.est_width, est_width_statistics['center'], est_width_statistics['scale'])
        workers_planned = self.normalize(self.workers_planned, workers_planned_statistics['center'], workers_planned_statistics['scale'])
        est_children_card = self.normalize(self.est_children_card, est_children_card_statistics['center'], est_children_card_statistics['scale'])

        return [op_name, est_startup_cost, est_cost, est_card, est_width, workers_planned, est_children_card]

    def flatten(self):
        """
        Flatten the tree into a list of feature vectors (breadth-first).
        """
        nodes = [self]
        feature_list = []
        while nodes:
            node = nodes.pop(0)
            feature_list.append(node.to_features())
            nodes.extend(node.children)
        return feature_list


def parse_plan(parsed_plan, statistics):
    """
    Parse the JSON-like parsed plan into a tree of QueryPlanNode.
    """
    def create_node(plan_data):
        # Extract the plan parameters and recursively process children
        plan_parameters = plan_data.get('plan_parameters', {})
        children = [create_node(child) for child in plan_data.get('children', [])]
        return QueryPlanNode(plan_parameters, statistics, children)

    return create_node(parsed_plan).flatten()




