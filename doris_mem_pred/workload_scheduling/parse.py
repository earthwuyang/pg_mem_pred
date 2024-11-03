import re
import math



# Define Node Types and Mappings
NodeType = {
    'PhysicalIntersect', 
    'PhysicalLimit', 
    'PhysicalCTEProducer', 
    'PhysicalHashJoin', 
    'PhysicalUnion', 
    'PhysicalExcept', 
    'PhysicalPartitionTopN', 
    'PhysicalResultSink', 
    'PhysicalQuickSort', 
    'PhysicalDistribute', 
    'PhysicalFilter', 
    'PhysicalHashAggregate', 
    'PhysicalAssertNumRows', 
    'PhysicalCTEConsumer', 
    'PhysicalOlapScan', 
    'PhysicalCTEAnchor', 
    'PhysicalTopN', 
    'PhysicalProject', 
    'PhysicalRepeat', 
    'PhysicalWindow', 
    'PhysicalNestedLoopJoin',
    'PhysicalEmptyRelation',
    'PhysicalOneRowRelation',
    'PhysicalStorageLayerAggregate'
}
nodetype2idx = {t: i for i, t in enumerate(NodeType)}

# Distribution Spec Mappings (if needed)
distributionSpec_list = {
    'DistributionSpecReplicated', 
    'DistributionSpecGather', 
    'DistributionSpecExecutionAny', 
    'DistributionSpecHash'
}
distributionSpec2idx = {t: i for i, t in enumerate(distributionSpec_list)}

# Define the Node Class
class Node:
    def __init__(self, node_id, name, id_, order, attributes, cardinality, table, node_level):
        self.nodeid = node_id
        self.name = name
        self.id = id_
        self.order = order  # Root node has order -1
        self.attributes = attributes
        self.cardinality = cardinality
        self.table = table
        self.node_level = node_level
        self.children = []

        self.table = None
        self.columns = None
        self.limit = 0

        if self.name == 'PhysicalProject':
            self.extract_project(attributes)
        if self.name == 'PhysicalDistribute':
            self.extract_distribute(attributes)
        if self.name == 'PhysicalFilter':
            self.extract_filter(attributes)
        if self.name == 'PhysicalTopN':
            self.extract_topn(attributes)

        nodetype = nodetype2idx.get(self.name, -1)
        card = math.log1p(int(float(self.cardinality))) if self.cardinality else 0
        num_of_columns = len(self.columns) if self.columns else 0
        limit = int(self.limit) if self.limit else 0

        # Feature Vector: [Node Type, Log(Cardinality), Num of Columns, Limit]
        self.features = [nodetype, card, num_of_columns, limit]

    def extract_project(self, attributes):
        project_pattern = r"projects=\[([^\]]+)\]"
        match = re.search(project_pattern, attributes)
        if match:
            self.columns = match.group(1).split(',')
        else:
            self.columns = []

    def extract_distribute(self, attributes):
        distribute_pattern = r"distributionSpec=([A-Za-z]+)"
        match = re.search(distribute_pattern, attributes)
        if match:
            self.distributionSpec = distributionSpec2idx.get(match.group(1), -1)
        else:
            self.distributionSpec = -1

    def extract_filter(self, attributes):
        text = attributes
        start_keyword = "predicates=("
        start_index = text.find(start_keyword)
        if start_index == -1:
            self.predicates = None
            return
        start_index += len(start_keyword)
        open_parens = 1
        end_index = start_index

        while open_parens > 0 and end_index < len(text):
            if text[end_index] == '(':
                open_parens += 1
            elif text[end_index] == ')':
                open_parens -= 1
            end_index += 1

        if open_parens == 0:
            self.predicates = text[start_index:end_index-1]
        else:
            self.predicates = None

    def extract_topn(self, attributes):
        topn_pattern = r"limit=([0-9]+)"
        match = re.search(topn_pattern, attributes)
        if match:
            self.limit = match.group(1)
        else:
            self.limit = 0

# Utility Functions
def extract_stats(attributes):
    stats_pattern = r'stats=([\d,\.]+)'
    match = re.search(stats_pattern, attributes)
    if match:
        return match.group(1).replace(',', '')  # e.g., '1,593,010' -> '1593010'
    return '0'

def parse_physical_node(node_str, node_level, node_id):
    node_pattern = r'([A-Za-z0-9]+)(?:\[([A-Za-z0-9_]+)\])?(?:@([0-9]+)?)? \((.+)'  
    match = re.match(node_pattern, node_str)
    if not match:
        raise ValueError(f"Unable to parse node: {node_str}")
    name, id_, order, attributes = match.groups()
    table = None
    if name == "PhysicalOlapScan":
        table = id_
        id_ = -1
    cardinality = extract_stats(attributes)
    node = Node(
        node_id, 
        name, 
        int(id_) if id_ else -1, 
        int(order) if order else -1, 
        attributes, 
        int(float(cardinality)) if cardinality else 0, 
        table, 
        node_level
    )  
    return node

def parse_tree(lines):
    lines = lines[1:]
    stack = []
    edge_src_nodes = []
    edge_tgt_nodes = []
    features_list = []
    node_levels = []
    for node_id, line in enumerate(lines):
        prefix = line.split('Physical')[0]
        node_level = len(prefix)
        node_str = line.split(prefix)[1] if prefix else line
        node = parse_physical_node(node_str, node_level, node_id)
        features_list.append(node.features)
        node_levels.append(node.node_level)

        while stack and stack[-1].node_level >= node.node_level:
            stack.pop()
        if stack:
            parent = stack[-1]
            parent.children.append(node)
            edge_src_nodes.append(parent.nodeid)
            edge_tgt_nodes.append(node.nodeid)

        stack.append(node)

    return stack[0], edge_src_nodes, edge_tgt_nodes, features_list, node_levels  # stack[0] is the root node

def traverse_tree(node):
    yield node
    for child in node.children:
        yield from traverse_tree(child)