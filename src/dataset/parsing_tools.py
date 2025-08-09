import pandas


class ColumnNameGenerator:
    def __init__(self, old_name, new_name=None, neo4j_type=None):
        self.old_name = old_name
        if new_name is None:
            self.new_name = old_name
        else:
            self.new_name = new_name
        self.neo4j_type = neo4j_type

    def get_neo4j_name(self):
        if self.neo4j_type is None:
            return self.new_name
        else:
            return f'{self.new_name}:{self.neo4j_type}'

def get_csv_name_map(column_name_generator_list):
    return {name_info.old_name: name_info.new_name for name_info in column_name_generator_list}


def get_neo4j_name_map(column_name_generator_list):
    return {name_info.old_name: name_info.get_neo4j_name() for name_info in column_name_generator_list}