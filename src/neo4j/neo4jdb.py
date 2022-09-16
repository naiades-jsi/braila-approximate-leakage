from neo4j import GraphDatabase

class Neo4jDatabase:

    def __init__(self, uri, user, password):
        self._driver = GraphDatabase.driver(uri, auth=(user, password))

    def execute_query(self, query_str):
        with self._driver.session() as session:
            result = session.write_transaction(self._run_query, query_str)
        return result

    @staticmethod
    def _run_query(tx, query_str):
        result = tx.run(query_str)
        return result

