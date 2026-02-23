"""
Legal Knowledge Graph — br-legal-kg RDF integration.

Importa o Knowledge Graph juridico brasileiro (github.com/hansidm/br-legal-kg)
e fornece queries de navegacao para legislacao e relacoes entre normas.

Usa RDFLib para parsing e queries SPARQL in-memory. Para producao,
considerar Apache AGE (extensao Postgres) ou Neo4j.
"""

import logging
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger("jurisai.legal_kg")


class LegalKnowledgeGraph:
    """
    Knowledge Graph juridico brasileiro.

    Carrega grafo RDF (br-legal-kg) e permite queries de navegacao
    tipo "quais leis referenciam o Art. 5 da CF?" ou "quais artigos
    do CPC foram alterados pela Lei X?".
    """

    NAMESPACES = {
        "leg": "http://www.semanticweb.org/legislation#",
        "rdf": "http://www.w3.org/1999/02/22-rdf-syntax-ns#",
        "rdfs": "http://www.w3.org/2000/01/rdf-schema#",
        "owl": "http://www.w3.org/2002/07/owl#",
        "dcterms": "http://purl.org/dc/terms/",
    }

    def __init__(self, rdf_path: str | None = None) -> None:
        self._graph = None
        self._rdf_path = rdf_path

    def _ensure_loaded(self) -> None:
        """Lazy-load the RDF graph."""
        if self._graph is not None:
            return

        try:
            import rdflib
        except ImportError:
            raise ImportError("rdflib nao instalado. Execute: pip install rdflib>=7.0.0")

        self._graph = rdflib.Graph()

        if self._rdf_path and Path(self._rdf_path).exists():
            fmt = "turtle" if self._rdf_path.endswith(".ttl") else "xml"
            self._graph.parse(self._rdf_path, format=fmt)
            logger.info("KG carregado: %d triplas de %s", len(self._graph), self._rdf_path)
        else:
            logger.warning("KG RDF nao encontrado em %s, grafo vazio", self._rdf_path)

        for prefix, uri in self.NAMESPACES.items():
            self._graph.bind(prefix, uri)

    def load_from_file(self, path: str) -> int:
        """Load RDF data from a file."""
        import rdflib

        if self._graph is None:
            self._graph = rdflib.Graph()
            for prefix, uri in self.NAMESPACES.items():
                self._graph.bind(prefix, uri)

        before = len(self._graph)
        fmt = "turtle" if path.endswith(".ttl") else "xml"
        self._graph.parse(path, format=fmt)
        added = len(self._graph) - before
        logger.info("KG: +%d triplas de %s (total: %d)", added, path, len(self._graph))
        return added

    def load_from_github(self, repo: str = "hansidm/br-legal-kg") -> int:
        """
        Download and load the br-legal-kg from GitHub.

        Downloads the main RDF file from the repository.
        """
        import tempfile

        import httpx

        raw_url = f"https://raw.githubusercontent.com/{repo}/main/ontology.ttl"
        alt_urls = [
            f"https://raw.githubusercontent.com/{repo}/main/kg.ttl",
            f"https://raw.githubusercontent.com/{repo}/master/ontology.ttl",
            f"https://raw.githubusercontent.com/{repo}/main/data/ontology.owl",
        ]

        for url in [raw_url] + alt_urls:
            try:
                resp = httpx.get(url, timeout=60.0, follow_redirects=True)
                if resp.status_code == 200:
                    with tempfile.NamedTemporaryFile(
                        suffix=".ttl" if url.endswith(".ttl") else ".owl",
                        delete=False,
                        mode="w",
                        encoding="utf-8",
                    ) as f:
                        f.write(resp.text)
                        return self.load_from_file(f.name)
            except Exception as e:
                logger.debug("Tentando %s: %s", url, e)
                continue

        logger.error("Nao foi possivel baixar KG de %s", repo)
        return 0

    def query_sparql(self, sparql: str) -> list[dict[str, str]]:
        """Execute a SPARQL query against the knowledge graph."""
        self._ensure_loaded()
        results: list[dict[str, str]] = []

        try:
            qres = self._graph.query(sparql)
            for row in qres:
                result = {}
                for i, var in enumerate(qres.vars):
                    result[str(var)] = str(row[i]) if row[i] else ""
                results.append(result)
        except Exception as e:
            logger.error("SPARQL query error: %s", e)

        return results

    def find_law(self, identifier: str) -> dict[str, Any] | None:
        """
        Find a law by identifier (e.g., "Lei 10.406/2002", "CPC").

        Returns structured info about the law and its relations.
        """
        self._ensure_loaded()
        query = f"""
        SELECT ?subject ?predicate ?object
        WHERE {{
            ?subject ?predicate ?object .
            FILTER(
                CONTAINS(LCASE(STR(?subject)), "{identifier.lower()}") ||
                CONTAINS(LCASE(STR(?object)), "{identifier.lower()}")
            )
        }}
        LIMIT 50
        """
        results = self.query_sparql(query)
        if not results:
            return None

        return {
            "identifier": identifier,
            "triples": results,
            "total": len(results),
        }

    def find_references_to(self, law: str) -> list[dict[str, str]]:
        """Find all laws/articles that reference a given law."""
        self._ensure_loaded()
        query = f"""
        SELECT ?source ?predicate
        WHERE {{
            ?source ?predicate ?target .
            FILTER(CONTAINS(LCASE(STR(?target)), "{law.lower()}"))
            FILTER(?predicate != rdf:type)
        }}
        LIMIT 100
        """
        return self.query_sparql(query)

    def find_amendments(self, law: str) -> list[dict[str, str]]:
        """Find all amendments to a given law."""
        self._ensure_loaded()
        query = f"""
        SELECT ?amending_law ?predicate ?detail
        WHERE {{
            ?amending_law ?predicate ?target .
            FILTER(CONTAINS(LCASE(STR(?target)), "{law.lower()}"))
            FILTER(
                CONTAINS(LCASE(STR(?predicate)), "altera") ||
                CONTAINS(LCASE(STR(?predicate)), "modifica") ||
                CONTAINS(LCASE(STR(?predicate)), "amend")
            )
            OPTIONAL {{ ?amending_law rdfs:label ?detail }}
        }}
        LIMIT 100
        """
        return self.query_sparql(query)

    def find_revocations(self, law: str) -> list[dict[str, str]]:
        """Find revocations related to a given law."""
        self._ensure_loaded()
        query = f"""
        SELECT ?revoking_law ?predicate ?detail
        WHERE {{
            ?revoking_law ?predicate ?target .
            FILTER(CONTAINS(LCASE(STR(?target)), "{law.lower()}"))
            FILTER(
                CONTAINS(LCASE(STR(?predicate)), "revoga") ||
                CONTAINS(LCASE(STR(?predicate)), "revoke")
            )
            OPTIONAL {{ ?revoking_law rdfs:label ?detail }}
        }}
        LIMIT 50
        """
        return self.query_sparql(query)

    def get_article_network(self, law: str, article: str) -> dict[str, Any]:
        """
        Get the reference network for a specific article.

        Returns all articles that reference or are referenced by the target.
        """
        self._ensure_loaded()
        search_term = f"art. {article}"

        query = f"""
        SELECT ?source ?predicate ?target
        WHERE {{
            ?source ?predicate ?target .
            FILTER(
                (CONTAINS(LCASE(STR(?source)), "{law.lower()}") &&
                 CONTAINS(LCASE(STR(?source)), "{search_term.lower()}")) ||
                (CONTAINS(LCASE(STR(?target)), "{law.lower()}") &&
                 CONTAINS(LCASE(STR(?target)), "{search_term.lower()}"))
            )
        }}
        LIMIT 100
        """
        results = self.query_sparql(query)
        return {
            "law": law,
            "article": article,
            "references": results,
            "total_connections": len(results),
        }

    def stats(self) -> dict[str, int]:
        """Get basic statistics about the knowledge graph."""
        self._ensure_loaded()
        return {
            "total_triples": len(self._graph),
            "subjects": len(set(self._graph.subjects())),
            "predicates": len(set(self._graph.predicates())),
            "objects": len(set(self._graph.objects())),
        }

    def add_triple(self, subject: str, predicate: str, obj: str) -> None:
        """Add a triple to the graph (for dynamic updates)."""
        self._ensure_loaded()
        import rdflib
        self._graph.add((
            rdflib.URIRef(subject),
            rdflib.URIRef(predicate),
            rdflib.Literal(obj),
        ))

    def save(self, path: str, fmt: str = "turtle") -> None:
        """Save the graph to file."""
        self._ensure_loaded()
        self._graph.serialize(destination=path, format=fmt)
        logger.info("KG salvo: %d triplas em %s", len(self._graph), path)
