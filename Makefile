WD_ENT=data/kg-index/wikidata-entities
WD_PROP=data/kg-index/wikidata-properties

QLEVER_TIMEOUT=1h

WD_URL=https://qlever.cs.uni-freiburg.de/api/wikidata
WD_ACCESS_TOKEN=null

FB_URL=https://qlever.cs.uni-freiburg.de/api/freebase
FB_ACCESS_TOKEN=null

DBPEDIA_URL=https://qlever.cs.uni-freiburg.de/api/dbpedia
DBPEDIA_ACCESS_TOKEN=null

SEARCH_INDEX=qgram

.PHONY: all data querylogs indices examples
all: data querylogs indices examples

data:
	@echo "Preparing simple questions"
	@python scripts/prepare_data.py \
	--wikidata-simple-questions third_party/KGQA-datasets/simple_wikidata_qa \
	--output data/wikidata-simplequestions \
	--entities $(WD_ENT) \
	--properties $(WD_PROP) \
	--progress
	@echo "Preparing lc quad wikidata"
	@python scripts/prepare_data.py \
	--lc-quad2-wikidata third_party/KGQA-datasets/lcquad_v2 \
	--output data/wikidata-lcquad2 \
	--entities $(WD_ENT) \
	--properties $(WD_PROP) \
	--progress
	@echo "Preparing qald 10"
	@python scripts/prepare_data.py \
	--qald-10 third_party/KGQA-datasets/qald/qald-10.py \
	--output data/wikidata-qald10 \
	--entities $(WD_ENT) \
	--properties $(WD_PROP) \
	--progress
	@echo "Preparing mcwq"
	@python scripts/prepare_data.py \
	--mcwq data/raw/mcwq \
	--output data/wikidata-mcwq \
	--entities $(WD_ENT) \
	--properties $(WD_PROP) \
	--progress
	@echo "Preparing qa wiki"
	@python scripts/prepare_data.py \
	--qa-wiki data/raw/qa_wiki/qa_wiki.tsv \
	--output data/wikidata-qa-wiki \
	--entities $(WD_ENT) \
	--properties $(WD_PROP) \
	--progress
	@echo "Preparing OpenHermes2.5"
	@mkdir -p data/open-hermes-2.5
	@python scripts/prepare_openhermes.py data/open-hermes-2.5

WD_QUERY_LOG_SOURCE=organic

querylogs:
	@echo "Preparing wikidata query logs"
	@python scripts/prepare_wikidata_query_logs.py \
	--files data/wikidata-query-logs/downloads/*.tsv \
	--output-dir data/wikidata-query-logs \
	--entities $(WD_ENT) \
	--properties $(WD_PROP) \
	--$(WD_QUERY_LOG_SOURCE)-only \
	--progress

examples:
	@echo "Creating example indices"
	@mkdir -p data/example-index
	@python scripts/prepare_examples.py \
	data/wikidata-simplequestions/train_examples.tsv \
	data/example-index/wikidata-simplequestions.bin \
	--progress
	@python scripts/prepare_examples.py \
	data/wikidata-qa-wiki/train_examples.tsv \
	data/wikidata-lcquad2/train_examples.tsv \
	data/wikidata-qald10/train_examples.tsv \
	data/wikidata-mcwq/train_examples.tsv \
	data/example-index/wikidata.bin \
	--progress

search-indices: wikidata-data wikidata-search-indices freebase-data freebase-search-indices dbpedia-data dbpedia-search-indices

wikidata-data:
	# wikidata entities
	# https://qlever.cs.uni-freiburg.de/wikidata/A8IsDS
	@mkdir -p data/search-index/wikidata-entities/$(SEARCH_INDEX)
	@curl -s $(WD_URL) -H "Accept: text/tab-separated-values" \
	--data-urlencode query="PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#> PREFIX skos: <http://www.w3.org/2004/02/skos/core#> PREFIX schema: <http://schema.org/> PREFIX wikibase: <http://wikiba.se/ontology#> PREFIX wd: <http://www.wikidata.org/entity/> PREFIX wdt: <http://www.wikidata.org/prop/direct/> SELECT DISTINCT ?label ?score (GROUP_CONCAT(DISTINCT ?alias; SEPARATOR=\";\") AS ?synonyms) ?id (GROUP_CONCAT(DISTINCT ?info; SEPARATOR=\"\t\") AS ?infos) WHERE { { ?id @en@rdfs:label ?label } MINUS { ?id wdt:P31/wdt:P279* wd:Q17442446 } OPTIONAL { ?id @en@skos:altLabel ?alias } OPTIONAL { ?id ^schema:about/wikibase:sitelinks ?score } OPTIONAL { { ?id @en@schema:description ?info } UNION { ?id wdt:P279 ?subclass_ . ?id @en@rdfs:label ?info } UNION { ?id wdt:P31 ?subclass_ . ?id @en@rdfs:label ?info } UNION { ?id wdt:P106 ?subclass_ . ?id @en@rdfs:label ?info } } } GROUP BY ?label ?score ?id ORDER BY DESC(?score)" \
	--data-urlencode timeout=$(QLEVER_TIMEOUT) \
	--data-urlencode access-token=$(WD_ACCESS_TOKEN) \
	| python scripts/prepare_search_index.py \
	> data/search-index/wikidata-entities/data.tsv
	# wikidata entities small (top 1M entities by sitelinks)
	@mkdir -p data/search-index/wikidata-entities-small/$(SEARCH_INDEX)
	@head -n 1000001 data/search-index/wikidata-entities/data.tsv \
	> data/search-index/wikidata-entities-small/data.tsv
	# wikidata properties
	# https://qlever.cs.uni-freiburg.de/wikidata/dAqs2J
	@mkdir -p data/search-index/wikidata-properties/$(SEARCH_INDEX)
	@curl -s $(WD_URL) -H "Accept: text/tab-separated-values" \
	--data-urlencode query="PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#> PREFIX skos: <http://www.w3.org/2004/02/skos/core#> PREFIX schema: <http://schema.org/> PREFIX wikibase: <http://wikiba.se/ontology#> PREFIX wd: <http://www.wikidata.org/entity/> PREFIX wdt: <http://www.wikidata.org/prop/direct/> SELECT DISTINCT ?label ?score (GROUP_CONCAT(DISTINCT ?alias; SEPARATOR=\";\") AS ?synonyms) ?id (GROUP_CONCAT(DISTINCT ?info; SEPARATOR=\"\t\") AS ?infos) WHERE { { SELECT ?p (COUNT(?p) AS ?score) WHERE { ?s ?p ?o } GROUP BY ?p } ?id wikibase:directClaim ?p . ?id @en@rdfs:label ?label . OPTIONAL { ?id @en@skos:altLabel ?alias } OPTIONAL { { ?id @en@schema:description ?info } UNION { ?id wdt:P1647 ?sub_ . ?sub_ @en@rdfs:label ?info } } } GROUP BY ?label ?score ?id ORDER BY DESC(?score)" \
	--data-urlencode timeout=$(QLEVER_TIMEOUT) \
	--data-urlencode access-token=$(WD_ACCESS_TOKEN) \
	| python scripts/prepare_search_index.py \
	> data/search-index/wikidata-properties/data.tsv

wikidata-search-indices:
	@python scripts/build_search_index.py \
	data/search-index/wikidata-entities/data.tsv \
	data/search-index/wikidata-entities/$(SEARCH_INDEX) \
	--with-mapping \
	--type $(SEARCH_INDEX)
	@python scripts/build_search_index.py \
	data/search-index/wikidata-entities-small/data.tsv \
	data/search-index/wikidata-entities-small/$(SEARCH_INDEX) \
	--with-mapping \
	--type $(SEARCH_INDEX)
	@python scripts/build_search_index.py \
	data/search-index/wikidata-properties/data.tsv \
	data/search-index/wikidata-properties/$(SEARCH_INDEX) \
	--with-mapping \
	--type $(SEARCH_INDEX)

freebase-data:
	# freebase entities
	# https://qlever.cs.uni-freiburg.de/freebase/s4iS7f
	@mkdir -p data/search-index/freebase-entities/$(SEARCH_INDEX)
	@curl -s $(FB_URL) -H "Accept: text/tab-separated-values" \
	--data-urlencode query="PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#> PREFIX fb: <http://rdf.freebase.com/ns/> PREFIX skos: <http://www.w3.org/2004/02/skos/core#> PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#> SELECT DISTINCT ?label ?score (GROUP_CONCAT(DISTINCT ?alias; SEPARATOR=\";\") AS ?synonyms) ?id (GROUP_CONCAT(DISTINCT ?info; SEPARATOR=\"\t\") AS ?infos) WHERE { { SELECT ?id (COUNT(?p) AS ?score) WHERE { ?id ?p ?o } GROUP BY ?id } ?id @en@fb:type.object.name ?label . OPTIONAL { ?id @en@fb:common.topic.alias ?alias } OPTIONAL { { ?id @en@fb:common.topic.description ?info } UNION { ?id fb:common.topic.notable_types ?notable_ . ?notable_ @en@fb:type.object.name ?info } } } GROUP BY ?label ?score ?id ORDER BY DESC(?score)" \
	--data-urlencode timeout=$(QLEVER_TIMEOUT) \
	--data-urlencode access-token=$(FB_ACCESS_TOKEN) \
	| python scripts/prepare_search_index.py \
	> data/search-index/freebase-entities/data.tsv
	# freebase properties
	# https://qlever.cs.uni-freiburg.de/freebase/xfgfFv
	@mkdir -p data/search-index/freebase-properties/$(SEARCH_INDEX)
	@curl -s $(FB_URL) -H "Accept: text/tab-separated-values" \
	--data-urlencode query="PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#> PREFIX fb: <http://rdf.freebase.com/ns/> PREFIX skos: <http://www.w3.org/2004/02/skos/core#> PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#> SELECT DISTINCT ?label ?score (GROUP_CONCAT(DISTINCT ?alias; SEPARATOR=\";\") AS ?synonyms) ?id (GROUP_CONCAT(DISTINCT ?info; SEPARATOR=\"\t\") AS ?infos) WHERE { { SELECT ?id (COUNT(?id) AS ?score) WHERE { ?s ?id ?o } GROUP BY ?id } ?id @en@fb:type.object.name ?label . ?id fb:type.object.type fb:type.property . OPTIONAL { ?id @en@fb:common.topic.alias ?alias } OPTIONAL { { ?id fb:type.property.schema ?schema_ . ?schema_ @en@fb:type.object.name ?info } UNION { ?id fb:type.property.expected_type ?type_ . ?type_ @en@rdfs:label ?info } } } GROUP BY ?label ?score ?id ORDER BY DESC(?score)" \
	--data-urlencode timeout=$(QLEVER_TIMEOUT) \
	--data-urlencode access-token=$(FB_ACCESS_TOKEN) \
	| python scripts/prepare_search_index.py \
	> data/search-index/freebase-properties/data.tsv

freebase-search-indices:
	@python scripts/build_search_index.py \
	data/search-index/freebase-entities/data.tsv \
	data/search-index/freebase-entities/$(SEARCH_INDEX) \
	--with-mapping \
	--type $(SEARCH_INDEX)
	@python scripts/build_search_index.py \
	data/search-index/freebase-properties/data.tsv \
	data/search-index/freebase-properties/$(SEARCH_INDEX) \
	--with-mapping \
	--type $(SEARCH_INDEX)

dbpedia-data:
	# dbpedia entities
	# https://qlever.cs.uni-freiburg.de/dbpedia/Y3PXkL
	@mkdir -p data/search-index/dbpedia-entities/$(SEARCH_INDEX)
	@curl -s $(DBPEDIA_URL) -H "Accept: text/tab-separated-values" \
	--data-urlencode query="PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#> PREFIX dbo: <http://dbpedia.org/ontology/> PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#> PREFIX dbp: <http://dbpedia.org/property/> PREFIX foaf: <http://xmlns.com/foaf/0.1/> SELECT DISTINCT ?label ?score (GROUP_CONCAT(DISTINCT ?alias; SEPARATOR=\";\") AS ?synonyms) ?id (GROUP_CONCAT(DISTINCT ?info; SEPARATOR=\"\t\") AS ?infos) WHERE { { SELECT ?id (COUNT(?id) AS ?score) WHERE { ?id ?p ?o } GROUP BY ?id } ?id @en@rdfs:label ?label . OPTIONAL { { ?id @en@dbp:synonyms ?alias } UNION { ?id @en@dbo:alias ?alias } UNION { ?id @en@dbo:alternativeName ?alias } UNION { ?id @en@foaf:nick ?alias } } OPTIONAL { { ?id @en@rdfs:comment ?info } UNION { ?id rdfs:subClassOf|rdf:type ?type_ . FILTER(STRSTARTS(STR(?type_), \"http://dbpedia.org/ontology/\")) . ?type_ @en@rdfs:label ?type } } } GROUP BY ?label ?score ?id ORDER BY DESC(?score)" \
	--data-urlencode timeout=$(QLEVER_TIMEOUT) \
	--data-urlencode access-token=$(DBPEDIA_ACCESS_TOKEN) \
	| python scripts/prepare_search_index.py \
	> data/search-index/dbpedia-entities/data.tsv
	# dbpedia properties
	# https://qlever.cs.uni-freiburg.de/dbpedia/Yo0wIh
	@mkdir -p data/search-index/dbpedia-properties/$(SEARCH_INDEX)
	@curl -s $(DBPEDIA_URL) -H "Accept: text/tab-separated-values" \
	--data-urlencode query="PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#> PREFIX dbo: <http://dbpedia.org/ontology/> PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#> PREFIX dbp: <http://dbpedia.org/property/> PREFIX foaf: <http://xmlns.com/foaf/0.1/> SELECT DISTINCT ?label ?score (GROUP_CONCAT(DISTINCT ?alias; SEPARATOR=\";\") AS ?synonyms) ?id (GROUP_CONCAT(DISTINCT ?info; SEPARATOR=\"\t\") AS ?infos) WHERE { { SELECT ?id (COUNT(?id) AS ?score) WHERE { ?s ?id ?o } GROUP BY ?id } ?id @en@rdfs:label ?label . ?id rdf:type rdf:Property . BIND(\"\" AS ?alias) OPTIONAL { { ?id @en@rdfs:comment ?info } UNION { ?id rdfs:subPropertyOf ?type_ . ?type_ @en@rdfs:label ?info } UNION { ?id rdfs:range ?range_ . ?range_ @en@rdfs:label ?info } UNION { ?id rdfs:domain ?domain_ . ?domain_ @en@rdfs:label ?info } } } GROUP BY ?label ?score ?id ORDER BY DESC(?score)" \
	--data-urlencode timeout=$(QLEVER_TIMEOUT) \
	--data-urlencode access-token=$(DBPEDIA_ACCESS_TOKEN) \
	| python scripts/prepare_search_index.py \
	> data/search-index/dbpedia-properties/data.tsv

dbpedia-search-indices:
	@python scripts/build_search_index.py \
	data/search-index/dbpedia-entities/data.tsv \
	data/search-index/dbpedia-entities/$(SEARCH_INDEX) \
	--with-mapping \
	--type $(SEARCH_INDEX)
	@python scripts/build_search_index.py \
	data/search-index/dbpedia-properties/data.tsv \
	data/search-index/dbpedia-properties/$(SEARCH_INDEX) \
	--with-mapping \
	--type $(SEARCH_INDEX)
