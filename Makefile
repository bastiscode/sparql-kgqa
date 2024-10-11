WD_ENT=data/search-index/wikidata-entities-small
WD_PROP=data/search-index/wikidata-properties

WD_URL=https://qlever.cs.uni-freiburg.de/api/wikidata
WD_ACCESS_TOKEN=null
WD_QUERY_LOG_SOURCE=organic

FB_ENT=data/search-index/freebase-entities
FB_PROP=data/search-index/freebase-properties

FB_URL=https://qlever.cs.uni-freiburg.de/api/freebase
FB_ACCESS_TOKEN=null

DBPEDIA_ENT=data/search-index/dbpedia-entities
DBPEDIA_PROP=data/search-index/dbpedia-properties

DBPEDIA_URL=https://qlever.cs.uni-freiburg.de/api/dbpedia
DBPEDIA_ACCESS_TOKEN=null

DBLP_ENT=data/search-index/dblp-entities
DBLP_PROP=data/search-index/dblp-properties

DBLP_URL=https://qlever.cs.uni-freiburg.de/api/dblp
DBLP_ACCESS_TOKEN=null

NUM_PROCESSES=4

QLEVER_TIMEOUT=1h

SEARCH_INDEX=qgram

OVERWRITE=

all: search-data search-indices other-data wikidata-data freebase-data dbpedia-data dblp-data example-indices

other-data:
	@mkdir -p data/open-hermes-2.5
	@python scripts/prepare_openhermes.py data/open-hermes-2.5

wikidata-data:
	@python scripts/prepare_data2.py \
	--wikidata-simple-questions \
	--output data/wikidata-simplequestions \
	--entities $(WD_ENT) \
	--properties $(WD_PROP) \
	--progress \
	-n $(NUM_PROCESSES) $(OVERWRITE)
	@python scripts/prepare_data2.py \
	--lc-quad2-wikidata \
	--output data/wikidata-lcquad2 \
	--entities $(WD_ENT) \
	--properties $(WD_PROP) \
	--progress \
	-n $(NUM_PROCESSES) $(OVERWRITE)
	@python scripts/prepare_data2.py \
	--qald-10 \
	--output data/wikidata-qald10 \
	--entities $(WD_ENT) \
	--properties $(WD_PROP) \
	--progress \
	-n $(NUM_PROCESSES) $(OVERWRITE)
	@python scripts/prepare_data2.py \
	--qald-7 data/raw/qald-7 \
	--output data/wikidata-qald7 \
	--entities $(WD_ENT) \
	--properties $(WD_PROP) \
	--progress \
	-n $(NUM_PROCESSES) $(OVERWRITE)
	@python scripts/prepare_data2.py \
	--mcwq data/raw/mcwq \
	--output data/wikidata-mcwq \
	--entities $(WD_ENT) \
	--properties $(WD_PROP) \
	--progress \
	-n $(NUM_PROCESSES) $(OVERWRITE)
	@python scripts/prepare_data2.py \
	--wwq data/raw/wikiwebquestions \
	--output data/wikidata-wwq \
	--entities $(WD_ENT) \
	--properties $(WD_PROP) \
	--progress \
	-n $(NUM_PROCESSES) $(OVERWRITE)
	# todo: add kqa pro
	@python scripts/prepare_data2.py \
	--qa-wiki data/raw/qa_wiki/qa_wiki.tsv \
	--output data/wikidata-qa-wiki \
	--entities $(WD_ENT) \
	--properties $(WD_PROP) \
	--progress \
	-n $(NUM_PROCESSES) $(OVERWRITE)
	@python scripts/prepare_data2.py \
	--qlever-wikidata data/raw/qlever_wikidata/data.tsv \
	--output data/wikidata-qlever-wikidata \
	--entities $(WD_ENT) \
	--properties $(WD_PROP) \
	--progress \
	-n $(NUM_PROCESSES) $(OVERWRITE)
	@mkdir -p data/wikidata-query-logs
	@python scripts/prepare_wikidata_query_logs.py \
	--files data/raw/wikidata-query-logs/*.tsv \
	--output-dir data/wikidata-query-logs \
	--entities $(WD_ENT) \
	--properties $(WD_PROP) \
	--$(WD_QUERY_LOG_SOURCE)-only \
	--progress $(OVERWRITE)

freebase-data:
	@python scripts/prepare_data2.py \
	--grail-qa \
	--output data/freebase-grail-qa \
	--entities $(FB_ENT) \
	--properties $(FB_PROP) \
	--progress \
	-n $(NUM_PROCESSES) $(OVERWRITE)
	@python scripts/prepare_data2.py \
	--wqsp \
	--output data/freebase-wqsp \
	--entities $(FB_ENT) \
	--properties $(FB_PROP) \
	--progress \
	-n $(NUM_PROCESSES) $(OVERWRITE)
	@python scripts/prepare_data2.py \
	--cwq \
	--output data/freebase-cwq \
	--entities $(FB_ENT) \
	--properties $(FB_PROP) \
	--progress \
	-n $(NUM_PROCESSES) $(OVERWRITE)
	@python scripts/prepare_data2.py \
	--cfq data/raw/cfq1.1/cfq \
	--output data/freebase-cfq \
	--entities $(FB_ENT) \
	--properties $(FB_PROP) \
	--progress \
	-n $(NUM_PROCESSES) $(OVERWRITE)

dbpedia-data:
	@python scripts/prepare_data2.py \
	--lc-quad1-dbpedia \
	--output data/dbpedia-lcquad1 \
	--entities $(DBPEDIA_ENT) \
	--properties $(DBPEDIA_PROP) \
	--progress \
	-n $(NUM_PROCESSES) $(OVERWRITE)
	@python scripts/prepare_data2.py \
	--qald-9 \
	--output data/dbpedia-qald9 \
	--entities $(DBPEDIA_ENT) \
	--properties $(DBPEDIA_PROP) \
	--progress \
	-n $(NUM_PROCESSES) $(OVERWRITE)

dblp-data:
	@python scripts/prepare_data2.py \
	--dblp-quad \
	--output data/dblp-quad \
	--entities $(DBLP_ENT) \
	--properties $(DBLP_PROP) \
	--progress \
	-n $(NUM_PROCESSES) $(OVERWRITE)

example-indices:
	@for f in $(wildcard data/*/train_raw.jsonl); do \
		python scripts/build_sim_index.py \
		`dirname $$f`/train_examples.index \
		$$f \
		--progress $(OVERWRITE) \
	done

search-data: wikidata-search-data freebase-search-data dbpedia-search-data
search-indices: wikidata-search-indices freebase-search-indices dbpedia-search-indices

wikidata-search-data:
	# wikidata entities
	# https://qlever.cs.uni-freiburg.de/wikidata/AbkI1W
	@mkdir -p data/search-index/wikidata-entities
	@curl -s $(WD_URL) -H "Accept: text/tab-separated-values" \
	--data-urlencode query="PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#> PREFIX skos: <http://www.w3.org/2004/02/skos/core#> PREFIX schema: <http://schema.org/> PREFIX wikibase: <http://wikiba.se/ontology#> PREFIX wd: <http://www.wikidata.org/entity/> PREFIX wdt: <http://www.wikidata.org/prop/direct/> SELECT DISTINCT ?label ?score (GROUP_CONCAT(DISTINCT ?alias; SEPARATOR=\";;;\") AS ?synonyms) ?id (GROUP_CONCAT(DISTINCT ?info; SEPARATOR=\";;;\") AS ?infos) WHERE { { ?id @en@rdfs:label ?label } MINUS { ?id wdt:P31/wdt:P279* wd:Q17442446 } OPTIONAL { ?id @en@skos:altLabel ?alias } OPTIONAL { ?id ^schema:about/wikibase:sitelinks ?score } OPTIONAL { { ?id @en@schema:description ?info } UNION { ?id wdt:P106/@en@rdfs:label ?info } } } GROUP BY ?label ?score ?id ORDER BY DESC(?score)" \
	--data-urlencode timeout=$(QLEVER_TIMEOUT) \
	--data-urlencode access-token=$(WD_ACCESS_TOKEN) \
	| python scripts/prepare_search_index.py \
	> data/search-index/wikidata-entities/data.tsv
	# wikidata entities small (top 10M entities by sitelinks)
	@mkdir -p data/search-index/wikidata-entities-small/$(SEARCH_INDEX)
	@head -n 10000001 data/search-index/wikidata-entities/data.tsv \
	> data/search-index/wikidata-entities-small/data.tsv
	# wikidata properties
	# https://qlever.cs.uni-freiburg.de/wikidata/dAqs2J
	@mkdir -p data/search-index/wikidata-properties
	@curl -s $(WD_URL) -H "Accept: text/tab-separated-values" \
	--data-urlencode query="PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#> PREFIX skos: <http://www.w3.org/2004/02/skos/core#> PREFIX schema: <http://schema.org/> PREFIX wikibase: <http://wikiba.se/ontology#> PREFIX wd: <http://www.wikidata.org/entity/> PREFIX wdt: <http://www.wikidata.org/prop/direct/> SELECT DISTINCT ?label ?score (GROUP_CONCAT(DISTINCT ?alias; SEPARATOR=\";;;\") AS ?synonyms) ?id (GROUP_CONCAT(DISTINCT ?info; SEPARATOR=\";;;\") AS ?infos) WHERE { { SELECT ?p (COUNT(?p) AS ?score) WHERE { ?s ?p ?o } GROUP BY ?p } ?id wikibase:directClaim ?p . ?id @en@rdfs:label ?label . OPTIONAL { ?id @en@skos:altLabel ?alias } OPTIONAL { { ?id @en@schema:description ?info } UNION { ?id wdt:P1647 ?sub_ . ?sub_ @en@rdfs:label ?info } } } GROUP BY ?label ?score ?id ORDER BY DESC(?score)" \
	--data-urlencode timeout=$(QLEVER_TIMEOUT) \
	--data-urlencode access-token=$(WD_ACCESS_TOKEN) \
	| python scripts/prepare_search_index.py \
	> data/search-index/wikidata-properties/data.tsv

wikidata-search-indices:
	@mkdir -p data/search-index/wikidata-entities/$(SEARCH_INDEX)
	@python scripts/build_search_index.py \
	data/search-index/wikidata-entities/data.tsv \
	data/search-index/wikidata-entities/$(SEARCH_INDEX) \
	--with-mapping \
	--type $(SEARCH_INDEX)
	@mkdir -p data/search-index/wikidata-entities-small/$(SEARCH_INDEX)
	@python scripts/build_search_index.py \
	data/search-index/wikidata-entities-small/data.tsv \
	data/search-index/wikidata-entities-small/$(SEARCH_INDEX) \
	--with-mapping \
	--type $(SEARCH_INDEX)
	@mkdir -p data/search-index/wikidata-properties/$(SEARCH_INDEX)
	@python scripts/build_search_index.py \
	data/search-index/wikidata-properties/data.tsv \
	data/search-index/wikidata-properties/$(SEARCH_INDEX) \
	--with-mapping \
	--type $(SEARCH_INDEX)

freebase-search-data:
	# freebase entities
	# https://qlever.cs.uni-freiburg.de/freebase/s4iS7f
	@mkdir -p data/search-index/freebase-entities
	@curl -s $(FB_URL) -H "Accept: text/tab-separated-values" \
	--data-urlencode query="PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#> PREFIX fb: <http://rdf.freebase.com/ns/> PREFIX skos: <http://www.w3.org/2004/02/skos/core#> PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#> SELECT DISTINCT ?label ?score (GROUP_CONCAT(DISTINCT ?alias; SEPARATOR=\";;;\") AS ?synonyms) ?id (GROUP_CONCAT(DISTINCT ?info; SEPARATOR=\";;;\") AS ?infos) WHERE { { SELECT ?id (COUNT(?p) AS ?score) WHERE { ?id ?p ?o } GROUP BY ?id } ?id @en@fb:type.object.name ?label . OPTIONAL { ?id @en@fb:common.topic.alias ?alias } OPTIONAL { { ?id @en@fb:common.topic.description ?info } UNION { ?id fb:common.topic.notable_types ?notable_ . ?notable_ @en@fb:type.object.name ?info } } } GROUP BY ?label ?score ?id ORDER BY DESC(?score)" \
	--data-urlencode timeout=$(QLEVER_TIMEOUT) \
	--data-urlencode access-token=$(FB_ACCESS_TOKEN) \
	| python scripts/prepare_search_index.py \
	> data/search-index/freebase-entities/data.tsv
	# freebase properties
	# https://qlever.cs.uni-freiburg.de/freebase/xfgfFv
	@mkdir -p data/search-index/freebase-properties
	@curl -s $(FB_URL) -H "Accept: text/tab-separated-values" \
	--data-urlencode query="PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#> PREFIX fb: <http://rdf.freebase.com/ns/> PREFIX skos: <http://www.w3.org/2004/02/skos/core#> PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#> SELECT DISTINCT ?label ?score (GROUP_CONCAT(DISTINCT ?alias; SEPARATOR=\";;;\") AS ?synonyms) ?id (GROUP_CONCAT(DISTINCT ?info; SEPARATOR=\";;;\") AS ?infos) WHERE { { SELECT ?id (COUNT(?id) AS ?score) WHERE { ?s ?id ?o } GROUP BY ?id } ?id @en@fb:type.object.name ?label . ?id fb:type.object.type fb:type.property . OPTIONAL { ?id @en@fb:common.topic.alias ?alias } OPTIONAL { { ?id fb:type.property.schema ?schema_ . ?schema_ @en@fb:type.object.name ?info } UNION { ?id fb:type.property.expected_type ?type_ . ?type_ @en@rdfs:label ?info } } } GROUP BY ?label ?score ?id ORDER BY DESC(?score)" \
	--data-urlencode timeout=$(QLEVER_TIMEOUT) \
	--data-urlencode access-token=$(FB_ACCESS_TOKEN) \
	| python scripts/prepare_search_index.py \
	> data/search-index/freebase-properties/data.tsv

freebase-search-indices:
	@mkdir -p data/search-index/freebase-entities/$(SEARCH_INDEX)
	@python scripts/build_search_index.py \
	data/search-index/freebase-entities/data.tsv \
	data/search-index/freebase-entities/$(SEARCH_INDEX) \
	--with-mapping \
	--type $(SEARCH_INDEX)
	@mkdir -p data/search-index/freebase-properties/$(SEARCH_INDEX)
	@python scripts/build_search_index.py \
	data/search-index/freebase-properties/data.tsv \
	data/search-index/freebase-properties/$(SEARCH_INDEX) \
	--with-mapping \
	--type $(SEARCH_INDEX)

dbpedia-search-data:
	# dbpedia entities
	# https://qlever.cs.uni-freiburg.de/dbpedia/Y3PXkL
	@mkdir -p data/search-index/dbpedia-entities
	@curl -s $(DBPEDIA_URL) -H "Accept: text/tab-separated-values" \
	--data-urlencode query="PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#> PREFIX dbo: <http://dbpedia.org/ontology/> PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#> PREFIX dbp: <http://dbpedia.org/property/> PREFIX foaf: <http://xmlns.com/foaf/0.1/> SELECT DISTINCT ?label ?score (GROUP_CONCAT(DISTINCT ?alias; SEPARATOR=\";;;\") AS ?synonyms) ?id (GROUP_CONCAT(DISTINCT ?info; SEPARATOR=\";;;\") AS ?infos) WHERE { { SELECT ?id (COUNT(?id) AS ?score) WHERE { ?id ?p ?o } GROUP BY ?id } ?id @en@rdfs:label ?label . OPTIONAL { { ?id @en@dbp:synonyms ?alias } UNION { ?id @en@dbo:alias ?alias } UNION { ?id @en@dbo:alternativeName ?alias } UNION { ?id @en@foaf:nick ?alias } } OPTIONAL { { ?id @en@rdfs:comment ?info } UNION { ?id rdfs:subClassOf|rdf:type ?type_ . FILTER(STRSTARTS(STR(?type_), \"http://dbpedia.org/ontology/\")) . ?type_ @en@rdfs:label ?type } } } GROUP BY ?label ?score ?id ORDER BY DESC(?score)" \
	--data-urlencode timeout=$(QLEVER_TIMEOUT) \
	--data-urlencode access-token=$(DBPEDIA_ACCESS_TOKEN) \
	| python scripts/prepare_search_index.py \
	> data/search-index/dbpedia-entities/data.tsv
	# dbpedia properties
	# https://qlever.cs.uni-freiburg.de/dbpedia/Yo0wIh
	@mkdir -p data/search-index/dbpedia-properties
	@curl -s $(DBPEDIA_URL) -H "Accept: text/tab-separated-values" \
	--data-urlencode query="PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#> PREFIX dbo: <http://dbpedia.org/ontology/> PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#> PREFIX dbp: <http://dbpedia.org/property/> PREFIX foaf: <http://xmlns.com/foaf/0.1/> SELECT DISTINCT ?label ?score (GROUP_CONCAT(DISTINCT ?alias; SEPARATOR=\";;;\") AS ?synonyms) ?id (GROUP_CONCAT(DISTINCT ?info; SEPARATOR=\";;;\") AS ?infos) WHERE { { SELECT ?id (COUNT(?id) AS ?score) WHERE { ?s ?id ?o } GROUP BY ?id } ?id @en@rdfs:label ?label . ?id rdf:type rdf:Property . BIND(\"\" AS ?alias) OPTIONAL { { ?id @en@rdfs:comment ?info } UNION { ?id rdfs:subPropertyOf ?type_ . ?type_ @en@rdfs:label ?info } UNION { ?id rdfs:range ?range_ . ?range_ @en@rdfs:label ?info } UNION { ?id rdfs:domain ?domain_ . ?domain_ @en@rdfs:label ?info } } } GROUP BY ?label ?score ?id ORDER BY DESC(?score)" \
	--data-urlencode timeout=$(QLEVER_TIMEOUT) \
	--data-urlencode access-token=$(DBPEDIA_ACCESS_TOKEN) \
	| python scripts/prepare_search_index.py \
	> data/search-index/dbpedia-properties/data.tsv

dbpedia-search-indices:
	@mkdir -p data/search-index/dbpedia-entities/$(SEARCH_INDEX)
	@python scripts/build_search_index.py \
	data/search-index/dbpedia-entities/data.tsv \
	data/search-index/dbpedia-entities/$(SEARCH_INDEX) \
	--with-mapping \
	--type $(SEARCH_INDEX)
	@mkdir -p data/search-index/dbpedia-properties/$(SEARCH_INDEX)
	@python scripts/build_search_index.py \
	data/search-index/dbpedia-properties/data.tsv \
	data/search-index/dbpedia-properties/$(SEARCH_INDEX) \
	--with-mapping \
	--type $(SEARCH_INDEX)

dblp-search-data:
	# dblp entities
	# https://qlever.cs.uni-freiburg.de/dblp/Mepe2l
	@mkdir -p data/search-index/dblp-entities
	@curl -s $(DBLP_URL) -H "Accept: text/tab-separated-values" \
	--data-urlencode query="PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#> PREFIX dbo: <http://dbpedia.org/ontology/> PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#> PREFIX dbp: <http://dbpedia.org/property/> PREFIX foaf: <http://xmlns.com/foaf/0.1/> SELECT DISTINCT (SAMPLE(COALESCE(?pname, ?lab)) AS ?label) ?score (GROUP_CONCAT(DISTINCT ?alias; SEPARATOR=\";;;\") AS ?synonyms) ?id (GROUP_CONCAT(DISTINCT ?info; SEPARATOR=\";;;\") AS ?infos) WHERE { { SELECT ?id (COUNT(?id) AS ?score) WHERE { ?id ?p ?o } GROUP BY ?id } OPTIONAL { ?id <https://dblp.org/rdf/schema#primaryCreatorName> ?pname } ?id rdfs:label ?lab . BIND(\"\" AS ?alias) OPTIONAL { { ?id rdfs:comment ?info } UNION { ?id rdfs:subClassOf|rdf:type ?type_ . FILTER(STRSTARTS(STR(?type_), \"https://dblp.org/\")) . ?type_ rdfs:label ?type } } } GROUP BY ?score ?id ORDER BY DESC(?score)" \
	--data-urlencode timeout=$(QLEVER_TIMEOUT) \
	--data-urlencode access-token=$(DBLP_ACCESS_TOKEN) \
	| python scripts/prepare_search_index.py \
	> data/search-index/dblp-entities/data.tsv
	# dblp properties
	# https://qlever.cs.uni-freiburg.de/dblp/HwqyBj
	@mkdir -p data/search-index/dblp-properties
	@curl -s $(DBLP_URL) -H "Accept: text/tab-separated-values" \
	--data-urlencode query="PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#> PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#> SELECT DISTINCT ?label ?score (GROUP_CONCAT(DISTINCT ?alias; SEPARATOR=\";;;\") AS ?synonyms) ?id (GROUP_CONCAT(DISTINCT ?info; SEPARATOR=\";;;\") AS ?infos) WHERE { { SELECT ?id (COUNT(?id) AS ?score) WHERE { ?s ?id ?o } GROUP BY ?id } ?id rdfs:label ?label . ?id rdf:type rdf:Property . BIND(\"\" AS ?alias) OPTIONAL { { ?id rdfs:comment ?info } UNION { ?id rdfs:subPropertyOf ?type_ . ?type_ rdfs:label ?info } UNION { ?id rdfs:range ?range_ . ?range_ rdfs:label ?info } UNION { ?id rdfs:domain ?domain_ . ?domain_ rdfs:label ?info } } } GROUP BY ?label ?score ?id ORDER BY DESC(?score)" \
	--data-urlencode timeout=$(QLEVER_TIMEOUT) \
	--data-urlencode access-token=$(DBLP_ACCESS_TOKEN) \
	| python scripts/prepare_search_index.py \
	--dblp-properties \
	> data/search-index/dblp-properties/data.tsv

dblp-search-indices:
	@mkdir -p data/search-index/dblp-entities/$(SEARCH_INDEX)
	@python scripts/build_search_index.py \
	data/search-index/dblp-entities/data.tsv \
	data/search-index/dblp-entities/$(SEARCH_INDEX) \
	--with-mapping \
	--type $(SEARCH_INDEX)
	@mkdir -p data/search-index/dblp-properties/$(SEARCH_INDEX)
	@python scripts/build_search_index.py \
	data/search-index/dblp-properties/data.tsv \
	data/search-index/dblp-properties/$(SEARCH_INDEX) \
	--with-mapping \
	--type $(SEARCH_INDEX)

orkg-search-data:
	# orkg entities
	# https://qlever.cs.uni-freiburg.de/orkg/TnqXnv
	@mkdir -p data/search-index/orkg-entities
	@curl -s $(ORKG_URL) -H "Accept: text/tab-separated-values" \
	--data-urlencode query="PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#> PREFIX dbo: <http://dbpedia.org/ontology/> PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#> PREFIX dbp: <http://dbpedia.org/property/> PREFIX foaf: <http://xmlns.com/foaf/0.1/> PREFIX orkgp: <http://orkg.org/orkg/predicate/> SELECT DISTINCT ?label ?score (GROUP_CONCAT(DISTINCT ?alias; SEPARATOR=\";;;\") AS ?synonyms) ?id (GROUP_CONCAT(DISTINCT ?info; SEPARATOR=\";;;\") AS ?infos) WHERE { { SELECT ?id (COUNT(?id) AS ?score) WHERE { ?id ?p ?o } GROUP BY ?id } ?id rdfs:label ?label . BIND(\"\" AS ?alias) OPTIONAL { { ?id rdf:type/rdfs:label ?info } UNION { ?id orkgp:description ?info } } } GROUP BY ?label ?score ?id ORDER BY DESC(?score)" \
	--data-urlencode timeout=$(QLEVER_TIMEOUT) \
	--data-urlencode access-token=$(ORKG_ACCESS_TOKEN) \
	| python scripts/prepare_search_index.py \
	> data/search-index/orkg-entities/data.tsv
	# orkg properties
	# https://qlever.cs.uni-freiburg.de/orkg/SxQBdo
	@mkdir -p data/search-index/orkg-properties
	@curl -s $(ORKG_URL) -H "Accept: text/tab-separated-values" \
	--data-urlencode query="PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#> PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#> PREFIX orkgc: <http://orkg.org/orkg/class/> SELECT DISTINCT ?label ?score (GROUP_CONCAT(DISTINCT ?alias; SEPARATOR=\";;;\") AS ?synonyms) ?id (GROUP_CONCAT(DISTINCT ?info; SEPARATOR=\";;;\") AS ?infos) WHERE { { SELECT ?id (COUNT(?id) AS ?score) WHERE { ?s ?id ?o } GROUP BY ?id } ?id rdfs:label ?label . ?id rdf:type orkgc:Predicate . BIND(\"\" AS ?alias) BIND(\"\" AS ?info) } GROUP BY ?label ?score ?id ORDER BY DESC(?score)" \
	--data-urlencode timeout=$(QLEVER_TIMEOUT) \
	--data-urlencode access-token=$(ORKG_ACCESS_TOKEN) \
	| python scripts/prepare_search_index.py \
	> data/search-index/orkg-properties/data.tsv

orkg-search-indices:
	@mkdir -p data/search-index/orkg-entities/$(SEARCH_INDEX)
	@python scripts/build_search_index.py \
	data/search-index/orkg-entities/data.tsv \
	data/search-index/orkg-entities/$(SEARCH_INDEX) \
	--with-mapping \
	--type $(SEARCH_INDEX)
	@mkdir -p data/search-index/orkg-properties/$(SEARCH_INDEX)
	@python scripts/build_search_index.py \
	data/search-index/orkg-properties/data.tsv \
	data/search-index/orkg-properties/$(SEARCH_INDEX) \
	--with-mapping \
	--type $(SEARCH_INDEX)
