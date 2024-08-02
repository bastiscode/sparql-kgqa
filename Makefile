WD_ENT=data/kg-index/wikidata-entities
WD_PROP=data/kg-index/wikidata-properties

QLEVER_TIMEOUT=1h
WD_URL=https://qlever.cs.uni-freiburg.de/api/wikidata
WD_ACCESS_TOKEN=null

ENT_SUFFIX="</kge>"
PROP_SUFFIX="</kgp>"

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

indices:
	@echo "Creating wikidata continuation indices"
	@tu.create_continuation_index \
	--input-file $(WD_PROP)/index.tsv \
	--output-dir data/art-index/wikidata-properties \
	--common-suffix $(PROP_SUFFIX)
	@tu.create_continuation_index \
	--input-file $(WD_ENT)/index.tsv \
	--output-dir data/art-index/wikidata-entities \
	--common-suffix $(ENT_SUFFIX)

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

qgram-indices:
	# wikidata entities
	# https://qlever.cs.uni-freiburg.de/wikidata/0gMAIw
	@mkdir -p data/qgram-index/wikidata-entities
	@curl -s $(WD_URL) -H "Accept: text/tab-separated-values" \
	--data-urlencode query="PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#> PREFIX skos: <http://www.w3.org/2004/02/skos/core#> PREFIX schema: <http://schema.org/> PREFIX wikibase: <http://wikiba.se/ontology#> PREFIX wd: <http://www.wikidata.org/entity/> PREFIX wdt: <http://www.wikidata.org/prop/direct/> SELECT DISTINCT ?label ?score (GROUP_CONCAT(?alias; SEPARATOR=\";\") AS ?synonyms) ?id ?description WHERE { { ?id @en@rdfs:label ?label } MINUS { ?id wdt:P31/wdt:P279* wd:Q17442446 } OPTIONAL { ?id @en@skos:altLabel ?alias } OPTIONAL { ?id ^schema:about/wikibase:sitelinks ?score } OPTIONAL { ?id @en@schema:description ?description } } GROUP BY ?label ?score ?id ?description ORDER BY DESC(?score)" \
	--data-urlencode timeout=$(QLEVER_TIMEOUT) \
	--data-urlencode access-token=$(WD_ACCESS_TOKEN) \
	| python scripts/prepare_qgram_index.py \
	> data/qgram-index/wikidata-entities/data.tsv
	@python scripts/build_qgram_index.py \
	data/qgram-index/wikidata-entities/data.tsv \
	data/qgram-index/wikidata-entities/index.bin \
	--with-mapping --mapping-prefix "<http://www.wikidata.org/entity/"
	# wikidata entities small (top 1M entities by sitelinks)
	@mkdir -p data/qgram-index/wikidata-entities-small
	@head -n 1000001 data/qgram-index/wikidata-entities/data.tsv \
	> data/qgram-index/wikidata-entities-small/data.tsv
	@python scripts/build_qgram_index.py \
	data/qgram-index/wikidata-entities-small/data.tsv \
	data/qgram-index/wikidata-entities-small/index.bin \
	--with-mapping --mapping-prefix "<http://www.wikidata.org/entity/"
	# wikidata entities medium (top 10M entities by sitelinks)
	@mkdir -p data/qgram-index/wikidata-entities-medium
	@head -n 10000001 data/qgram-index/wikidata-entities/data.tsv \
	> data/qgram-index/wikidata-entities-medium/data.tsv
	@python scripts/build_qgram_index.py \
	data/qgram-index/wikidata-entities-medium/data.tsv \
	data/qgram-index/wikidata-entities-medium/index.bin \
	--with-mapping --mapping-prefix "<http://www.wikidata.org/entity/"
	# wikidata properties
	# https://qlever.cs.uni-freiburg.de/wikidata/ablT44
	@mkdir -p data/qgram-index/wikidata-properties
	@curl -s $(WD_URL) -H "Accept: text/tab-separated-values" \
	--data-urlencode query="PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#> PREFIX skos: <http://www.w3.org/2004/02/skos/core#> PREFIX schema: <http://schema.org/> PREFIX wikibase: <http://wikiba.se/ontology#> PREFIX wd: <http://www.wikidata.org/entity/> PREFIX wdt: <http://www.wikidata.org/prop/direct/> SELECT DISTINCT ?label ?score (GROUP_CONCAT(?alias; SEPARATOR=\";\") AS ?synonyms) ?id ?description WHERE { { SELECT ?p (COUNT(?p) AS ?score) WHERE { ?s ?p ?o } GROUP BY ?p } ?id wikibase:directClaim ?p . ?id @en@rdfs:label ?label . OPTIONAL { ?id @en@skos:altLabel ?alias } OPTIONAL { ?id @en@schema:description ?description } } GROUP BY ?label ?score ?id ?description ORDER BY DESC(?score)" \
	--data-urlencode timeout=$(QLEVER_TIMEOUT) \
	--data-urlencode access-token=$(WD_ACCESS_TOKEN) \
	| python scripts/prepare_qgram_index.py \
	> data/qgram-index/wikidata-properties/data.tsv
	@python scripts/build_qgram_index.py \
	data/qgram-index/wikidata-properties/data.tsv \
	data/qgram-index/wikidata-properties/index.bin \
	--with-mapping --mapping-prefix "<http://www.wikidata.org/entity/"
