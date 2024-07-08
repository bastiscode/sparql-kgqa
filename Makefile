WD_ENT=data/kg-index/wikidata-entities
WD_PROP=data/kg-index/wikidata-properties

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

