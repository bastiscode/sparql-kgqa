## SPARQL Generation

Generate SPARQL from natural language queries with large language models.

### Installation

This project requires Python 3.10 or higher.

#### From PyPI

```bash
pip install ad-freiburg-sparql-kgqa
```

#### From source

```bash
git clone git@github.com:ad-freiburg/sparql-kgqa.git
cd sparql-kgqa
pip install -e .

```

### Usage

#### From Python

#### From command line

After installation the command `sparql-kgqa` is available in your python environment. 
It lets you use the sparql generation models directly from the command line.
Below are examples of how to use `sparql-kgqa`. See `sparql-kgqa -h` for all options.

```bash
# print version
sparql-kgqa -v

# list available models
sparql-kgqa -l

# by default sparql-kgqa tries to read stdin, generate sparql for the input
# line by line and prints the completed lines to stdout
# therefore, you can for example use sparql generation with pipes
echo "who is chancellor of Germany?" | sparql-kgqa
cat "path/to/input/file.txt" | sparql-kgqa > output.txt

# generate sparql for a given input
sparql-kgqa -p "who is chancellor of Germany?"

# generate sparql for a given input file line by line
sparql-kgqa -f path/to/input/file.txt
# optionally specify an output file path where the sparql queries are saved
sparql-kgqa -f path/to/input/file.txt -o output.txt

# start an interactive sparql generation session
sparql-kgqa -i

# add knowledge graphs to the sparql generation process
# (multiple knowledge graphs possible by passing multiple -kg arguments)
sparql-kgqa -p "who is chancellor of Germany?" -kg "path/to/entities.bin" "path/to/relations.bin" "kg_name"

# start a sparql generation server with the following endpoints:
### /models [GET] --> output: available models as json 
### /info [GET] --> output: info about backend as json
### /live [WS] websocket endpoint for live sparql generation (only single unbatched requests)
sparql-kgqa --server <config_file>

### OPTIONS
### Pass the following flags to the sparql-kgqa command to customize its behaviour
-m <model_name> # use a different sparql generation model than the default one 
--cpu # force execution on CPU, by default a GPU is used if available
--progress # display a progress bar (always on when a file is repaired using -f)
-b <batch_size> # specify a different batch size
-batch-max-tokens <batch_max_tokens> # limit batch by a number of tokens and not by number of samples
-u # do not sort the inputs before completeing
-e <experiment_dir> # specify the path to an experiment directory to load the model from 
                    # (equivalent to SPARQLGenerator.from_experiment(experiment_dir) in Python API)
--force-download # force download of the sparql generation model even if it was already downloaded
--progress # show a progress bar while completing
--report # print a report on the runtime of the model after finishing the completion
```

> Note: When first using `sparql-kgqa` with a pretrained model, the model needs to be downloaded, so depending on
> your internet speed the command might take considerably longer.

> Note: Loading the sparql generation model requires an initial startup time each time you
> invoke the `sparql-kgqa` command. CPU startup time is around 1s, GPU startup time around 3.5s, so for small
> inputs or files you should probably pass the `--cpu` flag to force CPU execution for best performance.

> See [configs/server.yaml](configs/server.yaml) for an exemplary server configuration file.

### Documentation

#### Use pretrained model

If you just want to use this project to generate SPARQL, this is the recommended way.

```python
from sparql_kgqa import SPARQLGenerator

gen = SPARQLGenerator.from_pretrained(
    # pretrained model to load, get all available models from available_models(),
    # if None, loads the default model
    model=None,
    # the device to run the model on
    # ("cuda" by default)
    device="cuda",
    # optional path to a cache directory where downloaded models will be extracted to,
    # if None, we check the env variable SPARQL_GENERATION_CACHE_DIR, if it is not set 
    # we use a default cache directory at <install_path>/api/.cache 
    # (None by default)
    cache_dir=None,
    # optional path to a download directory where pretrained models will be downloaded to,
    # if None, we check the env variable SPARQL_GENERATION_DOWNLOAD_DIR, if it is not set 
    # we use a default download directory at <install_path>/api/.download
    # (None by default)
    download_dir=None,
    # force download of model even if it already exists in download dir
    # (False by default)
    force_download=False
)
```

When used for the first time with the command line interface or Python API the pretrained model will be automatically downloaded. 
However, you can also download our pretrained models first as zip files, put them in a directory on your local drive 
and set `SPARQL_GENERATION_DOWNLOAD_DIR` (or the `download_dir` parameter above) to this directory.

#### Use own model

Once you trained your own model you can use it in the following way.

```python
from sparql_kgqa import SPARQLGenerator

gen = SPARQLGenerator.from_experiment(
    # path to the experiment directory that is created by your training run
    experiment_dir="path/to/experiment_dir",
    # the device to run the model on
    # ("cuda" by default)
    device="cuda"
)
```

### Directory structure

The most important directories you might want to look at are:

```
configs -> (example yaml config files for training and server)
src -> (library code used by this project)
```

### Docker

You can also run this project using docker. Build the image using

`docker build -t sparql-kgqa .`

If you have an older GPU build the image using

`docker build -t sparql-kgqa -f Dockerfile.old .`

By default, the entrypoint is set to the `sparql-kgqa` command, 
so you can use the Docker setup like described [here](#from-command-line) earlier.

You can mount /sparql-kgqa/cache and /sparql-kgqa/download to volumes on your machine, such that
you do not need to download the models every time.

```bash
# generate sparql for a natural language query
docker run sparql-kgqa -p "who is chancellor of Germany?"

# generate sparql for a file
docker run sparql-kgqa -f path/to/file.txt

# start a server
docker run sparql-kgqa --server path/to/config.yaml

# with volumes
docker run -v $(pwd)/.cache:/sparql-kgqa/cache -v $(pwd)/.download:/sparql-kgqa/download \
  sparql-kgqa -p "who is chancellor of Germany?"

# optional parameters recommended when using a GPU:
# --gpus all --ipc=host --ulimit memlock=-1 --ulimit stack=67108864

Note
----
Make sure you have docker version >= 19.03, a nvidia driver
and the nvidia container toolkit installed (see https://github.com/NVIDIA/nvidia-docker)
if you want to run the container with GPU support.
```
