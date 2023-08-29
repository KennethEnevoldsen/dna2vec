# DNA-ESA

This repository contains code files for DNA-ESA. The main code files can be found under `src/dna2vec`, while the data is saved under the `data` directory.

## Installation

```
git clone {repo url}
cd {repo}
pip install -e .
```

Furthermore, if you wish to simulate reads (used for training and evaluation) you will have to install [ART](https://www.niehs.nih.gov/research/resources/software/biostatistics/art/index.cfm). We do this using: 

```bash
apt install art-nextgen-simulation-tools
```


## Training the model

To train the model, simply run:

```
python train.py
```

The training script includes the config object. The config object contains all the hyperparameters for the model.

## Running the Splicer

To run the Splicer, use the following command:

```
python src/dna2vec/splicer.py --N 1000000 --mode random --datapath data/
```

## Running the Pinecone-Supported Encoder and Test

### Test

To run the Pinecone-supported encoder and test, use the following command:

```
python src/dna2vec/pinecone_store.py --inputpath /home/pholur/dna-2-vec/data/subsequences_sample.txt --reupload n --drop n
```

### Upsert

To run the Pinecone-supported encoder and upsert, use the following command:

```
python src/dna2vec/pinecone_store.py --inputpath /home/pholur/dna-2-vec/data/subsequences_sample.txt --reupload y --drop n
```

### Drop

To run the Pinecone-supported encoder and drop, use the following command:

```
python src/dna2vec/pinecone_store.py --inputpath /home/pholur/dna-2-vec/data/subsequences_sample.txt --reupload n --drop y
```

# Ideas to try out


- [ ] Training optimizations
    - [x] Add gradient clipping
    - [x] Add learning rate scheduler
    - [x] Add gradient accumulation
- [ ] Add more data
    - [x] Add all of the human genome
- [ ] Issues

- [ ] fix loss to mean loss? shouldn't change with batch size
