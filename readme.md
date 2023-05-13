# DNA-2-Vec

This repository contains code files for DNA-2-Vec. The main code files can be found under `src/dna2vec`, while the data is saved under the `data` directory.

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
    - [ ] Add learning rate scheduler
    - [x] Add gradient accumulation
- [ ] Add more data
    - [ ] Add all of the human genome
- [ ] Issues
    - [ ] There might be a memory leak somewhere
