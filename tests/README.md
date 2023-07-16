## Testbench for DNA2Vec


### Setup
Ensure that your data folder `datapath` contains a subfolder for each section (in our case `chromosome_*`) and within each subfolder there is a `.fasta` file containing the genome (as a `string`) of that subsection.

### Step 1. Collate the data to stage upstream into Pinecone
Use: `stage_upstream.py`. Arguments:
```bash
python stage_upstream 
    --datapath <path_to_data>           % path to subfolders
    --mode_train <train_mode>           % how to parse the FASTA file
    --rawfile <raw_file>                % source FASTA file
    --unit_length <length>              % length of grounded fragment
    --meta <meta_data>                  % identifier string
    --overlap <overlap_value>           % overlap between fragments
    --topath <output_path>              % name of upstream file
    --ntrain <num_train>                % max limit of fragments

```

For example:
```bash
python stage_upstream.py --datapath data/chromosome_2/ --mode_train hard_serialized --rawfile NC_000002.fasta --unit_length 1000 --meta CH2 --overlap 200 --topath floodfill.pkl --ntrain 500000
```



### Step 2. Upstream into Pinecone
Set up [Pinecone](https://app.pinecone.io/organizations/-NUbbjSKn59kR22U_SS6/settings/projects), a cloud-hosted vector store. You will need to subscribe to the **Free > Standard** plan found [here](https://app.pinecone.io/organizations/-NUbbjSKn59kR22U_SS6/settings/billing/plans). But don't worry, you can always delete the store after you are done using it. Simply rerun this step to populate from scratch.

Use: `upsert.py`. **Note**: You will need to change the `api_key` and `environment` variables in `pinecone_store.py` to get your upsert to work. Arguments:
```bash
python upsert.py 
    --recipes <str list of aliases or paths to data dumps (.pkl)> 
    --checkpoints <str list of model checkpoints>
```
List delimiters are semicolons (`;`). Also you can concatenate datastores by using the comma (`,`).
For example:
```bash
python upsert.py --recipes "ch2;ch3;ch2,ch3" --checkpoints "trained-ch2-1000"
```

### Step 3a. Naïve Permutation and Accuracy Evaluation

Ensure that Pinecone instances are running and the data is populated. Else go back to Step 2. To run permutation accuracy computations at scale, run `test_cache_permute.py`. Arguments:
```bash
python test_cache_permute.py 
    --recipes               % <data recipe combinations>
    --checkpoints           % <model checkpoint>
    --mode                  % <permutation mode>
```
The additional argument `mode` specifies the type of permutation that is applied on the sequence.For example:
```bash
python test_cache_permute.py --recipes "ch2,ch3" --checkpoints "trained-ch2-1000" --mode "end deplete"
```

Results corresponding to these evaluations are deposited in `DATA_PATH`. You can visualize the results of these evaluations using the testbench `test_permutes.ipynb`.

### Step 3b. Manifold Visualization

### Step 4. Ablation Studies

