# Cohere-Parallel-Language-Sentence-Alignment

<a target="_blank" href="https://colab.research.google.com/github/lukmanaj/Cohere-Parallel-Language-Sentence-Alignment/blob/main/Cohere_Align_Sentences.ipynb">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
</a>

# Cohere-Align
 
This repo takes two text files in the source and target languages, and returns sentences that are most likely translations of each other.

Before running, create an account on [cohere](https://cohere.com) to get your api key.

Then install cohere, using the following command

```
pip install cohere
```

To align sentences, create two text files, with each line containing a distinct text, for the source and target languages. Afterwards , run the following command.

### Cohere
```
python3 scripts/cohere_align.py \
   --cohere_api_key '<api_key>' \
   -m 'embed-multilingual-v2.0' \
   -s src.txt \
   -t trg.txt \
   -o cohere \
   --retrieval 'nn' \
   --dot \
   --cuda
 ```
There's also a comparison with laser autoencoder for the same files 

### Laser
```
python3 scripts/laser_align.py \
  -s src.txt \
  -t trg.txt \
  -o cohere \
  --src_lang ha \
  --trg_lang en \
  --retrieval 'nn' \
  --dot \
  --cuda
```

where `m` is model name, `s` is source text path, `t` is target text path, `o` is output directory path, and provide the `cuda` option if you have GPU. For more parameters, see the [alignment script](https://github.com/lukmanaj/Cohere-Parallel-Language-Sentence-Alignment/blob/main/scripts/cohere_align.py).

You can also use the [jupyter notebook](https://github.com/lukmanaj/Cohere-Parallel-Language-Sentence-Alignment/blob/main/Cohere_Align_Sentences.ipynb) above to align the sentences.
