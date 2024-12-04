# Rapid Response Antimicrobial Peptide Design Strategy Driven by Meta-Learning for Emerging Drug-Resistant Pathogens

## Abstract
Purpose:Antimicrobial resistance (AMR) constitues a pressing public health crisis, posing a substantial threat to global health security. In order to swiftly respond to and control the spread of emerging drug-resistant bacteria at the onset of their proliferation, our aim is to develop a  Rapid Response Antimicrobial Peptide (AMP) design strategy (RR-ADS).
Methods: Our approach harnesses the combined power of meta-learning and reinforcement learning to enable the model to achieve robust generalization with a minimal dataset, thereby expediting the response process. This strategy ensures the rapid and efficient generation of AMPs that exhibit superior biocompatibility and are specifically tailored to combat drug-resistant pathogens.
Results: Our model has achieved satisfactory results across multiple evaluation metrics, demonstrating the capability to accurately identify and generate AMPs targeted against drug-resistant bacteria with minimal sample sizes. Building on this, we completed the generation and validation of AMPs against multidrug-resistant Acinetobacter baumannii within two weeks, with a positive rate of 93.3%.
Conclusion: RR-ADS has effectively demonstrated the potential of meta-learning in tasks involving bioactive peptides and holds promise as an effective alternative measure to address infectious disease public health emergencies.

### Description of the document catalogue
eg:

```
filetree 
├── README.md
├── Meta_dataset.py
├── args.py
├── model
│  ├── Meta_loop.py
│  ├── RNN.py
│  ├── RNN_Scoring.py
│  ├── Transformer_module.py
├── Pep
│  ├── added_tokens.json
│  ├── config.json
│  ├── merges.txt
│  ├── special_tokens_map.json
│  ├── tokenizer_config.json
│  ├── voc.json
│  ├── vocab.json
├── train
│  ├── Meta_finetune.py
│  ├── Meta_train.py
│  ├── RNN_Reinforcement.py
│  ├── RNN_train.py
│  ├── Transformer_train.py
├── utils
│  ├── RNN_utils.py
```

### Requirements
```
scikit-learn            1.3.0
pytorch                 2.2.1
pytorch-lightning       2.4.0
transformers            4.46.2
tokenizers              0.20.3
tqdm                    4.65.0
```
