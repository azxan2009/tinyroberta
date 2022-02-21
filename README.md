---
language: en
datasets:
- squad_v2
license: cc-by-4.0
---

# tinyroberta-squad2

## Overview
**Language model:** tinyroberta-squad2  
**Language:** English  
**Downstream-task:** Extractive QA  
**Training data:** SQuAD 2.0  
**Eval data:** SQuAD 2.0  
**Code:**  
**Infrastructure**: 4x Tesla v100

## Hyperparameters

```
batch_size = 96
n_epochs = 4
base_LM_model = "deepset/tinyroberta-squad2-step1"
max_seq_len = 384
learning_rate = 3e-5
lr_schedule = LinearWarmup
warmup_proportion = 0.2
doc_stride = 128
max_query_length = 64
distillation_loss_weight = 0.75
temperature = 1.5
teacher = "deepset/robert-large-squad2"
``` 

## Performance
Evaluated on the SQuAD 2.0 dev set with the [official eval script](https://worksheets.codalab.org/rest/bundles/0x6b567e1cf2e041ec80d7098f031c5c9e/contents/blob/).

```
"exact": 78.19422218478901,
"f1": 81.41848853895601,
  
"total": 11873,
"HasAns_exact": 76.51821862348179,
"HasAns_f1": 82.97599770968706,
"HasAns_total": 5928,
"NoAns_exact": 79.86543313709,
"NoAns_f1": 79.86543313709,
"NoAns_total": 5945,
  
"best_exact": 79.34810073275499,
"best_exact_thresh": -4.051729202270508,
"best_f1": 82.26435201691056,
"best_f1_thresh": -4.051729202270508
```

## Usage

### In Transformers
```python
from transformers import AutoModelForQuestionAnswering, AutoTokenizer, pipeline

model_name = "deepset/tinyroberta-squad2"

# a) Get predictions
nlp = pipeline('question-answering', model=model_name, tokenizer=model_name)
QA_input = {
    'question': 'Why is model conversion important?',
    'context': 'The option to convert models between FARM and transformers gives freedom to the user and let people easily switch between frameworks.'
}
res = nlp(QA_input)

# b) Load model & tokenizer
model = AutoModelForQuestionAnswering.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)
```

### In FARM

```python
from farm.modeling.adaptive_model import AdaptiveModel
from farm.modeling.tokenization import Tokenizer
from farm.infer import Inferencer

model_name = "deepset/tinyroberta-squad2"

# a) Get predictions
nlp = Inferencer.load(model_name, task_type="question_answering")
QA_input = [{"questions": ["Why is model conversion important?"],
             "text": "The option to convert models between FARM and transformers gives freedom to the user and let people easily switch between frameworks."}]
res = nlp.inference_from_dicts(dicts=QA_input, rest_api_schema=True)

# b) Load model & tokenizer
model = AdaptiveModel.convert_from_transformers(model_name, device="cpu", task_type="question_answering")
tokenizer = Tokenizer.load(model_name)
```

### In haystack
For doing QA at scale (i.e. many docs instead of single paragraph), you can load the model also in [haystack](https://github.com/deepset-ai/haystack/):
```python
reader = FARMReader(model_name_or_path="deepset/roberta-base-squad2")
# or 
reader = TransformersReader(model_name_or_path="deepset/roberta-base-squad2",tokenizer="deepset/roberta-base-squad2")
```


## Authors
Branden Chan: `branden.chan [at] deepset.ai`
Timo MÃ¶ller: `timo.moeller [at] deepset.ai`
Malte Pietsch: `malte.pietsch [at] deepset.ai`
Tanay Soni: `tanay.soni [at] deepset.ai`
Michel Bartels: `michel.bartels [at] deepset.ai`

## About us
![deepset logo](https://workablehr.s3.amazonaws.com/uploads/account/logo/476306/logo)
We bring NLP to the industry via open source!  
Our focus: Industry specific language models & large scale QA systems.  
  
Some of our work: 
- [German BERT (aka "bert-base-german-cased")](https://deepset.ai/german-bert)
- [GermanQuAD and GermanDPR datasets and models (aka "gelectra-base-germanquad", "gbert-base-germandpr")](https://deepset.ai/germanquad)
- [FARM](https://github.com/deepset-ai/FARM)
- [Haystack](https://github.com/deepset-ai/haystack/)

Get in touch:
[Twitter](https://twitter.com/deepset_ai) | [LinkedIn](https://www.linkedin.com/company/deepset-ai/) | [Slack](https://haystack.deepset.ai/community/join) | [GitHub Discussions](https://github.com/deepset-ai/haystack/discussions) | [Website](https://deepset.ai)

By the way: [we're hiring!](http://www.deepset.ai/jobs) 