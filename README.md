## Heterogeneous Memory Network for Generative Question Answering 

The implementation of paper: 
Yao Fu and Yansong Feng, _Natural Answer Generation from Heterogeneous Memory_, NAACL 2018 

The blog of this paper can be found [here](https://francix.github.io/NaturalAnswer.html).

The Cumulative Attention Mechanism is implemented in model.py (Actually it is just one more line of code, if you implement attention mechanism as a function) -- so this mechanism is highly recommended to improve your model performance. 

In addition to the original QA dataset, it has been tested on many data-to-text generation tasks after the paper was published, and achieved very good performance. 

The [Neural Checklist](https://github.com/uwnlp/neural-checklist) model was originally implemented in Lua Torch, here we reimplemented in tensorflow. 

(Sorry that the model is a bit of messy, this was my first experience in DL4NLP)
