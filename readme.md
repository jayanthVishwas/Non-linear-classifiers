# Implementation of "A Fast and Accurate Dependency Parser using Neural Networks "

Implementation of neural network model of the dependency parser from the paper A Fast and Accurate Dependency Parser using Neural Networks. The model is trained on the features obtained from the arc-eager parse of the labeled dependency trees.

## Arc-Eager Parse
In the arc standard transition system, the oracle waits for a particular word to obtain all its child
dependencies, before assigning it a parent word
through right arc. This is necessary as the transitions in the arc standard system remove the dependent word from the stack as soon as it is attached
to a parent. On the contrary, the arc eager transition system adds a right dependency arc as soon
as possible, without removing the dependent word
from consideration in future states. Instead, it introduces a separate reduce operator, that pops the
topmost element from the stack.

At each step, the arc-standard algorithm performs one of three operations — LEFT-ARC, RIGHT-ARC, SHIFT or REDUCE — depending upon the dependency tree of the sentence. The operations change the buffer and stack contents, and adds new dependency edges between nodes of the stack.


## Dependencies:

- Numpy
- Pickle
- PyTorch

## Usage

1. Run preparedata.py, which parses the train.orig.conll and dev.orig.conll. It creates features.txt which contain features, arc label and transition.
2. Run train.py, which converts the features.txt into vectors/tensors and trains it based on the neural network designed in model.py 
    
    args = -u "hidden units" -l "learning rate" -b "batch size" -e 2 -i "train data" -v "validation data" -o "train.model"

3. Run parse.py, which parses the sentences in the tree bank using trained model and creates a predicted tree.
    args -m train.model -i dev.orig.conll -o output.conll

4. run "java -cp stanford-parser.jar edu.stanford.nlp.trees.DependencyScoring -g dev.orig.conll -conllx True -s output.conll" to get the label attachment and unlabelled attachment scores.

## Reference

Chen, Danqi, and Christopher D. Manning. "A fast and accurate dependency parser using neural networks." Proceedings of the 2014 conference on empirical methods in natural language processing (EMNLP). 2014.