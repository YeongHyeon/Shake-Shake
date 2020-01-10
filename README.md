Shake-Shake Regularization
=====

## Concept
<div align="center">
  <img src="./figures/shake.png" width="600">  
  <p>The concept of Shake-Shake Regularization [1].</p>
</div>

## Performance

| |Basic CNN|Shake-Shake CNN|
|:---|:---:|:---:|
|Accuracy|0.99340|0.xxxxx|
|Precision|0.99339|0.xxxxx|
|Recall|0.99329|0.xxxxx|
|F1-Score|0.99334|0.xxxxx|

### Basic CNN
```
Confusion Matrix
[[ 979    0    0    0    0    0    0    1    0    0]
 [   0 1132    0    1    0    0    1    1    0    0]
 [   0    0 1029    0    0    0    0    3    0    0]
 [   0    0    1 1006    0    3    0    0    0    0]
 [   0    0    1    0  975    0    2    0    0    4]
 [   1    0    0    7    0  882    1    0    0    1]
 [   4    2    0    0    0    1  950    0    1    0]
 [   1    3    3    2    0    0    0 1018    1    0]
 [   3    0    1    1    0    1    0    0  966    2]
 [   0    0    0    1    6    2    0    3    0  997]]
Class-0 | Precision: 0.99089, Recall: 0.99898, F1-Score: 0.99492
Class-1 | Precision: 0.99560, Recall: 0.99736, F1-Score: 0.99648
Class-2 | Precision: 0.99420, Recall: 0.99709, F1-Score: 0.99565
Class-3 | Precision: 0.98821, Recall: 0.99604, F1-Score: 0.99211
Class-4 | Precision: 0.99388, Recall: 0.99287, F1-Score: 0.99338
Class-5 | Precision: 0.99213, Recall: 0.98879, F1-Score: 0.99045
Class-6 | Precision: 0.99581, Recall: 0.99165, F1-Score: 0.99372
Class-7 | Precision: 0.99220, Recall: 0.99027, F1-Score: 0.99124
Class-8 | Precision: 0.99793, Recall: 0.99179, F1-Score: 0.99485
Class-9 | Precision: 0.99303, Recall: 0.98811, F1-Score: 0.99056

Total | Accuracy: 0.99340, Precision: 0.99339, Recall: 0.99329, F1-Score: 0.99334
```

### Shake-Shake CNN (Basic CNN + Shake-Shake Regularization)
```
Confusion Matrix

```

## Requirements
* Python 3.6.8  
* Tensorflow 1.14.0  
* Numpy 1.17.1  
* Matplotlib 3.1.1  

## Reference
[1] Gastaldi, Xavier. <a href="https://arxiv.org/abs/1705.07485">Shake-Shake Regularization.</a> arXiv preprint arXiv:1705.07485 (2017).
