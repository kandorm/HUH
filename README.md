# HUH

Implementation of "[Integrating User History into Heterogeneous Graph for Dialogue Act Recognition](https://www.aclweb.org/anthology/2020.coling-main.372.pdf)" (COLING 2020). Some code in this repository is based on the excellent open-source project https://github.com/iYUYUE/CIKM-2019-CDA.

## Requirements

* Python 3 with PyTorch, pandas and sklearn

## Data

* [MRDA](https://github.com/NathanDuran/MRDA-Corpus)
* [MSDialog-Intent](https://ciir.cs.umass.edu/downloads/msdialog/)

## Train

#### 1st-phase

```
python src/run.py train --cd 0.25 --filters 200 --ld 0.3 --lr 0.0001 --lstm_hidden 400
```

#### 2nd-phase

```
python src/run.py train --cd 0.25 --filters 200 --ld 0.3 --lr 0.0001 --lstm_hidden 400 --model ./model/geqybwhcae/ --model_file 5 --usergraph
```

## Test

#### 1st-phase

```
python src/run.py test --model ./model/geqybwhcae/ --model_file 5 --filters 200 --lstm_hidden 400
```

#### 2nd-phase

```
python src/run.py test --filters 200 --lstm_hidden 400 --model ./model/ypciloaimy/ --model_file 2 --usergraph
```

### Tune

#### 1st-phase

```
python src/run.py tune
```

#### 2nd-phase

```
python src/run.py tune --model ./model/ncxxnquefg/ --model_file 4 --usergraph
```

