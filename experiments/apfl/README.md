# APFL: Adaptive Personalized Federated Learning

[**Paper**](https://doi.org/10.48550/arXiv.2003.13461)

## Experiments

### EMNIST DENSE

```
python3 run_apfl.py -flagfile=apfl.EMNIST_DENSE.flags -root_dir=/tmp/apfl
```

|              | Loss       | Accuracy  |
|--------------|------------|-----------|
| Aggregated   | 0.33864135 | 0.9585129 |
| Best client  | 0.006118   | 1.0       |
| Worst client | 1.7152065  | 0.4       |

Setup
- Macbook Pro 2019
- CPU: 1.4 GHz Quad-Core Intel Core i5
- RAM: 16 GB 2133 MHz LPDDR3

Speed:
- Average round duration: 4.987510 sec.
- 500 rounds: ~42 min.

Parameters:
- Number of communication rounds: 500
- Number of clients per round: 340 (~10%)
- Batch size: 20
- Server optimizer: Adam
- Server learning rate: 1.0
- Client optimizer: SGD
- Client learning rate: 0.001

Other:
- Checkpoint size: 2.52GB

Comment:
The experiment was run on a Macbook Pro 2019 with the hyperparameters outlined above. The setup differs from the experiment in the original paper in two ways. First, the authors of the paper select a subset of 1000 clients with 10% number of clients per round. However, they do not specify on which bases the clients were selected, therefore we run the experiment on all ~3400 clients while keeping the number of clients per round at 10%. The second difference is that we keep a static learning rate, instead of using learning rate decay. Else, we believe the setup matches the one from the paper. The accuracy on the test set is slightly lower than in paper (0.981), however we are confident that a higher accuracy can be achieved by running the algorithm for more rounds.
