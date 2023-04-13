# ISSTA_2023

The generator of "synthesized data" mentioned in the submission paper (ISSTA 2023)

### Dependency
- numpy
- pandas
- gplearn
- sympy

```commandline
// install dependencies
python setup.py install
```
### Usage

#### SimulatedDataset API
```python
class SimulatedDataset:
    def __init__(self, n_features, support, size, seed: np.random.RandomState,
                 n_confused_column=0, n_ext_column=0,
                 init_depth=(2, 2), init_method='half and half'):
        """
        init a synthesized dataset
       :param n_features: the number of features(columns) in the assertion
       :param support: support
       :param size: the number of records(rows)
       :param seed: the seed to generate data
       :param n_confused_column: the number of additional data columns which are filled with obfuscation
       :param n_ext_column: the number of additional data columns
       :param init_depth: init depth (min, max)
       :param init_method: init method: 'half and half' | 'full'
       """

    def to_csv(self, path):
        """
        :param path: saved file's path
        """
```

#### Code Sample
```python
if __name__ == '__main__':
    for i in range(50): # generate 50 groups of data
        dataset = SimulatedDataset(n_features=3, support=0.8, size=10000,
                                   n_confused_column=1, n_ext_column=1,
                                   init_depth=(2, 2), init_method='half and half',
                                   seed=np.random.RandomState(i))
        dataset.to_csv(f"./x_3_sup_0.8_cx_1_ex_1_{i}.csv")
```