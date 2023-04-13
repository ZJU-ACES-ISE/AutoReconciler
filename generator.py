import numpy as np
import pandas as pd

from gplearn.functions import add2, sub2, mul2, div2, _Function
from sympy import simplify


def read_words(filename):
    with open(filename, 'r') as file:
        words = [line.strip() for line in file.readlines()]
    return words


dic = {
    'VALUE:AMT': read_words('./dictionary/value_amt.txt'),
    'VALUE:RATE': read_words('./dictionary/value_rate.txt'),
    'VALUE:NUM': read_words('./dictionary/value_num.txt')
}


class SimulatedDataset:
    def __init__(self, n_features, support, size, seed: np.random.RandomState,
                 n_confused_column=0, n_ext_column=0,
                 with_column_names=False,
                 init_depth=(2, 2), init_method='half and half'):
        """
        :param n_features: the number of features(columns) in the assertion
        :param support: support
        :param size: the number of records(rows)
        :param seed: the seed to generate data
        :param n_confused_column: the number of additional data columns which are filled with obfuscation
        :param n_ext_column: the number of additional data columns
        :param with_column_names: if use real column names (names are the result of generated or real data after desensitization)
        :param init_depth: init depth (min, max)
        :param init_method: init method: 'half and half' | 'full'
        """
        self.n_features = n_features
        self.support = support
        self.size = size

        while True:
            fit = 0
            random_state = np.random.RandomState(seed.randint(100))
            program = Program(function_map={add2: 0.4, sub2: 0.4, mul2: 0.1, div2: 0.1},
                              arities={2: [add2, sub2, mul2, div2]},
                              init_depth=init_depth,
                              init_method=init_method,
                              n_features=n_features,
                              with_column_names=with_column_names,
                              const_range=(-1.0, 1.0),
                              random_state=random_state)

            input = np.array([1000 * seed.randn(n_features) for i in range(size)]).round(2)
            sym = str(program.simplify())
            if 'zoo' in sym or 'X' not in sym:
                continue

            output = program.execute(input)
            for i in range(len(output)):
                if seed.uniform(0, 1) > support:
                    try:
                        output[i] = seed.uniform(-abs(output[i]), abs(output[i])) + output[i]
                    except Exception:
                        continue
                else:
                    fit += 1

            confused_column = []
            for i in range(n_confused_column):
                tmp = output * seed.uniform(-1, 1)
                confused_column.append(tmp)
            confused_column = np.array(confused_column).round(2).T

            ext_column = np.array([1000 * seed.randn(n_ext_column) for i in range(size)]).round(2)

            self.program = program
            self.formular = sym

            if with_column_names:
                columns = [program.label_name]
                columns.extend(program.feature_names)
            else:
                columns = ['Y']
                columns.extend([f'X{j}' for j in range(n_features)])

            final = np.column_stack((output, input))
            if n_ext_column > 0:
                if with_column_names:
                    columns.extend([generate_name(None, random_state) for j in range(n_ext_column)])
                else:
                    columns.extend([f'EX{j}' for j in range(n_ext_column)])
                final = np.column_stack((final, ext_column))
            if n_confused_column > 0:
                if with_column_names:
                    columns.extend([generate_name(None, random_state) for j in range(n_ext_column)])
                else:
                    columns.extend([f'CX{j}' for j in range(n_confused_column)])
                final = np.column_stack((final, confused_column))
            self.data = pd.DataFrame(final, columns=columns)
            self.fit_rate = fit / size
            break

    def to_csv(self, path):
        self.data.to_csv(path)


class Program:
    types = ['VALUE:AMT', 'VALUE:NUM', 'VALUE:RATE']

    def __init__(self,
                 function_map,
                 arities,
                 init_depth,
                 init_method,
                 n_features,
                 with_column_names,
                 const_range,
                 random_state,
                 program=None):

        self.function_map = function_map
        self.arities = arities
        self.init_depth = (init_depth[0], init_depth[1] + 1)
        self.init_method = init_method
        self.n_features = n_features
        self.with_column_names = with_column_names,
        self.feature_names = None
        self.label_name = None
        self.feature_types = None
        self.label_type = None
        self.const_range = const_range
        self.program = program

        if self.program is not None:
            if not self.validate_program():
                raise ValueError('The supplied program is incomplete.')
        else:
            # Create a naive random program
            self.program = self.build_program(random_state)
            while not self.check_semantics():
                self.program = self.build_program(random_state)
            if with_column_names:
                self.fill_names(random_state)

        self.raw_fitness_ = None
        self.fitness_ = None
        self.parents = None
        self._n_samples = None
        self._max_samples = None
        self._indices_state = None

    def build_program(self, random_state, const_rate=0.1):
        """Build a naive random program.

        Parameters
        ----------
        random_state : RandomState instance
            The random number generator.

        Returns
        -------
        program : list
            The flattened tree representation of the program.

        """
        if self.init_method == 'half and half':
            method = ('full' if random_state.randint(2) else 'grow')
        else:
            method = self.init_method
        max_depth = random_state.randint(*self.init_depth)

        # Start a program with a function to avoid degenerative programs
        function = random_state.uniform(0, 1)
        for func, rate in self.function_map.items():
            if function <= rate:
                function = func
                break
            else:
                function -= rate
        program = [function]
        terminal_stack = [function.arity]

        # random set types
        if self.with_column_names:
            rt = random_state.randint(0, 3, size=self.n_features)
            self.feature_types = [self.types[ii] for ii in rt]
            self.label_type = self.types[random_state.randint(0, 3)]

        while terminal_stack:
            depth = len(terminal_stack)
            choice = self.n_features + len(self.function_map)
            choice = random_state.randint(choice)
            # Determine if we are adding a function or terminal
            if (depth < max_depth) and (method == 'full' or
                                        choice <= len(self.function_map)):
                function = random_state.uniform(0, 1)
                for func, rate in self.function_map.items():
                    if function <= rate:
                        function = func
                        break
                    else:
                        function -= rate
                program.append(function)
                terminal_stack.append(function.arity)
            else:
                # We need a terminal, add a variable or constant
                if random_state.uniform(0, 1) < const_rate:
                    terminal = round(random_state.uniform(*self.const_range), 2)
                    if self.const_range is None:
                        # We should never get here
                        raise ValueError('A constant was produced with '
                                         'const_range=None.')
                else:
                    terminal = random_state.randint(self.n_features)
                program.append(terminal)
                terminal_stack[-1] -= 1
                while terminal_stack[-1] == 0:
                    terminal_stack.pop()
                    if not terminal_stack:
                        return program
                    terminal_stack[-1] -= 1

        # We should never get here
        return None

    def execute(self, X):
        """Execute the program according to X.

        Parameters
        ----------
        X : {array-like}, shape = [n_samples, n_features]
            Training vectors, where n_samples is the number of samples and
            n_features is the number of features.

        Returns
        -------
        y_hats : array-like, shape = [n_samples]
            The result of executing the program on X.

        """
        # Check for single-node programs
        node = self.program[0]
        if isinstance(node, float):
            return np.repeat(node, X.shape[0])
        if isinstance(node, int):
            return X[:, node]

        apply_stack = []

        for node in self.program:

            if isinstance(node, _Function):
                apply_stack.append([node])
            else:
                # Lazily evaluate later
                apply_stack[-1].append(node)

            while len(apply_stack[-1]) == apply_stack[-1][0].arity + 1:
                # Apply functions that have sufficient arguments
                function = apply_stack[-1][0]
                terminals = [np.repeat(t, X.shape[0]) if isinstance(t, float)
                             else X[:, t] if isinstance(t, int)
                else t for t in apply_stack[-1][1:]]
                intermediate_result = function(*terminals)
                if len(apply_stack) != 1:
                    apply_stack.pop()
                    apply_stack[-1].append(intermediate_result)
                else:
                    return intermediate_result

        # We should never get here
        return None

    def __str__(self):
        """Overloads `print` output of the object to resemble a LISP tree."""
        terminals = [0]
        output = ''
        for i, node in enumerate(self.program):
            if isinstance(node, _Function):
                terminals.append(node.arity)
                output += node.name + '('
            else:
                if isinstance(node, int):
                    if self.feature_names is None or self.label_name is None:
                        output += 'X%s' % node
                    else:
                        output += self.feature_names[node]
                else:
                    output += '%.3f' % node
                terminals[-1] -= 1
                while terminals[-1] == 0:
                    terminals.pop()
                    terminals[-1] -= 1
                    output += ')'
                if i != len(self.program) - 1:
                    output += ', '
        return output

    def execute(self, X):
        """Execute the program according to X.

        Parameters
        ----------
        X : {array-like}, shape = [n_samples, n_features]
            Training vectors, where n_samples is the number of samples and
            n_features is the number of features.

        Returns
        -------
        y_hats : array-like, shape = [n_samples]
            The result of executing the program on X.

        """
        # Check for single-node programs
        node = self.program[0]
        if isinstance(node, float):
            return np.repeat(node, X.shape[0])
        if isinstance(node, int):
            return X[:, node]

        apply_stack = []

        for node in self.program:

            if isinstance(node, _Function):
                apply_stack.append([node])
            else:
                # Lazily evaluate later
                apply_stack[-1].append(node)

            while len(apply_stack[-1]) == apply_stack[-1][0].arity + 1:
                # Apply functions that have sufficient arguments
                function = apply_stack[-1][0]
                terminals = [np.repeat(t, X.shape[0]) if isinstance(t, float)
                             else X[:, t] if isinstance(t, int)
                else t for t in apply_stack[-1][1:]]
                intermediate_result = function(*terminals)
                if len(apply_stack) != 1:
                    apply_stack.pop()
                    apply_stack[-1].append(intermediate_result)
                else:
                    return intermediate_result

        # We should never get here
        return None

    def simplify(self):
        symbol = {'add': '+', 'sub': '-', 'mul': '*', 'div': '/'}
        program = self.program.copy()
        for i in range(len(program) - 1, -1, -1):
            if isinstance(program[i], _Function):
                program[i] = f'({program[i + 1]} {symbol[program[i].name]} {program[i + 2]})'
                del program[i + 2]
                del program[i + 1]
            elif isinstance(program[i], int):
                program[i] = f'X{program[i]}'
        return simplify(program[0])

    def check_semantics(self):
        allowed = {
            'add': [
                ('VALUE:AMT', 'VALUE:AMT', 'VALUE:AMT'),
                ('VALUE:NUM', 'VALUE:NUM', 'VALUE:NUM'),
                ('VALUE:RATE', 'VALUE:RATE', 'VALUE:RATE'),
            ],
            'sub': [
                ('VALUE:AMT', 'VALUE:AMT', 'VALUE:AMT'),
                ('VALUE:NUM', 'VALUE:NUM', 'VALUE:NUM'),
                ('VALUE:RATE', 'VALUE:RATE', 'VALUE:RATE'),
            ],
            'mul': [
                ('VALUE:AMT', 'VALUE:NUM', 'VALUE:AMT'),
                ('VALUE:AMT', 'VALUE:RATE', 'VALUE:AMT'),
                ('VALUE:NUM', 'VALUE:AMT', 'VALUE:AMT'),
                ('VALUE:NUM', 'VALUE:RATE', 'VALUE:NUM'),
                ('VALUE:RATE', 'VALUE:AMT', 'VALUE:AMT'),
                ('VALUE:RATE', 'VALUE:NUM', 'VALUE:NUM'),
                ('VALUE:RATE', 'VALUE:RATE', 'VALUE:RATE'),
            ],
            'div': [
                ('VALUE:AMT', 'VALUE:AMT', ['VALUE:NUM', 'VALUE:RATE']),
                ('VALUE:AMT', 'VALUE:NUM', 'VALUE:AMT'),
                ('VALUE:AMT', 'VALUE:RATE', 'VALUE:AMT'),
                ('VALUE:NUM', 'VALUE:NUM', 'VALUE:RATE'),
                ('VALUE:RATE', 'VALUE:RATE', 'VALUE:RATE'),
            ]
        }
        node = self.program[0]
        if isinstance(node, float) or self.label_type == 'VALUE':
            return True
        elif isinstance(node, int):
            return self.feature_types[node] == 'VALUE' or self.label_type == self.feature_types[node]

        apply_stack = []
        for node in self.program:
            if isinstance(node, _Function):
                apply_stack.append([node])
            elif isinstance(node, int):
                # Lazily evaluate later
                apply_stack[-1].append(self.feature_types[node])
            else:
                apply_stack[-1].append('VALUE')

            while len(apply_stack[-1]) == apply_stack[-1][0].arity + 1:
                # Apply functions that have sufficient arguments
                function = apply_stack[-1][0]
                intermediate_result = []
                if function.name in allowed.keys():
                    for rule in allowed[function.name]:
                        match = True
                        for i in range(function.arity):
                            var_type = apply_stack[-1][i + 1]
                            if isinstance(var_type, str) and rule[i] != var_type and var_type != 'VALUE':
                                match = False
                                break
                            elif isinstance(var_type, list) and rule[i] not in var_type:
                                match = False
                                break
                        if match:
                            intermediate_result.append(rule[-1])
                    if len(intermediate_result) == 0:
                        return False
                    elif len(intermediate_result) == 1:
                        intermediate_result = intermediate_result[0]
                else:
                    intermediate_result = 'VALUE'
                if len(apply_stack) != 1:
                    apply_stack.pop()
                    apply_stack[-1].append(intermediate_result)
                else:
                    if intermediate_result == 'VALUE':
                        return True
                    elif isinstance(intermediate_result, str):
                        return self.label_type == intermediate_result
                    elif isinstance(intermediate_result, list):
                        return self.label_type in intermediate_result
        # We should never get here
        return None

    def fill_names(self, random_state):
        self.label_name = generate_name(self.label_type, random_state)
        self.feature_names = []
        for t in self.feature_types:
            self.feature_names.append(generate_name(t, random_state))


def generate_name(type, random_state):
    choices = []
    if type is None:
        choices.extend(dic['VALUE:AMT'])
        choices.extend(dic['VALUE:RATE'])
        choices.extend(dic['VALUE:NUM'])
    else:
        choices.extend(dic[type])
    return random_state.choice(choices)


if __name__ == '__main__':

    # generate sample data (in sample_data.zip)
    for i in range(50):
        dataset = SimulatedDataset(n_features=3, support=0.8, size=10000,
                                   n_confused_column=1, n_ext_column=1,
                                   with_column_names=False,
                                   init_depth=(2, 2), init_method='half and half',
                                   seed=np.random.RandomState(i))
        dataset.to_csv(f"./sample_data/x_3_sup_0.8_cx_1_ex_1_without_name_{i}.csv")

# %%
