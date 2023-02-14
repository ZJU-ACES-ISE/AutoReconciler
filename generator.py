import numpy as np
import pandas as pd

from gplearn.functions import add2, sub2, mul2, div2, _Function
from sympy import simplify


class SingleRuleGenerator:

    def __init__(self, n_features, support, size, seed: np.random.RandomState,
                 n_confused_column=0, n_ext_column=0,
                 init_depth=(2, 2), init_method='half and half',
                 rule_number=0):
        self.n_features = n_features
        self.support = support
        self.size = size

        while True:
            fit = 0
            program = Program(function_map={add2: 0.4, sub2: 0.4, mul2: 0.1, div2: 0.1},
                              arities={2: [add2, sub2, mul2, div2]},
                              init_depth=init_depth,
                              init_method=init_method,
                              n_features=n_features,
                              const_range=(-1.0, 1.0),
                              random_state=np.random.RandomState(seed.randint(100)))

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
            columns = ['Y']
            columns.extend([f'X{j}' for j in range(n_features)])

            final = np.column_stack((output, input))
            if n_ext_column > 0:
                columns.extend([f'EX{j}' for j in range(n_ext_column)])
                final = np.column_stack((final, ext_column))
            if n_confused_column > 0:
                columns.extend([f'CX{j}' for j in range(n_confused_column)])
                final = np.column_stack((final, confused_column))
            self.data = pd.DataFrame(final, columns=columns)
            self.fit_rate = fit / size
            break


class MultipleRuleGenerator:

    def __init__(self, n_features, support, size, seed: np.random.RandomState,
                 n_ext_column=0,
                 init_depth=(2, 2), init_method='half and half',
                 n_rule=0):
        self.n_features = n_features
        self.support = support
        self.size = size
        self.program = []
        self.formular = []
        self.fit_rate = []


        input = []
        output = []
        for i in range(n_rule):
            while True:
                fit = 0
                program = Program(function_map={add2: 0.4, sub2: 0.4, mul2: 0.1, div2: 0.1},
                                  arities={2: [add2, sub2, mul2, div2]},
                                  init_depth=init_depth,
                                  init_method=init_method,
                                  n_features=n_features,
                                  const_range=(-1.0, 1.0),
                                  random_state=np.random.RandomState(seed.randint(100)))

                input.append(np.array([1000 * seed.randn(n_features) for i in range(size)]).round(2))
                sym = str(program.simplify())
                if 'zoo' in sym or 'X' not in sym:
                    continue

                output.append(program.execute(input[i]))
                for j in range(len(output[i])):
                    if seed.uniform(0, 1) > support:
                        try:
                            output[i][j] = seed.uniform(-abs(output[i][j]), abs(output[i][j])) + output[i][j]
                        except Exception:
                            continue
                    else:
                        fit += 1
                self.program.append(program)
                self.formular.append(sym)
                self.fit_rate.append(fit / size)
                break

        # confused_column = []
        # for i in range(n_confused_column):
        #     tmp = output * seed.uniform(-1, 1)
        #     confused_column.append(tmp)
        # confused_column = np.array(confused_column).round(2).T

        ext_column = np.array([1000 * seed.randn(n_ext_column) for i in range(size)]).round(2)

        columns = []
        final = None
        for i in range(n_rule):
            columns.append('Y' + str(i))
            columns.extend([f'X{j}' for j in range(n_features * i, n_features * (i + 1))])
            if final is None:
                final = np.column_stack((output[i], input[i]))
            else:
                final = np.column_stack((final, output[i], input[i]))

        if n_ext_column > 0:
            columns.extend([f'EX{j}' for j in range(n_ext_column)])
            final = np.column_stack((final, ext_column))
        # if n_confused_column > 0:
        #     columns.extend([f'CX{j}' for j in range(n_confused_column)])
        #     final = np.column_stack((final, confused_column))
        self.data = pd.DataFrame(final, columns=columns)


class Program:
    def __init__(self,
                 function_map,
                 arities,
                 init_depth,
                 init_method,
                 n_features,
                 const_range,
                 random_state,
                 program=None):

        self.function_map = function_map
        self.arities = arities
        self.init_depth = (init_depth[0], init_depth[1] + 1)
        self.init_method = init_method
        self.n_features = n_features
        self.feature_names = None
        self.const_range = const_range
        self.program = program

        if self.program is not None:
            if not self.validate_program():
                raise ValueError('The supplied program is incomplete.')
        else:
            # Create a naive random program
            self.program = self.build_program(random_state)

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
                    if self.feature_names is None:
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


if __name__ == '__main__':
    # SingleRuleGenerator(n_features=2, support=0.8, size=1000, n_confused_column=3, n_ext_column=2, seed=np.random.RandomState(0))
    MultipleRuleGenerator(n_features=2, support=0.8, size=1000, seed=np.random.RandomState(0), n_rule=2)
