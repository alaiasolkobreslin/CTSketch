import scallopy
import itertools

class CreateMNISTScallopSum:

    def __init__(self, N, provenance, k):
        self.N = N
        self.scl_ctx = scallopy.ScallopContext(provenance=provenance, k=k)

    def add_relation(self, index):
        self.scl_ctx.add_relation('digit_' + str(index), int, input_mapping=list(range(10)))
        
    def unique_strings(self):
        length = 1
        num_strings = 0
        while num_strings < self.N + 3:
            for s in itertools.product('abcdefghijklmnopqrstuvwxyz', repeat=length):
                num_strings += 1
                string = ''.join(s)
                if string == 'as' or s == 'if' or s == 'or':
                    continue
                else:
                    yield ''.join(s)
            length += 1

    def add_sum_rule(self):
        rule = f"sum_{self.N}"
        arguments = "("
        digits = ""
        first = True
        generator = self.unique_strings()
        for i in range(self.N):
            s = next(generator)
            self.add_relation(i+1)
            if first:
                arguments += s
                digits += f"digit_{str(i+1)}({s})"
                first = False
            else:
                arguments += f" + {s}"
                digits += f", digit_{str(i+1)}({s})"
        rule += arguments + ") :- " + digits
        print(rule)
        self.scl_ctx.add_rule(rule)
        return self.scl_ctx
        
# if __name__ == "__main__":
#     create = CreateMNISTScallopSum(1024, "difftopkproofs", 3)
#     create.add_sum_rule()