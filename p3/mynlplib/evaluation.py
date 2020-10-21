from mynlplib.parsing import ParserState, DepGraphEdge
from mynlplib.utils import DummyCombiner
from mynlplib.constants import Actions, ROOT_TOK

def dependency_graph_from_oracle(sentence, actions):
    """
    Take a sentence and a sequence of actions
    and return a set of dependency edges for further evaluation
    """
    stack = ParserState(sentence, [None]*len(sentence), DummyCombiner())
    dependency_graph = set()
    
    for act in [ Actions.action_to_ix[a] for a in actions ]:
        if act == Actions.SHIFT:
            stack.shift()
        elif act == Actions.ARC_L:
            dependency_graph.add(stack.arc_left())
        elif act == Actions.ARC_R:
            dependency_graph.add(stack.arc_right())
    
    return dependency_graph


def exact_match(predicted, gold):
    return predicted == gold


def fscore(predicted, gold):
    TP = float(len(predicted.intersection(gold)))
    FN = float(len(gold.difference(predicted)))
    FP = float(len(predicted.difference(gold)))
    return 2 * TP / (2 * TP + FN + FP)


def attachment(predicted, gold):
    correct = 0
    total = 0
    for gold_edge in gold:
        for predicted_edge in predicted:
            if predicted_edge.modifier == gold_edge.modifier:
                if predicted_edge.head == gold_edge.head:
                    correct += 1
                total += 1
                break
    return correct, total


def compute_attachment(parser, data):
    correct = 0
    total = 0
    for sentence, actions in data:
        if len(sentence) <= 1: continue
        gold = dependency_graph_from_oracle(sentence, actions)
        predicted = parser.predict(sentence)
        correct_t, total_t = attachment(predicted, gold)
        correct += correct_t
        total += total_t
    return float(correct) / total


def compute_metric(parser, data, metric):
    val = 0.
    for sentence, actions in data:
        if len(sentence) > 1:
            gold = dependency_graph_from_oracle(sentence, actions)
            predicted = parser.predict(sentence)
            val += metric(predicted, gold)
    return val / len(data)


def output_preds(filename, parser, data):
    with open(filename, "w") as outfile:
        for sentence in data:
            pred_graph = parser.predict(sentence)
            for i, word in enumerate(sentence):
                for edge in pred_graph:
                    if edge.modifier[1] == i:
                        outfile.write("{}\t{}\t{}\t{}\n".format(i, word, edge.head[0], edge.head[1]))
                        break
            outfile.write("\n")


def compute_output_attachment(output_file, key_file):
    import csv
    of = open(output_file)
    of_lines = of.read().split('\n')
    of.close()
    kf = open(key_file)
    kf_lines = kf.read().split('\n')
    kf.close()
    correct, total = 0, 0
    for _i, (ofl, kfl) in enumerate(zip(of_lines, kf_lines)):
        if len(ofl) == 0 and len(kfl) == 0:
            continue
        oflp = ofl.split('\t')
        kflp = kfl.split('\t')
        if len(oflp) != 4 or len(kflp) != 4:
            print(_i, ofl, kfl)
            raise ValueError("Each line of the file should have 4 elements!")
        match = True
        for _o, _k in zip(oflp, kflp):
            if _o != _k:
                match = False
                break
        if match:
            correct += 1
        total += 1
    return float(correct) / total
