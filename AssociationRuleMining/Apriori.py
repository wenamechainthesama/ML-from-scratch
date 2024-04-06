from functools import reduce
from itertools import combinations, product

"""
Concept:
https://medium.com/analytics-vidhya/apriori-algorithm-in-association-rule-learning-9287fe17e944
https://medium.com/@byanalytixlabs/apriori-algorithm-in-data-mining-implementation-examples-and-more-ab17662ecb0e
https://towardsdatascience.com/apriori-association-rule-mining-explanation-and-python-implementation-290b42afdfc6

Implementation:
https://github.com/chonyy/apriori_python
https://github.com/deepshig/apriori-python
"""


class Apriori:
    def __init__(self, min_support, min_confidence):
        self.min_support = min_support
        self.min_confidence = min_confidence

    def find_rules(self, X):
        itemsets_supports = {}

        # Compute support for all items
        flatten_data = reduce(lambda x, y: x + y, X)
        unique_items = set(flatten_data)
        items = set([frozenset([item]) for item in unique_items])

        # Remove items with support < min_support
        filtered_items = self._filter_itemsets(X, items)
        itemsets_supports = {**itemsets_supports, **filtered_items}

        # Repeat until done
        combo_length = 2
        while True:
            itemsets = [
                frozenset(itemset)
                for itemset in combinations(unique_items, combo_length)
            ]
            filtered_items = self._filter_itemsets(X, itemsets)
            if filtered_items == {}:
                break
            itemsets_supports = {**itemsets_supports, **filtered_items}
            combo_length += 1

        association_rules = self._association_rules(itemsets_supports)
        return association_rules

    def _filter_itemsets(self, X, itemsets):
        filtered_items = {}
        for itemset in itemsets:
            support = self._itemset_support(X, itemset)
            if support >= self.min_support:
                filtered_items.setdefault(itemset, support)

        return filtered_items

    def _itemset_support(self, X, itemset):
        support_count = 0
        for row in X:
            if set(itemset).issubset(row):
                support_count += 1
        support = support_count / len(X)
        return support

    def _association_rules(self, itemsets_supports: dict):
        rules = []
        for item in itemsets_supports.keys():
            if len(item) == 1:
                continue
            subsets = list(product(item, repeat=len(item)))
            new_rules = self._subset_rules(item, subsets, itemsets_supports)
            rules.extend(new_rules)

        return rules

    def _subset_rules(self, item, subsets, itemsets_supports):
        new_rules = []
        for subset in subsets:
            difference = set(item) - set(subset)
            if not difference:
                continue
            subset = frozenset(subset)
            union = subset | difference
            confidence = itemsets_supports[union] / itemsets_supports[subset]
            if confidence >= self.min_confidence:
                new_rules.append((subset, frozenset(difference), confidence))

        return new_rules


if __name__ == "__main__":
    itemsets = [
        ["apple", "beer", "rice", "chicken"],
        ["apple", "beer", "rice"],
        ["apple", "beer"],
        ["apple", "mango"],
        ["milk", "beer", "rice", "chicken"],
        ["milk", "beer", "rice"],
        ["milk", "beer"],
        ["milk", "mango"],
    ]

    apriori = Apriori(0.5, 0.5)
    rules = list(sorted(apriori.find_rules(itemsets), key=lambda x: x[2]))
    print(rules)
