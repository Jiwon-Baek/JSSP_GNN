# tiny genetic programming by © moshe sipper, www.moshesipper.com
import os
import string
import numpy as np
import pandas as pd

from random import random, randint
from statistics import mean
from copy import deepcopy

from environment.env import QuayScheduling


def add(x, y): return x + y


def sub(x, y): return x - y


def mul(x, y): return x * y


class GPTree:
    def __init__(self, min_depth, xo_rate, prob_mutation, functions, terminals, data=None, left=None, right=None):
        self.min_depth = min_depth
        self.xo_rate = xo_rate
        self.prob_mutation = prob_mutation
        self.functions = functions
        self.terminals = terminals
        self.data = data
        self.left = left
        self.right = right

    def node_label(self):  # string label
        if (self.data in self.functions):
            return self.data.__name__
        else:
            return str(self.data)

    def print_tree(self, prefix=""):  # textual printout
        print("%s%s" % (prefix, self.node_label()))
        if self.left:  self.left.print_tree(prefix + "   ")
        if self.right: self.right.print_tree(prefix + "   ")

    def compute_tree(self, state):
        if self.data in self.functions:
            return self.data(self.left.compute_tree(state), self.right.compute_tree(state))
        else:
            idx = int(self.data[-1])
            return state[:, idx - 1]

    def random_tree(self, grow, max_depth, depth=0):  # create random tree using either grow or full method
        if depth < self.min_depth or (depth < max_depth and not grow):
            self.data = self.functions[randint(0, len(self.functions) - 1)]
        elif depth >= max_depth:
            self.data = self.terminals[randint(0, len(self.terminals) - 1)]
        else:  # intermediate depth, grow
            if random() > 0.5:
                self.data = self.terminals[randint(0, len(self.terminals) - 1)]
            else:
                self.data = self.terminals[randint(0, len(self.terminals) - 1)]
        if self.data in self.functions:
            self.left = GPTree(self.min_depth, self.xo_rate, self.prob_mutation, self.functions, self.terminals)
            self.left.random_tree(grow, max_depth, depth=depth + 1)
            self.right = GPTree(self.min_depth, self.xo_rate, self.prob_mutation, self.functions, self.terminals)
            self.right.random_tree(grow, max_depth, depth=depth + 1)

    def mutation(self):
        if random() < self.prob_mutation:  # mutate at this node
            self.random_tree(grow=True, max_depth=2)
        elif self.left:
            self.left.mutation()
        elif self.right:
            self.right.mutation()

    def size(self):  # tree size in nodes
        if self.data in self.terminals: return 1
        l = self.left.size() if self.left else 0
        r = self.right.size() if self.right else 0
        return 1 + l + r

    def build_subtree(self):  # count is list in order to pass "by reference"
        t = GPTree(self.min_depth, self.xo_rate, self.prob_mutation, self.functions, self.terminals)
        t.data = self.data
        if self.left:  t.left = self.left.build_subtree()
        if self.right: t.right = self.right.build_subtree()
        return t

    def scan_tree(self, count, second):  # note: count is list, so it's passed "by reference"
        count[0] -= 1
        if count[0] <= 1:
            if not second:  # return subtree rooted here
                return self.build_subtree()
            else:  # glue subtree here
                self.data = second.data
                self.left = second.left
                self.right = second.right
        else:
            ret = None
            if self.left and count[0] > 1: ret = self.left.scan_tree(count, second)
            if self.right and count[0] > 1: ret = self.right.scan_tree(count, second)
            return ret

    def crossover(self, other):  # xo 2 trees at random nodes
        if random() < self.xo_rate:
            second = other.scan_tree([randint(1, other.size())], None)  # 2nd random subtree
            self.scan_tree([randint(1, self.size())], second)  # 2nd subtree "glued" inside 1st tree


# end class GPTree

def init_population(pop_size, min_depth, max_depth, xo_rate, prob_mutation, functions, terminals):  # ramped half-and-half
    pop = []
    for md in range(4, max_depth + 1):
        for i in range(int(pop_size / 10)):
            t = GPTree(min_depth, xo_rate, prob_mutation, functions, terminals)
            t.random_tree(grow=True, max_depth=md)  # grow
            pop.append(t)
        for i in range(int(pop_size / 10)):
            t = GPTree(min_depth, xo_rate, prob_mutation, functions, terminals)
            t.random_tree(grow=False, max_depth=md)  # full
            pop.append(t)
    return pop


def fitness(individual, data_dir):
    data_paths = os.listdir(data_dir)
    lst_cost = []
    for path in data_paths:
        env = QuayScheduling(data_dir + path, algorithm='GP', record_events=False)

        state, mask, _, _ = env.reset()
        done = False

        while not done:
            priority_score = individual.compute_tree(state)
            mask = mask.transpose(0, 1).flatten()
            priority_score[~mask] = -float('inf')
            action = np.argmax(priority_score)

            next_state, _, done, next_mask, _, _ = env.step(action)

            state = next_state
            mask = next_mask

            if done:
                delay_cost = 4000 * sum(env.monitor.delay.values())
                move_cost = 4000 * sum(env.monitor.move.values())
                loss_cost = 12 * sum(env.monitor.loss.values())
                lst_cost.append(delay_cost + move_cost + loss_cost)
                break

    return sum(lst_cost) / len(lst_cost)


def selection(population, fitnesses, tournament_size):  # select one individual using tournament selection
    tournament = [randint(0, len(population) - 1) for i in range(tournament_size)]  # select tournament contenders
    tournament_fitnesses = [fitnesses[tournament[i]] for i in range(tournament_size)]
    return deepcopy(population[tournament[tournament_fitnesses.index(min(tournament_fitnesses))]])