#! /usr/bin/env python3

import pickle
import network

with open("outputs/optimization-checkpoint.pkl", "rb") as f:
    data = pickle.load(f)

data["net"].write_dot_graph("graph.dot")
