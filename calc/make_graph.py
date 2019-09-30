#! /usr/bin/env python3

import pickle
import network

with open("optimization-checkpoint.pkl", "rb") as f:
    data = pickle.load(f)

data["net"].write_dot_graph("graph.dot")
