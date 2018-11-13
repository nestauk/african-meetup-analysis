import networkx as nx
import pandas as pd
import itertools
from collections import Counter
from node2vec import Node2Vec

def cooccurrence_graph(elements):
    # Get all of the unique entries you have
    varnames = tuple(sorted(set(itertools.chain(*elements))))

    # Get a list of all of the combinations you have
    expanded = [tuple(itertools.combinations(d, 2)) for d in elements]
    expanded = itertools.chain(*expanded)

    # Sort the combinations so that A,B and B,A are treated the same
    expanded = [tuple(sorted(d)) for d in expanded]

    # count the combinations
    return Counter(expanded)

def main():
    groups_members = pd.read_csv('../data/raw/groups_members.csv')
    groups = groups_members.drop_duplicates('group_id')
    groups.reset_index(inplace=True)

    # preprocess meetup tags
    tags = [[tag.strip() for tag in u.split(',')] for u in list(groups.urlkey)]

    cooccurrence = cooccurrence_graph(tags)

    G = nx.Graph()
    for k, v in cooccurrence.items():
        G.add_edge(k[0], k[1], weight=int(v))

    nx.write_graphml(G, '../models/v1/G_africa.graphml')
    # Precompute probabilities and generate walks
    node2vec = Node2Vec(G, dimensions=64, walk_length=30, num_walks=200, workers=4)

    # Embed nodes
    model = node2vec.fit(window=10, min_count=2, batch_words=4)

    # Save embeddings for later use
    model.wv.save_word2vec_format('../models/v1/node2vec_embeddings.model')

    # Save model for later use
    model.save('../models/v1/node2vec.model')

if __name__ == '__main__':
    main()
