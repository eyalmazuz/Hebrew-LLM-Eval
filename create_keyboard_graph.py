import pickle

import networkx as nx

# Define the full Hebrew keyboard layout, including numbers and punctuation
hebrew_keyboard_rows = ["`1234567890-=", "/'קראטוןםפ]", "[שדגכעיחלךף", "זסבהנמצתץ."]

# Initialize an empty graph
G: nx.Graph = nx.Graph()

# Add nodes and edges based on adjacency
for r, row in enumerate(hebrew_keyboard_rows):
    for c, key in enumerate(row):
        # Add the key as a node
        G.add_node(key, pos=(c, -r))

        # Add edges to the left key in the same row
        if c > 0:
            G.add_edge(row[c - 1], key)

        # Add edges to the key above if not the top row and there's a key in that column
        if r > 0 and c < len(hebrew_keyboard_rows[r - 1]):
            G.add_edge(hebrew_keyboard_rows[r - 1][c], key)

        # Add diagonal edges to upper-left and upper-right keys, if present
        if r > 0:
            if c > 0 and c - 1 < len(hebrew_keyboard_rows[r - 1]):
                G.add_edge(hebrew_keyboard_rows[r - 1][c - 1], key)
            if c + 1 < len(hebrew_keyboard_rows[r - 1]):
                G.add_edge(hebrew_keyboard_rows[r - 1][c + 1], key)

# Save the graph to a file
with open("keyboard.gpickle", "wb") as f:
    pickle.dump(G, f, pickle.HIGHEST_PROTOCOL)

# To load the graph back later, you can use:
# with open('keyboard.gpickle', 'rb') as f:
#     G = pickle.load(f)
