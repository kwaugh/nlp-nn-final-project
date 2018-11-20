import math

num_splits = 12

with open("clean-fr.txt", "r") as f:
    lines = f.readlines()
    split_size = math.ceil(len(lines) / float(num_splits))

    for i in range(num_splits):
        with open("fr_splits/" + str(i) + ".txt", "w") as g:
            s = split_size * i
            e = min(len(lines), (i + 1) * split_size)

            for j in range(s, e):
                g.write(lines[j])

with open("clean-en.txt", "r") as f:
    lines = f.readlines()
    split_size = math.ceil(len(lines) / float(num_splits))

    for i in range(num_splits):
        with open("en_splits/" + str(i) + ".txt", "w") as g:
            s = split_size * i
            e = min(len(lines), (i + 1) * split_size)

            for j in range(s, e):
                g.write(lines[j])

