import math

num_splits = 12

def split(input_file, output_prefix):
    with open(input_file, "r") as f:
        lines = f.readlines()
        split_size = math.ceil(len(lines) / float(num_splits))

        for i in range(num_splits):
            with open("{}/{}.txt".format(output_prefix, i), "w") as g:
                s = split_size * i
                e = min(len(lines), (i + 1) * split_size)

                for j in range(s, e):
                    g.write(lines[j])

if __name__ == '__main__':
    split("clean-fr.txt", "fr:_splits")
    split("clean-en.txt", "en_splits")
