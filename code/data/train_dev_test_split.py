import math

num_splits = 12

# file_prefix does not include file extension
def train_dev_test_split(file_prefix, file_extension):
    with open(file_prefix + file_extension, 'r') as f:
        lines = f.readlines()
        train_lines = math.ceil(len(lines) * 0.8)
        dev_lines = math.ceil(len(lines) * 0.1)

        suffix_to_lines = {
                'train': (0, train_lines),
                'dev': (train_lines, train_lines + dev_lines),
                'test': (train_lines + dev_lines, len(lines))
        }

        for suffix, line_pos in suffix_to_lines.items():
            with open('{}_{}.txt'.format(file_prefix, suffix), 'w') as g:
                for j in range(line_pos[0], line_pos[1]):
                    g.write(lines[j])

if __name__ == '__main__':
    train_dev_test_split('english', '.txt')
    train_dev_test_split('french', '.txt')
    train_dev_test_split('english_parse', '.txt')
    train_dev_test_split('french_parse', '.txt')
