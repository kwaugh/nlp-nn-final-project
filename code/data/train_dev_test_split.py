import argparse
import math

# file_prefix does not include file extension
def train_dev_test_split(file_prefix, file_extension, percentage):
    with open(file_prefix + file_extension, 'r') as f:
        lines = f.readlines()
        num_lines = int(len(lines) * percentage)
        train_lines = math.ceil(num_lines * 0.8)
        dev_lines = math.ceil(num_lines * 0.1)

        suffix_to_lines = {
                'train': (0, train_lines),
                'dev': (train_lines, train_lines + dev_lines),
                'test': (train_lines + dev_lines, num_lines)
        }

        for suffix, line_pos in suffix_to_lines.items():
            with open('{}_{}.txt'.format(file_prefix, suffix), 'w') as g:
                for j in range(line_pos[0], line_pos[1]):
                    g.write(lines[j])

if __name__ == '__main__':
    parser = argparse.ArgumentParser('description=train_dev_test_split.py')
    parser.add_argument(
        '--percentage',
        dest='percentage',
        type=float,
        default=1.0,
        help='What percentage of the europarl data should be used?')
    args = parser.parse_args()
    train_dev_test_split('english', '.txt', args.percentage)
    train_dev_test_split('french', '.txt', args.percentage)
    train_dev_test_split('english_parse', '.txt', args.percentage)
    train_dev_test_split('french_parse', '.txt', args.percentage)
