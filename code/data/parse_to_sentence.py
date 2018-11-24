# given a parsed sentence, it returns the raw sentence
def parse_to_sentence(parsed_sentence):
    split_sentence = parsed_sentence.strip().split(' ')
    clean_sentence = []
    for i in range(len(split_sentence)):
        if len(split_sentence[i]) > 1 and \
                split_sentence[i][-1] == ')' and \
                split_sentence[i][0] != ')':
                    clean_sentence.append(split_sentence[i].replace(')', ''))
    sentence = ' '.join(clean_sentence)
    sentence = sentence.replace(' ,', ',')
    sentence = sentence.replace(' \' ', '\'')
    sentence = sentence.replace(' \'', '\'')
    sentence = sentence.replace(' .', '.')
    sentence = sentence.replace(' ?', '?')
    sentence = sentence.replace(' !', '!')
    sentence = sentence.replace('-LRB-', '(')
    sentence = sentence.replace('-RRB-', ')')
    sentence = sentence.replace('``', '"')
    sentence = sentence.replace('`', '\'')
    sentence = sentence.replace('\'\'', '"')
    sentence = sentence.replace('-LSB-', '[')
    sentence = sentence.replace('-RSB-', ']')
    # filter non-ascii characters
    sentence = ''.join(i for i in sentence if ord(i) < 128)
    return sentence

if __name__ == '__main__':
    with open("en_splits/0.txt") as f_in:
        with open("en_out/0.txt") as f_out:
            input_lines = f_in.readlines()
            output_lines = f_out.readlines()
            diff_count = 0
            for i in range(len(output_lines)):
                output_converted = parse_to_sentence(output_lines[i])
                if input_lines[i].strip() != output_converted:
                    diff_count += 1
            print(diff_count)
