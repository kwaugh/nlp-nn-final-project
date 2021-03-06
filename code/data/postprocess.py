# postprocess the input/output pairs to make sure that they all line up
from parse_to_sentence import *
from multiprocessing import Pool

def make_input_output_match(input_file_name, output_file_name):
    with open(input_file_name) as input:
        with open(output_file_name) as output:
            input_lines = input.readlines()
            output_lines = output.readlines()

            input_idx = 0
            output_idx = 0
            # always discard from the longer one, because it probably meant that
            # two lines were concatenated together, and then get rid of two from
            # the input because they're probably the ones that got concatted
            # together

            # all the sentences should match, save for some spaces in the wrong
            # places like apostrophes, so let's compare that version of the
            # sentences
            num_diff = 0
            good_indices = []
            #print('len(input_lines): {}'.format(len(input_lines)))
            #print('len(output_lines): {}'.format(len(output_lines)))
            while input_idx < len(input_lines) and output_idx < len(output_lines):
                in_line = input_lines[input_idx].strip().replace(' ', '')
                out_line = parse_to_sentence(
                        output_lines[output_idx]).replace(' ', '')
                # There are cases where the input and output are really
                # close but off by one, like appending an extra period or
                # close parenthesis. This is good enough.
                if in_line != out_line and abs(len(in_line) - len(out_line)) > 2:
                    ''' useful for debugging
                    print('input_idx: {}'.format(input_idx))
                    print('in_line:  {}'.format(in_line))
                    print('out_line: {}'.format(out_line))
                    foo = list(parse_to_sentence(output_lines[output_idx]))
                    foo = list(map(lambda x: ord(x), foo))
                    print('foo: {}'.format(foo))
                    print('input_line:  {}'.format(input_lines[input_idx]))
                    print('output_line: {}'.format(output_lines[output_idx]))
                    print()
                    if num_diff > 5:
                        print('input_idx:  {}'.format(input_idx))
                        print('output_idx: {}'.format(output_idx))
                        print('EXITING')
                        exit()
                    # '''
                    num_diff += 1
                    if len(in_line) > len(out_line):
                        input_idx += 1
                        output_idx += 2
                    else:
                        input_idx += 2
                        output_idx += 1
                elif len(in_line.split(' ')) > 65 \
                        or len(out_line.split(' ')) > 65:
                    input_idx += 1
                    output_idx += 1
                else:
                    good_indices.append(input_idx)
                    input_idx += 1
                    output_idx += 1
            #print('num_diff: {}'.format(num_diff))
            # write the good input/output pairs to a file
    return set(good_indices)

def main(i):
    files = {
            'input': {
                'english': 'en_splits/{}.txt'.format(i),
                'french': 'fr_splits/{}.txt'.format(i),
                },
            'output': {
                'english': 'en_out/{}.txt'.format(i),
                'french': 'fr_out/{}.txt'.format(i),
                }}
    good_english_indices = make_input_output_match(
            files['input']['english'], files['output']['english'])
    good_french_indices = make_input_output_match(
            files['input']['french'], files['output']['french'])

    good_indices = good_english_indices.intersection(good_french_indices)
    old_data = {
            'input': {'english': [], 'french': []},
            'output': {'english': [], 'french': []},
    }

    for in_out in files:
        for lang in files[in_out]:
            with open(files[in_out][lang]) as f:
                old_data[in_out][lang] = f.readlines()

    print('num_thrown_away for {}.txt: {}'.format(
        i, len(old_data['input']['english']) - len(good_indices)))

    # save the new data on top
    '''
    for lang in files['input']:
        with open('postprocess/' + files['input'][lang], 'w') as f:
            for idx in good_indices:
                f.write('%s' % old_data['input'][lang])
    #'''

if __name__ == '__main__':
    try:
        pool = Pool(12)
        pool.map(main, list(range(0, 12)))
    finally:
        pool.close()
        pool.join()
