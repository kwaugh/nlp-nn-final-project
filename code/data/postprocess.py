# postprocess the input/output pairs to make sure that they all line up
from parse_to_sentence import *

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
            sanitized_input = []
            sanitized_output = []
            while input_idx < len(input_lines) and output_idx < len(output_lines):
                in_line = input_lines[input_idx].strip().replace(' ', '')
                out_line = parse_to_sentence(
                        output_lines[output_idx]).replace(' ', '')
                # There are cases where the input and output are really
                # close but off by one, like appending an extra period or
                # close parenthesis. This is good enough.
                if in_line != out_line and abs(len(in_line) - len(out_line)) > 2:
                    ''' useful for debugging
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
                    '''
                    num_diff += 1
                    if len(in_line) > len(out_line):
                        input_idx += 1
                        output_idx += 2
                    else:
                        input_idx += 2
                        output_idx += 1
                else:
                    sanitized_input.append(input_lines[input_idx])
                    sanitized_output.append(output_lines[output_idx])
                    input_idx += 1
                    output_idx += 1
            print('num_diff: {}'.format(num_diff))
            # write the good input/output pairs to a file
    return sanitized_input, sanitized_output

if __name__ == '__main__':
    for i in range(0, 11):
        '''
        input_file_name = 'en_splits/{}.txt'.format(i)
        output_file_name = 'en_out/{}.txt'.format(i)
        '''
        input_file_name = 'fr_splits/{}.txt'.format(i)
        output_file_name = 'fr_out/{}.txt'.format(i)
        sanitized_input, sanitized_output = make_input_output_match(
                input_file_name, output_file_name)
        '''
        with open(input_file_name, 'w') as input:
            for input_line in sanitized_input:
                input.write('%s\n' % input_line)
        with open(output_file_name, 'w') as output:
            for output_line in sanitized_output:
                output.write('%s\n' % output_line)
        '''
