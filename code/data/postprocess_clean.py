if __name__ == '__main__':
    good_indices = set()
    english_lines = []
    french_lines = []
    english_parse_lines = []
    french_parse_lines = []
    with open('english.txt') as english:
        with open('french.txt') as french:
            with open('english_parse.txt') as english_parse:
                with open('french_parse.txt') as french_parse:
                    english_lines = english.readlines()
                    french_lines = french.readlines()
                    english_parse_lines = english_parse.readlines()
                    french_parse_lines = french_parse.readlines()
                    for idx in range(len(english_lines)):
                        english_line = english_lines[idx]
                        french_line = french_lines[idx]
                        if len(english_line.split(' ')) < 65 and \
                                len(french_line.split(' ')) < 65:
                                    good_indices.add(idx)
    with open('english_trimmed.txt', 'w') as english:
        with open('french_trimmed.txt', 'w') as french:
            with open('english_parse_trimmed.txt', 'w') as english_parse:
                with open('french_parse_trimmed.txt', 'w') as french_parse:
                    for idx in range(len(english_lines)):
                        if idx in good_indices:
                            english.write('%s' % english_lines[idx])
                            french.write('%s' % french_lines[idx])
                            english_parse.write('%s' % english_parse_lines[idx])
                            french_parse.write('%s' % french_parse_lines[idx])
