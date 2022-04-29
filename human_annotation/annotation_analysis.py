
eval_shows = ['friends', 'tbbt']

dtype_counters = {}
rtype_counters = {}
evide_counters = {}
id2types_dicts = {}

for show_name in eval_shows:
    print('--------------------------------')
    print('   Loading {} annotations'.format(show_name))
    print('--------------------------------')
    # load tsv file
    filename = '{}.tsv'.format(show_name)
    # filename = 'human_annotation/annotations/The_Big_Bang_Theory.annotations.tsv'
    with open(filename, 'r') as fp:
        lines = fp.readlines()
        lines = [line[:-1].split('\t') for line in lines[1:]]

    # prepare for counters
    dtype_counters[show_name] = {
        'Direct Dep.':      [],
        'Indirect Dep.':    [],
        'No Dep.':          [],
    }

    rtype_counters[show_name] = {
        'Default Conjunction':  [],
        'Multihop-Character':   [],
        'Multihop-Textual':     [],
        'Commonsense':          [],
        'Not Required':         [],
    }

    evide_counters[show_name] = {
        'Linguistic Style': [],
        'Personality':      [],
        'Fact(Attribute)':  [],
        'Fact(Relation)':   [],
        'Fact(Status)':     [],
        'Fact':             [],
        'Memory':           [],
        'Inside Scene(Background)': [],
        'Inside Scene(Mention)':    [],
        'Inside Scene(Exclusion)':  [],
    }

    id2types_dicts[show_name] = {}
    
    dtype_counter = dtype_counters[show_name]
    rtype_counter = rtype_counters[show_name]
    evide_counter = evide_counters[show_name]
    id2types = id2types_dicts[show_name]

    # analyze each row
    for li, cells in enumerate(lines):
        # 0: episode
        # 1: need history?
        # 2: facts from memory
        # 3: character
        # 4: guessed char
        # 5: evidence-a1
        # 6: evidence-a2
        # 7: evidence-a3
        # 8: reasoning type

        episode_id = cells[0].split()[0]
        char_id = '{}\t{}'.format(episode_id, cells[3])
        id2types[char_id] = {'name': cells[4].lower(), 'dtype':[], 'rtype':[], 'etype':[]}

        # check if correct
        correct = cells[3].lower() == cells[4].lower()

        # dependence
        if cells[1] != 'Unsolvable':
            assert cells[1] in ['yes', 'no'], "[X] Unkown dependence type (sample {}): {}".format(li, cells[1])
            assert cells[2] in ['yes', 'no'], "[X] Unkown dependence type (sample {}): {}".format(li, cells[2])
            if cells[1] == 'yes':
                dtype = 'Direct Dep.'
            elif cells[2] == 'yes':
                dtype = 'Indirect Dep.'
            else:
                dtype = 'No Dep.'
            dtype_counter[dtype].append(correct)
            id2types[char_id]['dtype'].append(dtype)

        # reasoning type
        if cells[8] == "":
            if len(cells[6]) > 0:
                rtype = "Default Conjunction"
            else:
                rtype = "Not Required"
        elif cells[8] == "no":
            rtype = "Not Required"
        elif cells[8] == "multihop-textual":
            rtype = "Multihop-Textual"
        elif cells[8] == "multihop-character":
            rtype = "Multihop-Character"
        elif cells[8] == "commonsense":
            rtype = "Commonsense"
        else:
            assert False, "[X] Unkown reasoning type (sample {}): {}".format(li, cells[8])
        rtype_counter[rtype].append(correct)
        id2types[char_id]['rtype'].append(rtype)

        # clean evidence
        evidences = list()
        for col in range(5, 8):
            evidence = cells[col]
            if evidence == '':
                continue
            if evidence == 'personality':
                evidence = 'Personality'
            elif evidence == 'linguistic style':
                evidence = 'Linguistic Style'
            elif evidence == 'linguistic style':
                evidence = 'Linguistic Style'
            elif evidence == 'history event':
                evidence = 'Memory'
            elif '(' in evidence:
                b_idx = evidence.find('(')
                e_idx = evidence.find(')')
                sub_evid = ' '.join(evidence[b_idx + 1 : e_idx].split('_')).title()
                evidence = ' '.join(evidence[:b_idx].split('_')).title()
                if sub_evid == 'Exclusive':
                    sub_evid = 'Exclusion'
                evidence = evidence + '({})'.format(sub_evid)
            else:
                assert False, "[X] Unkown evidence type (Sample {}): {}".format(li, evidence)

            evidence = evidence.replace('fact', 'Fact')
            evidence = evidence.replace('inside_scene', 'Inside Scene')

            evide_counter[evidence].append(correct)
            if evidence.startswith('Fact'):
                evide_counter['Fact'].append(correct)

            id2types[char_id]['etype'].append(evidence)



    # print
    print('')
    print('#####  Dependnce  #####')
    for name, results in dtype_counter.items():
        ratio = len(results) / len(lines)
        maccu = sum(results) / len(results) if len(results) > 0 else 0
        print('  * {:30s} {:2.2f}%'.format(name, ratio * 100))

    print('')
    print('#####  Reasoning  #####')
    for name, results in rtype_counter.items():
        ratio = len(results) / len(lines)
        maccu = sum(results) / len(results) if len(results) > 0 else 0
        print('  * {:30s} {:2.2f}%'.format(name, ratio * 100))

    print('')
    print('##### Evidence #####')
    for name, results in evide_counter.items():
        ratio = len(results) / len(lines)
        maccu = sum(results) / len(results) if len(results) > 0 else 0
        print('  * {:30s} {:2.2f}%'.format(name, ratio * 100))
    print('\n')