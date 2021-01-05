from __future__ import print_function, division

import os
import json
import argparse

def file_bleu(ref_files, hyp_file, detok=True, verbose=False):
    bleus = multi_file_bleu(ref_files, [hyp_file], detok=detok, verbose=verbose)
    return bleus[0]


def multi_file_bleu(ref_files, hyp_files, detok=True, verbose=False):
    '''
    This is to get the average BLEU for hyp among ref0, ref1, ref2, ...
    :param hyp_files: a list of filenames for hypothesis
    :param ref_files: a list of filenames for references
    :return: print a bleu score
    '''
    from efficiency.function import shell

    # check for wrong input of ref_list, and correct it
    if isinstance(ref_files, str):
        ref_files = [ref_files]

    ref_files, hyp_files = \
        preprocess_files(ref_files, hyp_files, verbose=verbose)

    outputs = []
    script = BLEU_DETOK_FILE if detok else BLEU_FILE

    for hyp in hyp_files:
        cmd = 'perl {script} {refs} < {hyp} '.format(
            script=script,refs=' '.join(ref_files), hyp=hyp)
        if verbose: print('[cmd]', cmd)
        stdout, stderr = shell(cmd)

        bleu_prefix = 'BLEU = '

        if verbose and not stdout.startswith(bleu_prefix):
            print(stdout)

        if bleu_prefix in stdout:
            num = stdout.split(bleu_prefix, 1)[-1].split(',')[0]
            output = float(num)
        else:
            # if stdout.startswith('Illegal division by zero'):
            output = -1

        outputs += [output]

        if verbose:
            print('{}-ref bleu for {}: {}'.format(len(ref_files), hyp, output))
    return outputs


def list_bleu(refs, hyp, detok=True, tmp_dir=TMP_DIR, verbose=False, return_files=False):
    # check for wrong input of ref_list, and correct it
    for ref_list in refs:
        if isinstance(ref_list, str):
            refs = [refs]
            break

    import uuid
    uid = str(uuid.uuid4())
    folder = os.path.join(tmp_dir, uid)

    try:
        os.mkdir(folder)
        ref_files, hyp_files = lists2files(refs, [hyp], tmp_dir=folder)

        bleus = multi_file_bleu(ref_files=ref_files, hyp_files=hyp_files,
                                detok=detok, verbose=verbose)
        bleu = bleus[0]
    finally:
        if not return_files:
            import shutil

            shutil.rmtree(folder)

    if return_files:
        hyp_file = hyp_files[0]
        return bleu, ref_files, hyp_file
    else:
        return bleu

def multi_list_bleu(refs, hyps, detok=True, tmp_dir=TMP_DIR, verbose=False, return_files=False):
    # check for wrong input of ref_list, and correct it
    for ref_list in refs:
        if isinstance(ref_list, str):
            refs = [refs]
            break

    import uuid
    uid = str(uuid.uuid4())
    folder = os.path.join(tmp_dir, uid)

    try:
        os.mkdir(folder)
        ref_files, hyp_files = lists2files(refs, hyps, tmp_dir=folder)

        bleus = multi_file_bleu(ref_files=ref_files, hyp_files=hyp_files,
                                detok=detok, verbose=verbose)
    finally:
        if not return_files:
            import shutil

            shutil.rmtree(folder)

    if return_files:
        return bleus, ref_files, hyp_files
    else:
        return bleus

def lists2files(refs, hyps, tmp_dir=TMP_DIR):
    def _list2file(sents, file):
        writeout = '\n'.join(sents) + '\n'
        with open(file, 'w') as f:
            f.write(writeout)

    ref_files = [os.path.join(tmp_dir, 'ref{}.txt'.format(ref_ix))
                 for ref_ix, _ in enumerate(refs)]
    hyp_files = [os.path.join(tmp_dir, 'hyp{}.txt'.format(hyp_ix))
                 for hyp_ix, _ in enumerate(hyps)]

    _ = [_list2file(*item) for item in zip(refs, ref_files)]
    _ = [_list2file(*item) for item in zip(hyps, hyp_files)]
    return ref_files, hyp_files


def preprocess_files(ref_files, hyp_files, verbose=False):
    # Step 1. Check whether all files exist
    valid_refs = [f for f in ref_files if os.path.isfile(f)]
    valid_hyps = [f for f in hyp_files if os.path.isfile(f)]
    if verbose:
        print('[Info] Valid Reference Files: {}'.format(str(valid_refs)))
        print('[Info] Valid Hypothesis Files: {}'.format(str(valid_hyps)))

    # Step 2. Check whether all files has the same num of lines
    num_lines = []
    files = valid_refs + valid_hyps
    for file in files:
        with open(file) as f:
            lines = [line.strip() for line in f]
            num_lines += [len(lines)]
    if len(set(num_lines)) != 1:
        raise RuntimeError("[Error] File lengths are different! list(zip(files, num_lines)): {}".format(list(zip(files, num_lines))))

    if verbose:
        print("[Info] #lines in each file: {}".format(num_lines[0]))

    # Step 3. detokenization
    valid_refs = detok_files(valid_refs, tmp_dir=TMP_DIR, file_prefix='ref_dtk', verbose=verbose)
    valid_hyps = detok_files(valid_hyps, tmp_dir=TMP_DIR, file_prefix='hyp_dtk', verbose=verbose)

    return valid_refs, valid_hyps


def detok_files(files_in, tmp_dir=TMP_DIR, file_prefix='detok', verbose=False):
    '''
    This is to detokenize all files
    :param files: a list of filenames
    :return: a list of files after detokenization
    '''
    files_out = []
    if not os.path.isdir(tmp_dir): os.mkdir(tmp_dir)
    for ix, f_in in enumerate(files_in):
        f_out = os.path.join(tmp_dir, '{}{}.txt'.format(file_prefix, ix))
        files_out.append(f_out)

        cmd = 'perl {DETOK_FILE} -l en < {f_in} > {f_out} 2>/dev/null'.format(
            DETOK_FILE=DETOK_FILE, f_in=f_in, f_out=f_out)
        if verbose: print('[cmd]', cmd)
        os.system(cmd)
    return files_out



class Evaluate():


  def __init__(self, expected, actual):
    
    self.gold_triples = self.get_triples(expected)
    self.pred_triples = self.get_triples(actual)
    self.cities, self.teams = set(), set()
    ec = {} # equivalence classes
    for team in full_names:
        pieces = team.split()
        if len(pieces) == 2:
            ec[team] = [pieces[0], pieces[1]]
            self.cities.add(pieces[0])
            self.teams.add(pieces[1])
        elif pieces[0] == "Portland": # only 2-word team
            ec[team] = [pieces[0], " ".join(pieces[1:])]
            self.cities.add(pieces[0])
            self.teams.add(" ".join(pieces[1:]))
        else: # must be a 2-word City
            ec[team] = [" ".join(pieces[:2]), pieces[2]]
            self.cities.add(" ".join(pieces[:2]))
            self.teams.add(pieces[2])


  def same_ent(self, e1, e2):
    if e1 in self.cities or e1 in self.teams or e2 in self.cities or e2 in self.teams:
        return e1 == e2 or any((e1 in fullname and e2 in fullname for fullname in full_names))
    else:
        return e1 in e2 or e2 in e1

  def int_value(self, input):
    try: 
        value = int(input)
        # a_number = True
    except ValueError:
        value = text2num(input)
        
    return value

  def trip_match(self, t1, t2):
    return self.int_value(t1[1]) == self.int_value(t2[1]) and t1[2] == t2[2] and self.same_ent(t1[0], t2[0])


  def dedup_triples(self, triplist):
    """
    this will be inefficient but who cares. Why no python set with comparator REEEEEEE!
    """
    dups = set()
    for i in range(1, len(triplist)):
      if any(self.trip_match(triplist[i], triplist[j]) for j in range(i)):
          dups.add(i)
          
    return [thing for i, thing in enumerate(triplist) if i not in dups]

  def get_triples(self, file):
    all_triples = []
    curr = []
    with open(file) as f:
        for line in f:
            if line.isspace():
                all_triples.append(self.dedup_triples(curr))
                curr = []
            else:
                pieces = line.strip().split('|')
                curr.append(tuple(pieces))
    if len(curr) > 0:
        all_triples.append(self.dedup_triples(curr))
    return all_triples

  # THis is used
  def calc_precrec(self):
    total_tp, total_predicted, total_gold = 0, 0, 0
    assert len(self.gold_triples) == len(self.pred_triples)
    for i, triplist in enumerate(self.pred_triples):
        tp = sum((1 for j in range(len(triplist))
                    if any(self.trip_match(triplist[j], self.gold_triples[i][k])
                           for k in range(len(self.gold_triples[i])))))
        total_tp += tp
        total_predicted += len(triplist)
        total_gold += len(self.gold_triples[i])
    avg_prec = total_tp/total_predicted
    avg_rec = total_tp/total_gold
    print("totals:", total_tp, total_predicted, total_gold)
    print("prec:", avg_prec, "rec:", avg_rec)
    return avg_prec, avg_rec


  def norm_dld(self, l1, l2):
    ascii_start = 0
    # make a string for l1
    # all triples are unique...
    s1 = ''.join((chr(ascii_start+i) for i in range(len(l1))))
    s2 = ''
    next_char = ascii_start + len(s1)
    for j in range(len(l2)):
        found = None
        #next_char = chr(ascii_start+len(s1)+j)
        for k in range(len(l1)):
            if self.trip_match(l2[j], l1[k]):
                found = s1[k]
                #next_char = s1[k]
                break
        if found is None:
            s2 += chr(next_char)
            next_char += 1
            assert next_char <= 128
        else:
            s2 += found
    # return 1- , since this thing gives 0 to perfect matches etc
    return 1.0-normalized_damerau_levenshtein_distance(s1, s2)


#  THis is used
  def calc_dld(self):
    
    assert len(self.gold_triples) == len(self.pred_triples)
    total_score = 0
    for i, triplist in enumerate(self.pred_triples):
        total_score += self.norm_dld(triplist, self.gold_triples[i])
    avg_score = total_score/len(self.pred_triples)
    print("avg score:", avg_score)
    return avg_score


  def evaluate_gold_to_pred(self):
    self.calc_precrec()
    self.calc_dld()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-refs', default=['data/ref0.txt', 'data/ref1.txt'],
                        nargs='+', type=str,
                        help='a list of filenames for reference files, separated by space')
    parser.add_argument('-hyps', default=['data/hyp0.txt'], nargs='+', type=str,
                        help='a list of filenames for hypothesis files, separated by space')
    parser.add_argument('-data_dir', default='', type=str,
                        help='directory to save temporary outputs')
    parser.add_argument('-evalGold', default=['data/gold.txt'], nargs='+', type=str,
                        help='a list of filenames for gold files, separated by space')
    parser.add_argument('-evalPred', default=['data/pred.txt'], nargs='+', type=str,
                        help='a list of filenames for prediction files, separated by space')
    parser.add_argument('-verbose', action='store_true',
                        help='whether to allow printing out logs')
    

    args = parser.parse_args()
    main(args)