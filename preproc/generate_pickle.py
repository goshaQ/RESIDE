import argparse
import json
import pdb
import pickle
import random
import re
import sys
import time
from collections import defaultdict

import tensorflow as tf
from bert import run_classifier_with_tfhub
import tensorflow_hub as hub

import numpy as np
from orderedset import OrderedSet
from scipy.spatial.distance import cdist

sys.path.append('./')

from helper import mergeList, createModel, createTokenizer, prepareInputBERT, \
    buildPhr2ELMOGraph, getPhr2ELMO, buildPhr2BERTGraph, getPhr2BERT

parser = argparse.ArgumentParser(description='Main Preprocessing program')
parser.add_argument('--test', dest="FULL", action='store_false')
parser.add_argument('--pos', dest="MAX_POS", default=60, type=int,
                    help='Max position to consider for positional embeddings')
parser.add_argument('--mvoc', dest="MAX_VOCAB", default=150000, type=int,
                    help='Maximum vocabulary to consider')
parser.add_argument('--maxw', dest="MAX_WORDS", default=100, type=int)
parser.add_argument('--minw', dest="MIN_WORDS", default=5, type=int)
parser.add_argument('--num', dest="num_procs", default=40, type=int)
parser.add_argument('--thresh', dest="thresh", default=0.65, type=float)
parser.add_argument('--nfinetype', dest='wFineType', action='store_false')
parser.add_argument('--metric', default='cosine')
parser.add_argument('--data', default='riedel')
parser.add_argument('--log_steps', default=10000, type=int, help='Logging frequency in steps')
parser.add_argument('--seed', dest="seed", default=1234, type=int, help='Seed for randomization')

# Change the below two arguments together
parser.add_argument('--max_seq_length', default=48, type=int,
                    help='The length of sequence tokens')
parser.add_argument('--embed-type', default='ELMO',
                    help='Embeddings type (ELMO or BERT)')
parser.add_argument('--pretrained-elmo-model',
                    default='https://tfhub.dev/google/elmo/2',
                    help='Path to the pretrained ELMO model on TF Hub')
parser.add_argument('--pretrained-bert-model',
                    default='https://tfhub.dev/google/bert_uncased_L-12_H-768_A-12/1',
                    help='Path to the pretrained BERT model on TF Hub')

# Below arguments can be used for testing processing script (process a part of data instead of full)
parser.add_argument('--sample', dest='FULL', action='store_false',
                    help='To process the entire data or a sample of it')
parser.add_argument('--sample_size', dest='sample_size', default=200, type=int,
                    help='Sample size to use for testing processing script')
args = parser.parse_args()

print('Starting Data Pre-processing script...')

with open(f'./side_info/entity_type/{args.data}/type_info.json') as f:
    ent2type = json.load(f)
with open(f'./side_info/relation_alias/{args.data}/relation_alias_from_wikidata_ppdb_extended.json') as f:
    rel2alias = json.load(f)
with open(f'./preproc/{args.data}_relation2id.json') as f:
    rel2id = json.load(f)

# id2rel = dict([(v, k) for k, v in rel2id.items()])
alias2rel = defaultdict(set)
alias2id = {}

if args.embed_type == 'ELMO':
    embed_model = createModel(args.pretrained_elmo_model)
    tokenizer = None
elif args.embed_type == 'BERT':
    embed_model = createModel(args.pretrained_bert_model)
    tokenizer = createTokenizer(args.pretrained_bert_model)
else:
    raise ValueError('Incorrect embeddings type, try either BERT or ELMO')

sess = tf.Session()
sess.run(tf.global_variables_initializer())
tf.set_random_seed(args.seed)
random.seed(args.seed)
np.random.seed(args.seed)

for rel, aliases in rel2alias.items():
    for alias in aliases:
        if alias in alias2id:
            alias2rel[alias2id[alias]].add(rel)
        else:
            alias2id[alias] = len(alias2id)
            alias2rel[alias2id[alias]].add(rel)

# id2alias = dict([(v, k) for k, v in alias2id.items()])

temp = sorted(alias2id.items(), key=lambda x: x[1])
alias_list, _ = zip(*temp)

if args.embed_type == 'ELMO':
    elmo_inputs, elmo_outputs = buildPhr2ELMOGraph(embed_model)
    alias_embed = getPhr2ELMO(elmo_inputs, elmo_outputs, alias_list, sess)
elif args.embed_type == 'BERT':
    alias_list = prepareInputBERT(tokenizer, alias_list, args.max_seq_length)

    bert_inputs, bert_outputs = buildPhr2BERTGraph(embed_model, args.max_seq_length)
    alias_embed = getPhr2BERT(bert_inputs, bert_outputs, alias_list, sess)


data = {
    'train': [],
    'test': []
}


def read_file(file_path):
    dataset = {}

    with open(file_path) as f:
        for k, line in enumerate(f):
            bag = json.loads(line.strip())

            wrds_list = []
            pos1_list = []
            pos2_list = []
            sub_pos_list = []
            obj_pos_list = []
            dep_links_list = []
            phrase_list = []

            for sent in bag['sents']:

                if len(bag['sub']) > len(bag['obj']):
                    sub_idx = [i for i, e in enumerate(sent['rsent'].split()) if e == bag['sub']]
                    sub_start_off = [len(' '.join(sent['rsent'].split()[0: idx])) + (1 if idx != 0 else 0)
                                     for idx in sub_idx]
                    if not sub_start_off:
                        sub_start_off = [m.start() for m in
                                         re.finditer(bag['sub'].replace('_', ' '),
                                                     sent['rsent'].replace('_', ' '))]
                    reserve_span = [(start_off, start_off + len(bag['sub'])) for start_off in sub_start_off]

                    obj_idx = [i for i, e in enumerate(sent['rsent'].split()) if e == bag['obj']]
                    obj_start_off = [len(' '.join(sent['rsent'].split()[0: idx])) + (1 if idx != 0 else 0)
                                     for idx in obj_idx]
                    if not obj_start_off:
                        obj_start_off = [m.start() for m in
                                         re.finditer(bag['obj'].replace('_', ' '),
                                                     sent['rsent'].replace('_', ' '))]
                    obj_start_off = [off for off in obj_start_off if
                                     all([off < spn[0] or off > spn[1] for spn in reserve_span])]
                else:
                    obj_idx = [i for i, e in enumerate(sent['rsent'].split()) if e == bag['obj']]
                    obj_start_off = [len(' '.join(sent['rsent'].split()[0: idx])) + (1 if idx != 0 else 0)
                                     for idx in obj_idx]
                    if not obj_start_off:
                        obj_start_off = [m.start() for m in
                                         re.finditer(bag['obj'].replace('_', ' '),
                                                     sent['rsent'].replace('_', ' '))]
                    reserve_span = [(start_off, start_off + len(bag['obj'])) for start_off in obj_start_off]

                    sub_idx = [i for i, e in enumerate(sent['rsent'].split()) if e == bag['sub']]
                    sub_start_off = [len(' '.join(sent['rsent'].split()[0: idx])) + (1 if idx != 0 else 0)
                                     for idx in sub_idx]
                    if not sub_start_off:
                        sub_start_off = [m.start() for m in
                                         re.finditer(bag['sub'].replace('_', ' '),
                                                     sent['rsent'].replace('_', ' '))]
                    sub_start_off = [off for off in sub_start_off if
                                     all([off < spn[0] or off > spn[1] for spn in reserve_span])]

                sub_off = [(start_off, start_off + len(bag['sub']), 'sub') for start_off in sub_start_off]
                obj_off = [(start_off, start_off + len(bag['obj']), 'obj') for start_off in obj_start_off]

                if sub_off == [] or obj_off == [] or 'corenlp' not in sent:
                    continue
                spans = [sub_off[0]] + [obj_off[0]]
                off_begin, off_end, _ = zip(*spans)

                tid_map, tid2wrd = defaultdict(dict), defaultdict(list)

                tok_idx = 1
                sub_pos, obj_pos = None, None
                dep_links = []

                for s_n, corenlp_sent in enumerate(sent['corenlp']['sentences']):  # Iterating over sentences

                    i, tokens = 0, corenlp_sent['tokens']

                    while i < len(tokens):
                        if tokens[i]['characterOffsetBegin'] in off_begin:
                            _, end_offset, identity = spans[off_begin.index(tokens[i]['characterOffsetBegin'])]

                            if identity == 'sub':
                                sub_pos = tok_idx - 1  # Indexing starts from 0
                            else:
                                obj_pos = tok_idx - 1

                            while i < len(tokens) and tokens[i]['characterOffsetEnd'] <= end_offset:
                                tid_map[s_n][tokens[i]['index']] = tok_idx
                                tid2wrd[tok_idx].append(tokens[i]['originalText'])
                                i += 1

                            tok_idx += 1
                        else:
                            tid_map[s_n][tokens[i]['index']] = tok_idx
                            tid2wrd[tok_idx].append(tokens[i]['originalText'])

                            i += 1
                            tok_idx += 1

                if sub_pos is None or obj_pos is None:
                    print('Skipped entry!!')
                    print('{} | {} | {}'.format(bag['sub'], bag['obj'], sent['rsent']))
                    pdb.set_trace()
                    continue

                wrds = ['_'.join(e).lower() for e in tid2wrd.values()]
                pos1 = [i - sub_pos for i in range(tok_idx - 1)]  # tok_id = (number of tokens + 1)
                pos2 = [i - obj_pos for i in range(tok_idx - 1)]

                phrases = set()
                if sent['openie'] is not None:
                    for corenlp_sent in sent['openie']['sentences']:
                        for openie in corenlp_sent['openie']:
                            if openie['subject'].lower() == bag['sub'].replace('_', ' ') \
                                    and openie['object'].lower() == bag['obj'].replace('_', ' '):
                                phrases.add(openie['relation'])

                if abs(sub_pos - obj_pos) < 5:
                    middle_phr = ' '.join(sent['rsent'].split()[min(sub_pos, obj_pos) + 1: max(sub_pos, obj_pos)])
                    phrases.add(middle_phr)
                else:
                    middle_phr = ''

                for s_n, corenlp_sent in enumerate(sent['corenlp']['sentences']):
                    dep_edges = corenlp_sent['basicDependencies']
                    for dep in dep_edges:
                        if dep['governor'] == 0 or dep['dependent'] == 0:
                            continue  # Ignore ROOT
                        # -1, because indexing starts from 0
                        dep_links.append((tid_map[s_n][dep['governor']] - 1, tid_map[s_n][dep['dependent']] - 1, 0, 1))

                right_nbd_phrase, left_nbd_phrase, mid_phrase = set(), set(), set()
                for edge in dep_links:
                    if edge[0] == sub_pos or edge[0] == obj_pos:
                        if min(sub_pos, obj_pos) < edge[1] < max(sub_pos, obj_pos):
                            mid_phrase.add(wrds[edge[1]])
                        elif edge[1] < min(sub_pos, obj_pos):
                            left_nbd_phrase.add(wrds[edge[1]])
                        else:
                            right_nbd_phrase.add(wrds[edge[1]])

                    if edge[1] == sub_pos or edge[1] == obj_pos:
                        if min(sub_pos, obj_pos) < edge[0] < max(sub_pos, obj_pos):
                            mid_phrase.add(wrds[edge[0]])
                        elif edge[0] < min(sub_pos, obj_pos):
                            left_nbd_phrase.add(wrds[edge[0]])
                        else:
                            right_nbd_phrase.add(wrds[edge[0]])

                left_nbd_phrase = ' '.join(list(left_nbd_phrase - {bag['sub'], bag['obj']}))
                right_nbd_phrase = ' '.join(list(right_nbd_phrase - {bag['sub'], bag['obj']}))
                mid_phrase = ' '.join(list(mid_phrase))

                phrases.add(left_nbd_phrase)
                phrases.add(right_nbd_phrase)
                phrases.add(middle_phr)
                phrases.add(mid_phrase)

                if args.embed_type == 'ELMO':
                    wrds_list.append(sent['rsent'])
                else:
                    wrds_list.append(wrds)
                pos1_list.append(pos1)
                pos2_list.append(pos2)
                sub_pos_list.append(sub_pos)
                obj_pos_list.append(obj_pos)
                dep_links_list.append(dep_links)
                phrase_list.append(list(phrases - {''}))

            dataset[k] = {
                'sub': bag['sub'],
                'obj': bag['obj'],
                'rels': bag['rel'],
                'phrase_list': phrase_list,
                'sub_pos_list': sub_pos_list,
                'obj_pos_list': obj_pos_list,
                'wrds_list': wrds_list,
                'pos1_list': pos1_list,
                'pos2_list': pos2_list,
                'sub_type': ent2type[bag['sub_id']],
                'obj_type': ent2type[bag['obj_id']],
                'dep_links_list': dep_links_list,
            }

            if k % args.log_steps == 0 and k > 0:
                print('Completed {}, {}'.format(k, time.strftime("%d_%m_%Y %H:%M:%S")))
            if not args.FULL and k > args.sample_size:
                break
    return list(dataset.values())


print('Reading train bags')
data['train'] = read_file('data/{}_train_bags.json'.format(args.data))
print('Reading test bags')
data['test'] = read_file('data/{}_test_bags.json'.format(args.data))

print('Bags processed: Train:{}, Test:{}'.format(len(data['train']), len(data['test'])))

"""*************************** REMOVE OUTLIERS **************************"""


def remove_outliers(dataset):
    del_cnt = 0
    for i in range(len(dataset) - 1, -1, -1):
        bag = dataset[i]

        for j in range(len(bag['wrds_list']) - 1, -1, -1):
            dataset[i]['wrds_list'][j] = dataset[i]['wrds_list'][j][:args.MAX_WORDS]
            dataset[i]['pos1_list'][j] = dataset[i]['pos1_list'][j][:args.MAX_WORDS]
            dataset[i]['pos2_list'][j] = dataset[i]['pos2_list'][j][:args.MAX_WORDS]
            dataset[i]['dep_links_list'][j] = [e for e in dataset[i]['dep_links_list'][j] if
                                               e[0] < args.MAX_WORDS and e[1] < args.MAX_WORDS]
            if len(dataset[i]['dep_links_list'][j]) == 0:
                del dataset[i]['dep_links_list'][j]  # Delete sentences with no dependency links

        if len(dataset[i]['wrds_list']) == 0 or len(dataset[i]['dep_links_list']) == 0:
            del dataset[i]
            del_cnt += 1
            continue
    
    print('Bags deleted {}'.format(del_cnt))


remove_outliers(data['train'])
remove_outliers(data['test'])


"""*************************** GET PROBABLE RELATIONS **************************"""


def get_alias2rel(phr_embed, args):
    dist = cdist(phr_embed, alias_embed, metric=args.metric)

    rels = set()
    for i, cphr in enumerate(np.argmin(dist, 1)):
        if dist[i, cphr] < args.thresh:
            rels |= alias2rel[cphr]
    return [rel2id[r] for r in rels if r in rel2id]


def get_alias2rel_batch(phr_list, args, batch_split_indices, bag_split_indices):
    if args.embed_type == 'ELMO':
        phr_embed = getPhr2ELMO(elmo_inputs, elmo_outputs, phr_list, sess)
    else:
        phr_embed = getPhr2BERT(bert_inputs, bert_outputs, phr_list, sess)
    # print(phr_embed.shape)

    # embeds for multiple bags, should be split
    phr_embed = np.split(phr_embed, batch_split_indices)[:-1]
    # print((len(phr_embed), list(map(lambda x: x.shape, phr_embed)))

    res_list = []
    for i in range(len(phr_embed)):
        prob_rels = []
        phr_embed_i = np.split(phr_embed[i], bag_split_indices[i], axis=0)[:-1]
        for phr_embed_ij in phr_embed_i:
            prob_rels.append(get_alias2rel(phr_embed_ij, args))

        res_list.append(prob_rels)

    return res_list


def get_prob_rels(data, args, batch_size=1280):
    res_list = []

    n_phrases = 0
    phr_lists_batch = []
    batch_split_indices = []
    bag_split_indices = []

    for i in range(len(data)):
        phr_lists = data[i]['phrase_list']
        bag_split = np.cumsum(list(map(lambda x: len(x), phr_lists)))
        bag_split_indices.append(bag_split)
        n_phrases += bag_split[-1]

        phr_lists = mergeList(phr_lists)
        phr_lists_batch.append(phr_lists)

        if batch_split_indices:
            batch_split_indices.append(batch_split_indices[-1] + len(phr_lists))
        else:
            batch_split_indices.append(len(phr_lists))

        if n_phrases > batch_size:
            phr_list_batch = mergeList(phr_lists_batch)
            if args.embed_type == 'BERT':
                phr_list_batch = prepareInputBERT(tokenizer, phr_list_batch, args.max_seq_length)

            rel_list_batch = get_alias2rel_batch(phr_list_batch, args, batch_split_indices, bag_split_indices)
            res_list.append(rel_list_batch)
            del phr_list_batch, phr_lists_batch

            print('Completed {}, {}'.format(i+1, time.strftime("%d_%m_%Y %H:%M:%S")))

            n_phrases = 0
            phr_lists_batch = []
            batch_split_indices = []
            bag_split_indices = []

    phr_list_batch = mergeList(phr_lists_batch)
    if args.embed_type == 'BERT':
        phr_list_batch = prepareInputBERT(tokenizer, phr_list_batch, args.max_seq_length)

    rel_list_batch = get_alias2rel_batch(phr_list_batch, args, batch_split_indices, bag_split_indices)
    res_list.append(rel_list_batch)
    del phr_list_batch, phr_lists_batch

    print('Completed {}, {}'.format(len(data), time.strftime("%d_%m_%Y %H:%M:%S")))

    res_list = mergeList(res_list)

    return res_list


print('Computing probable relations for training bags...')
results = get_prob_rels(data['train'], args)
for i in range(len(results)):
    data['train'][i]['prob_rels'] = results[i]
    if len(data['train'][i]['prob_rels']) != len(data['train'][i]['phrase_list']):
        pdb.set_trace()
del results


print('Computing probable relations for test bags...')
results = get_prob_rels(data['test'], args)
for i in range(len(results)):
    data['test'][i]['prob_rels'] = results[i]
    if len(data['test'][i]['prob_rels']) != len(data['test'][i]['phrase_list']):
        pdb.set_trace()
del results

sess.close()

"""*************************** WORD 2 ID MAPPING **************************"""


def getIdMap(vals, begin_idx=0):
    ele2id = {}
    for id, ele in enumerate(vals):
        ele2id[ele] = id + begin_idx
    return ele2id


if args.embed_type == 'ELMO':
    vocab = {}
else:
    vocab = tokenizer.vocab
voc2id = vocab
id2voc = dict([(v, k) for k, v in voc2id.items()])

type_vocab = OrderedSet(['NONE'] + list(set(mergeList(ent2type.values()))))
type2id = getIdMap(type_vocab)

print('Chosen Vocabulary:\t{}'.format(len(vocab)))
print('Type Number:\t{}'.format(len(type2id)))

"""******************* CONVERTING DATA IN TENSOR FORM **********************"""


def getId(wrd, wrd2id, def_val='NONE'):
    if wrd in wrd2id:
        return wrd2id[wrd]
    else:
        return wrd2id[def_val]


def posMap(pos):
    if pos < -args.MAX_POS:
        return 0
    elif pos > args.MAX_POS:
        return (args.MAX_POS + 1) * 2
    else:
        return pos + (args.MAX_POS + 1)


def procData(data, split='train'):
    res_list = []

    for bag in data:
        # Labels will be K - hot
        res = {
            'X': bag['wrds_list'] if args.embed_type == 'ELMO'
            else [[getId(wrd, voc2id, '[UNK]') for wrd in wrds] for wrds in bag['wrds_list']],
            'Pos1': [[posMap(pos) for pos in pos1] for pos1 in bag['pos1_list']],
            'Pos2': [[posMap(pos) for pos in pos2] for pos2 in bag['pos2_list']],
            'Y': bag['rels'],
            'SubType': [getId(typ, type2id, 'NONE') for typ in bag['sub_type']],
            'ObjType': [getId(typ, type2id, 'NONE') for typ in bag['obj_type']],
            'SubPos': bag['sub_pos_list'],
            'ObjPos': bag['obj_pos_list'],
            'ProbY': bag['prob_rels'],
            'DepEdges': bag['dep_links_list']
        }

        if len(res['X']) != len(res['ProbY']):
            print('Skipped One')
            continue

        res_list.append(res)

    return res_list


final_data = {
    'train': procData(data['train'], 'train'),
    'test': procData(data['test'], 'test'),
    'voc2id': voc2id,
    'id2voc': id2voc,
    'type2id': type2id,
    'rel2id': rel2id,
    'max_pos': (args.MAX_POS + 1) * 2 + 1
}

embed_type = 'elmo' if args.embed_type == 'ELMO' else 'bert'
print(f'Saving pickle file as "data/{args.data}_processed_{embed_type}.pkl"')
pickle.dump(final_data, open(f'data/{args.data}_processed_{embed_type}.pkl', 'wb'))
