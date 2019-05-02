import argparse
import json
import sys
import time
from collections import defaultdict

sys.path.append('./')


def main(args):
    rel2id = json.loads(open('./preproc/{}_relation2id.json'.format(args.data)).read())

    print('Constructing training bags...')
    train_data = defaultdict(lambda: {'rels': defaultdict(list)})
    with open('./data/{}_train.json'.format(args.data)) as f:
        for i, line in enumerate(f):
            data = json.loads(line.strip())

            _id = '{}_{}'.format(data['sub'], data['obj'])
            train_data[_id]['sub_id'] = data['sub_id']
            train_data[_id]['obj_id'] = data['obj_id']
            train_data[_id]['sub'] = data['sub']
            train_data[_id]['obj'] = data['obj']

            train_data[_id]['rels'][rel2id.get(data['rel'], rel2id['NA'])].append({
                'sent': data['sent'],
                'corenlp': data['corenlp'],
                'rsent': data['rsent'],
                'openie': data['openie'],
            })

            if (i + 1) % args.log_steps == 0:
                print('Completed {}, {}'.format(i + 1, time.strftime("%d_%m_%Y %H:%M:%S")))

    count = 0
    with open('./data/{}_train_bags.json'.format(args.data), 'w') as f:
        for _id, data in train_data.items():
            for rel, sents in data['rels'].items():
                entry = {
                    'sub': data['sub'],
                    'obj': data['obj'],
                    'sub_id': data['sub_id'],
                    'obj_id': data['obj_id'],
                    'sents': sents,
                    'rel': [rel],
                }
                f.write(json.dumps(entry) + '\n')

                count += 1
                if count % args.log_steps == 0:
                    print('Writing Completed {}, {}'.format(count, time.strftime("%d_%m_%Y %H:%M:%S")))
    del train_data

    print('Constructing test bags...')
    test_data = defaultdict(lambda: {'sents': [], 'rels': set()})
    with open('./data/{}_test.json'.format(args.data)) as f:
        for i, line in enumerate(f):
            data = json.loads(line.strip())

            _id = '{}_{}'.format(data['sub'], data['obj'])
            test_data[_id]['sub_id'] = data['sub_id']
            test_data[_id]['obj_id'] = data['obj_id']
            test_data[_id]['sub'] = data['sub']
            test_data[_id]['obj'] = data['obj']
            test_data[_id]['rels'].add(rel2id.get(data['rel'], rel2id['NA']))

            test_data[_id]['sents'].append({
                'sent': data['sent'],
                'corenlp': data['corenlp'],
                'rsent': data['rsent'],
                'openie': data['openie'],
            })

            if (i + 1) % args.log_steps == 0:
                print('Completed {}, {}'.format(i + 1, time.strftime("%d_%m_%Y %H:%M:%S")))

    count = 0
    with open('./data/{}_test_bags.json'.format(args.data), 'w') as f:
        for _id, data in test_data.items():
            entry = {
                'sub': data['sub'],
                'obj': data['obj'],
                'sub_id': data['sub_id'],
                'obj_id': data['obj_id'],
                'sents': sents,
                'rel': [rel],
            }
            f.write(json.dumps(entry) + '\n')

            count += 1
            if count % args.log_steps == 0:
                print('Writing Completed {}, {}'.format(count, time.strftime("%d_%m_%Y %H:%M:%S")))
    del test_data


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('-data', default='riedel')
    parser.add_argument('-log_steps', default=10000, type=int,
                        help='Logging frequency in steps')
    args = parser.parse_args()

    main(args)
