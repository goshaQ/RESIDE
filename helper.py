import itertools
import json
import logging
import logging.config
import os
import pathlib
import sys

import bert
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
from nltk.tokenize import word_tokenize

# Set precision for numpy
np.set_printoptions(precision=4)


def getBERTEmbeddings(model, wrd_list):
    """
    Gives embedding for each word in input_list

    Parameters
    ----------
    model:		BERT model
    input_list:	List of inputs for which embedding is required

    Returns
    -------
    embed_matrix:	(len(input_list) x BERT hidden dim) matrix containing embedding
                    for each word in input_list in the same order
    """
    embed_list = []

    embed = model(inputs=prepareInputBERT(wrd_list),
                  as_dict=True,
                  signature='tokens')

    return np.array(embed_list, dtype=np.float32)


def getGloveEmbeddings(model, wrd_list, embed_dims):
    """
    Gives embedding for each word in wrd_list
    Parameters
    ----------
    model:		Word2vec model
    wrd_list:	List of words for which embedding is required
    embed_dims:	Dimension of the embedding
    Returns
    -------
    embed_matrix:	(len(wrd_list) x embed_dims) matrix containing embedding for each word in wrd_list in the same order
    """
    embed_list = []

    for wrd in wrd_list:
        if wrd in model.vocab:
            embed_list.append(model.word_vec(wrd))
        else:
            embed_list.append(np.random.randn(embed_dims))  # Generates a random vector for words not in vocab

    return np.array(embed_list, dtype=np.float32)


def prepareInputBERT(tokenizer, sentences, max_seq_length):
    """
    Converts a set of sentences to the format expected by BERT model

    Parameters
    ----------
    tokenizer:	BERT tokenizer
    sentences:	Set of sentences to be preprocessed

    Returns
    -------
    preprocessed:  	dictionary with input_ids, input_mask, and segment_ids
    """

    list_input_ids = []
    list_input_mask = []
    list_segment_ids = []

    for sentence in sentences:
        tokenized_sentence = tokenizer.tokenize(sentence)

        tokens = []
        segment_ids = []
        tokens.append('[CLS]')
        segment_ids.append(0)
        for token in tokenized_sentence:
            tokens.append(token)
            segment_ids.append(0)
        tokens.append('[SEP]')
        segment_ids.append(0)

        input_ids = tokenizer.convert_tokens_to_ids(tokens)
        input_mask = [1] * len(input_ids)

        while len(input_ids) < max_seq_length:
            input_ids.append(0)
            input_mask.append(0)
            segment_ids.append(0)

        # Check that the length of sequences is not higher than `max_seq_length`
        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length
        
        list_input_ids.append(input_ids)
        list_input_mask.append(input_mask)
        list_segment_ids.append(segment_ids)

    return {
        'input_ids': list_input_ids,
        'input_mask': list_input_mask,
        'segment_ids': list_segment_ids,
    }


def createModel(bert_model_hub, trainable=False):
    """
    Get model from the Hub module

    Parameters
    ----------
    bert_model_hub: Path to the pretrained module.

    Returns
    -------
    model:  model object
    """
    return hub.Module(bert_model_hub, trainable=trainable)


def createTokenizer(bert_model_hub):
    """
    Get the vocab file and casing info from the Hub module

    Parameters
    ----------
    bert_model_hub: Path to the pretrained BERT module.

    Returns
    -------
    tokenizer: FullTokenizer object
    """
    with tf.Graph().as_default():
        bert_module = hub.Module(bert_model_hub)
        tokenization_info = bert_module(signature='tokenization_info', as_dict=True)
        with tf.Session() as sess:
            vocab_file, do_lower_case = sess.run([tokenization_info['vocab_file'],
                                                  tokenization_info['do_lower_case']])
    
    return bert.tokenization.FullTokenizer(
        vocab_file=vocab_file, do_lower_case=do_lower_case)


def getPhr2vec(model, phr_list, embed_dims):
    """
    Gives embedding for each phrase in phr_list

    Parameters
    ----------
    model:		Word2vec model
    phr_list:	List of phrases
    embed_dims:	Dimension of the embedding

    Returns
    -------
    embed_matrix:	(len(phr_list) x embed_dims) matrix containing embedding for each phrase in the phr_list in the same order
    """
    embed_list = []

    for phr in phr_list:
        if phr in model.vocab:
            embed_list.append(model.word_vec(phr))
        else:
            vec = np.zeros(embed_dims, np.float32)
            wrds = word_tokenize(phr)
            for wrd in wrds:
                if wrd in model.vocab:
                    vec += model.word_vec(wrd)
                else:
                    vec += np.random.randn(embed_dims)
            embed_list.append(vec / len(wrds))

    return np.array(embed_list)


def buildPhr2ELMOGraph(model):
    """
    Builds placeholders and graph for ELMO embeddings computation

    Parameters
    ----------
    model:		    ELMO model

    Returns
    -------
    embed_matrix:	(len(phr_list) x embed_dims) matrix containing embedding for each phrase in the phr_list in the same order
    """

    elmo_inputs = tf.placeholder(tf.string, shape=(None))

    elmo_outputs = model(
        elmo_inputs,
        signature="default",
        as_dict=True)['default']

    return elmo_inputs, elmo_outputs


def getPhr2ELMO(elmo_inputs, elmo_outputs, phr_list, sess=None):
    """
    Gives embedding for each phrase in phr_list

    Parameters
    ----------
    elmo_inputs:	ELMO inputs placeholders
    elmo_outputs:	ELMO outputs op
    phr_list:	    List of phrases for which embeddings are required
    sess:           tf.Session object

    Returns
    -------
    embed_matrix:	(len(phr_list) x embed_dims) matrix containing embedding for each phrase in the phr_list in the same order
    """

    if sess is None:
        sess_is_none = True
        sess = tf.Session()
        sess.run(tf.global_variables_initializer())
    else:
        sess_is_none = False

    feed_dict = {
        elmo_inputs: phr_list,
    }

    result = sess.run(elmo_outputs,
                      feed_dict=feed_dict)
    if sess_is_none:
        sess.close()

    return result


def buildPhr2BERTGraph(model, max_seq_length):
    """
    Builds placeholders and graph for BERT embeddings computation

    Parameters
    ----------
    model:		    BERT model
    max_seq_length:	Maximum length of a sequence

    Returns
    -------
    embed_matrix:	(len(phr_list) x embed_dims) matrix containing embedding for each phrase in the phr_list in the same order
    """

    input_ids = tf.placeholder(tf.int32, shape=(None, max_seq_length))
    input_mask = tf.placeholder(tf.int32, shape=(None, max_seq_length))
    segment_ids = tf.placeholder(tf.int32, shape=(None, max_seq_length))

    bert_inputs = {
        'input_ids': input_ids,
        'input_mask': input_mask,
        'segment_ids': segment_ids
    }

    bert_outputs = model(
        inputs=bert_inputs,
        as_dict=True,
        signature='tokens')

    return bert_inputs, bert_outputs


def getPhr2BERT(bert_inputs, bert_outputs, phr_list, sess=None):
    """
    Gives embedding for each phrase in phr_list

    Parameters
    ----------
    bert_inputs:	BERT inputs placeholders
    bert_outputs:	BERT outputs op
    phr_list:	    List of phrases for which embeddings are required
    sess:           tf.Session object

    Returns
    -------
    embed_matrix:	(len(phr_list) x embed_dims) matrix containing embedding for each phrase in the phr_list in the same order
    """

    if sess is None:
        sess_is_none = True
        sess = tf.Session()
        sess.run(tf.global_variables_initializer())
    else:
        sess_is_none = False

    feed_dict = {
        bert_inputs['input_ids']: phr_list['input_ids'],
        bert_inputs['input_mask']: phr_list['input_mask'],
        bert_inputs['segment_ids']: phr_list['segment_ids'],
    }

    result = sess.run(bert_outputs['sequence_output'],
                      feed_dict=feed_dict)
    if sess_is_none:
        sess.close()

    return np.array(result)[:, 0, :]


def set_gpu(gpus):
    """
    Sets the GPU to be used for the run

    Parameters
    ----------
    gpus:           List of GPUs to be used for the run

    Returns
    -------
    """
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = gpus


def checkFile(filename):
    """
    Check whether file is present or not

    Parameters
    ----------
    filename:       Path of the file to check

    Returns
    -------
    """
    return pathlib.Path(filename).is_file()


def make_dir(dir_path):
    """
    Creates the directory if doesn't exist

    Parameters
    ----------
    dir_path:       Path of the directory

    Returns
    -------
    """
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)


def debug_nn(res_list, feed_dict):
    """
    Function for debugging Tensorflow model

    Parameters
    ----------
    res_list:       List of tensors/variables to view
    feed_dict:	Feed dict required for getting values

    Returns
    -------
    Returns the list of values of given tensors/variables after execution

    """
    import tensorflow as tf

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    sess.run(tf.global_variables_initializer())
    summ_writer = tf.summary.FileWriter("tf_board/debug_nn", sess.graph)
    res = sess.run(res_list, feed_dict=feed_dict)
    return res


def get_logger(name, log_dir, config_dir):
    """
    Creates a logger object

    Parameters
    ----------
    name:           Name of the logger file
    log_dir:        Directory where logger file needs to be stored
    config_dir:     Directory from where log_config.json needs to be read

    Returns
    -------
    A logger object which writes to both file and stdout

    """
    make_dir(log_dir)
    config_dict = json.load(open(config_dir + 'log_config.json'))
    config_dict['handlers']['file_handler']['filename'] = log_dir + name.replace('/', '-')
    logging.config.dictConfig(config_dict)
    logger = logging.getLogger(name)

    std_out_format = '%(asctime)s - [%(levelname)s] - %(message)s'
    consoleHandler = logging.StreamHandler(sys.stdout)
    consoleHandler.setFormatter(logging.Formatter(std_out_format))
    logger.addHandler(consoleHandler)

    return logger


def getChunks(inp_list, chunk_size):
    """
    Splits inp_list into lists of size chunk_size

    Parameters
    ----------
    inp_list:       List to be splittted
    chunk_size:     Size of each chunk required

    Returns
    -------
    chunks of the inp_list each of size chunk_size, last one can be smaller (leftout data)
    """
    return [inp_list[x:x + chunk_size] for x in range(0, len(inp_list), chunk_size)]


def partition(inp_list, n):
    """
    Paritions a given list into chunks of size n

    Parameters
    ----------
    inp_list:       List to be splittted
    n:     		    Number of equal partitions needed

    Returns
    -------
    Splits inp_list into n equal chunks
    """
    division = len(inp_list) / float(n)
    return [inp_list[int(round(division * i)): int(round(division * (i + 1)))] for i in range(n)]


def mergeList(list_of_list):
    """
    Merges list of list into a list

    Parameters
    ----------
    list_of_list:   List of list

    Returns
    -------
    A single list (union of all given lists)
    """
    return list(itertools.chain.from_iterable(list_of_list))
