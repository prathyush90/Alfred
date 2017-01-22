EN_WHITELIST = '0123456789abcdefghijklmnopqrstuvwxyz '

limit = {
        'maxq' : 25,
        'minq' : 2,
        'maxa' : 25,
        'mina' : 2
        }

UNK = 'unk'
VOCAB_SIZE = 8000

import nltk
import itertools
import numpy as np
import pickle

def parseMovieLines():
    movielines = {}
    with open("./dataFiles/movie_lines.txt",encoding = "ISO-8859-1") as f:
        for line in f:
            linearray = line.split("+++$+++")
            key       = linearray[0].replace(" ","")
            movielines[key] = linearray[4]

    return  movielines;

def getMovieConversations(movielines):
    encoders = [];
    decoders = [];
    with open("./dataFiles/movie_conversations.txt",encoding = "ISO-8859-1") as f:
        for line in f:
            convarray     = line.split("+++$+++")
            conversstring = convarray[3]
            conversstring1 = conversstring.replace("[","")
            conversstring2 = conversstring1.replace("]","")
            conversations  = conversstring2.split(",");
            index=0;
            for i in range(len(conversations)):
                index = i;
                if(index % 2 != 0 ):
                    break;
                key   =  conversations[index].replace("'","")
                key = key.replace("\n","")
                key   =  key.replace(" ","")
                key   = key.strip()
                value =  conversations[index+1].replace("'","")
                key = key.replace("\n", "")
                value = value.replace(" ","")
                value = value.strip()
                encoders.append(movielines[key]);
                decoders.append(movielines[value]);
            # movieconversations[movielines[key]] = movielines[value]
        return  encoders,decoders;

# with open("movie_dialogues_corpus.txt",'w') as f:
#     for x in movieconversations:
#         f.write(x+"+++$+++"+movieconversations[x]+"\n")

def appendTwitterConversations(encoders,decoders):
    index = 0;
    key = "";
    value = "";
    with open("./dataFiles/twitter_en.txt",encoding = "ISO-8859-1") as f:
        for line in f:
            index=index+1;
            if (index % 2 != 0):
                value = line.replace("\n", "");
                value = value.replace(" ", "")
                value = value.strip()
                decoders.append(value);
                encoders.append(key)
                key   = "";
                value = "";
            else:
                key = line.replace("\n", "")
                key = key.replace(" ", "")
                key = key.strip()

    return  encoders,decoders








def filter_line(line, whitelist):
    return ''.join([ ch for ch in line if ch in whitelist ])

def filter_data(qseq, aseq):
    filtered_q, filtered_a = [], []
    raw_data_len = len(qseq)

    assert len(qseq) == len(aseq)

    for i in range(raw_data_len):
        qlen, alen = len(qseq[i].split(' ')), len(aseq[i].split(' '))
        if qlen >= limit['minq'] and qlen <= limit['maxq']:
            if alen >= limit['mina'] and alen <= limit['maxa']:
                filtered_q.append(qseq[i])
                filtered_a.append(aseq[i])

    # print the fraction of the original data, filtered
    filt_data_len = len(filtered_q)
    filtered = int((raw_data_len - filt_data_len)*100/raw_data_len)
    print(str(filtered) + '% filtered from original data')

    return filtered_q, filtered_a


def index_(tokenized_sentences, vocab_size):
    # get frequency distribution
    freq_dist = nltk.FreqDist(itertools.chain(*tokenized_sentences))
    # get vocabulary of 'vocab_size' most used words
    vocab = freq_dist.most_common(vocab_size)
    # index2word
    index2word = ['_'] + [UNK] + [ x[0] for x in vocab ]
    # word2index
    word2index = dict([(w,i) for i,w in enumerate(index2word)] )
    return index2word, word2index, freq_dist


def filter_unk(qtokenized, atokenized, w2idx):
    data_len = len(qtokenized)

    filtered_q, filtered_a = [], []

    for qline, aline in zip(qtokenized, atokenized):
        unk_count_q = len([ w for w in qline if w not in w2idx ])
        unk_count_a = len([ w for w in aline if w not in w2idx ])
        if unk_count_a <= 2:
            if unk_count_q > 0:
                if unk_count_q/len(qline) > 0.2:
                    pass
            filtered_q.append(qline)
            filtered_a.append(aline)

    # print the fraction of the original data, filtered
    filt_data_len = len(filtered_q)
    filtered = int((data_len - filt_data_len)*100/data_len)
    print(str(filtered) + '% filtered from original data')

    return filtered_q, filtered_a


def zero_pad(qtokenized, atokenized, w2idx):
    # num of rows
    data_len = len(qtokenized)

    # numpy arrays to store indices
    idx_q = np.zeros([data_len, limit['maxq']], dtype=np.int32)
    idx_a = np.zeros([data_len, limit['maxa']], dtype=np.int32)

    for i in range(data_len):
        q_indices = pad_seq(qtokenized[i], w2idx, limit['maxq'])
        a_indices = pad_seq(atokenized[i], w2idx, limit['maxa'])

        #print(len(idx_q[i]), len(q_indices))
        #print(len(idx_a[i]), len(a_indices))
        idx_q[i] = np.array(q_indices)
        idx_a[i] = np.array(a_indices)

    return idx_q, idx_a


'''
 replace words with indices in a sequence
  replace with unknown if word not in lookup
    return [list of indices]
'''
def pad_seq(seq, lookup, maxlen):
    indices = []
    for word in seq:
        if word in lookup:
            indices.append(lookup[word])
        else:
            indices.append(lookup[UNK])
    return indices + [0]*(maxlen - len(seq))

def process_data():
    #parse movie_lines movie lines corpus.Find link in read me
    movielines          = parseMovieLines();
    encoders,decoders   = getMovieConversations(movielines)
    # Parse twitter chats.Find link in readme
    encoders,decoders   = appendTwitterConversations(encoders,decoders)
    #Change everything to Lowercase to make bot case insensitive
    encoders            = [line.lower() for line in encoders]
    decoders            = [line.lower() for line in decoders]

    #Use only words and numbers that are allowed removing unwanted special characters
    encoders            = [filter_line(line, EN_WHITELIST) for line in encoders]
    decoders            = [filter_line(line, EN_WHITELIST) for line in decoders]

    #remove too long and too short sentences
    encoders, decoders  = filter_data(encoders, decoders)

    #split lines to [[words],..]
    enctokenized = [[w.strip() for w in wordlist.split(' ') if w] for wordlist in encoders]
    dectokenized = [[w.strip() for w in wordlist.split(' ') if w] for wordlist in decoders]

    #word to index and take frequent words
    idx2w, w2idx, freq_dist = index_(enctokenized + dectokenized, vocab_size=VOCAB_SIZE)

    #remove sentences with too many unknown words
    enctokenized, dectokenized = filter_unk(enctokenized, dectokenized, w2idx)

    #convert words into numpy arrays by indices with zero pad
    idx_enc, idx_dec = zero_pad(enctokenized, dectokenized, w2idx)

    np.save('./data/idx_q.npy', idx_enc)
    np.save('./data/idx_a.npy', idx_dec)

    # let us now save the necessary dictionaries
    metadata = {
        'w2idx': w2idx,
        'idx2w': idx2w,
        'limit': limit,
        'freq_dist': freq_dist
    }

    # write to disk : data control dictionaries
    with open('./data/metadata.pkl', 'wb') as f:
        pickle.dump(metadata, f)

    # count of unknowns
    unk_count = (idx_enc == 1).sum() + (idx_dec == 1).sum()
    # count of words
    word_count = (idx_enc > 1).sum() + (idx_dec > 1).sum()

    print('% unknown : {0}'.format(100 * (unk_count / word_count)))
    print('Dataset count : ' + str(idx_enc.shape[0]))


if __name__ == '__main__':
    process_data()


def load_data(PATH=''):
    # read data control dictionaries
    with open(PATH + 'metadata.pkl', 'rb') as f:
        metadata = pickle.load(f)
    # read numpy arrays
    idx_enc = np.load(PATH + 'idx_q.npy')
    idx_dec = np.load(PATH + 'idx_a.npy')
    return metadata, idx_enc, idx_dec



#(train_enc,train_dec),(test_enc,test_dec) = split_dataset(encoders,decoders)


# with open("conversations_corpus_train_enc.txt",'w') as f:
#     for x in train_enc:
#         f.write(x+"\n");
# with open("conversations_corpus_train_dec.txt",'a') as f:
#     for x in train_dec:
#         f.write(x+"\n");
#
# with open("conversations_corpus_test_enc.txt",'w') as f:
#     for x in test_enc:
#         f.write(x+"\n");
# with open("conversations_corpus_test_dec.txt",'a') as f:
#     for x in test_dec:
#         f.write(x+"\n");

