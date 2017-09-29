#

import numpy as np
from main_simple import *
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from model import *

start_time = time.time()
# dset = Dataset()
dset = cPickle.load(open("../data/dset.pkl", "rb"))
dset.build_remain()
print("\n%.2f seconds to read dset" % (time.time() - start_time))

# build model
config.vocab_size = dset.total_words
config.vocab_norm_size = dset.norm_word_cnt
config.vocab_spec_size = dset.spec_word_cnt
config.max_q_len = dset.max_q_len
config.max_a_len = dset.max_a_len
config.max_m_len = dset.max_mem_len
with tf.variable_scope("model"):
  m = Model(config, "train")
  m.build()
config.is_train = False
with tf.variable_scope("model", reuse = True):
  mvalid = Model(config, "valid")
  print("building valid model")
  mvalid.build()
  # print("building test model")
  # mtest = Model(config)
  # mtest.build()
print("\ntime to build: %.2f\n\n" % (time.time() - start_time))

print("\nstart training ... ")
sess = tf.Session()
sess.run(tf.global_variables_initializer())
total_batches = dset.total_cases_train
print("%d batches in total" % total_batches)


qb, qlenb, qcb, ainb, aoub, alenb, acb, mb, mkb, mvb, mlenb = dset.get_next_batch("train", m.batch_size)
dec_init_record = np.zeros([m.batch_size, m.max_a_len, m.state_size])
# print("dec_init_record shape:", dec_init_record.shape)
feed_dict = dict()
# question
feed_dict[m.input_q] = qb
feed_dict[m.input_qlen] = qlenb
feed_dict[m.input_qc] = qcb
# answer
feed_dict[m.input_ain] = ainb
feed_dict[m.input_aou] = aoub
feed_dict[m.input_alen] = alenb
feed_dict[m.input_ac] = acb
# memory
feed_dict[m.input_mk] = mkb
feed_dict[m.input_mv] = mvb
feed_dict[m.input_mlen] = mlenb
# movie
feed_dict[m.input_movi] = mb
# initial mder record
feed_dict[m.dec_out_record] = dec_init_record
# dropout
feed_dict[m.keep_prob] = 0.8
all_out = sess.run(m.out, feed_dict)

loss = all_out[0]
train_op = all_out[1]
out_idx = all_out[2]
all_logits = all_out[3]
dbg_dec_o = all_out[4]
dbg_attn_mk_o = all_out[5]
dbg_attn_mk_e = all_out[6]
dbg_attn_mk_e_nomask = all_out[7]
dbg_attn_mv_o = all_out[8]
dbg_attn_mv_e = all_out[9]
dbg_attn_mv_e_nomask = all_out[10]
dbg_attn_out_o = all_out[11]
dbg_attn_out_e = all_out[12]
dbg_attn_out_mo = all_out[13]
dbg_attn_out_e_nomask = all_out[14]
dbg_dec_out_record = all_out[15]
dbg_attn_o_norm_e = all_out[16]
dbg_attn_o_spec_e = all_out[17]

# Supervised training
batch_num = 1000
start_time = time.time()
set_loss = 0.0
for bi in range(batch_num):
  qb, qlenb, qcb, ainb, aoub, alenb, acb, mb, mkb, mvb, mlenb = dset.get_next_batch("train", m.batch_size)
  dec_init_record = np.zeros([m.batch_size, m.max_a_len, m.state_size])
  # print("dec_init_record shape:", dec_init_record.shape)
  feed_dict = dict()
  # question
  feed_dict[m.input_q] = qb
  feed_dict[m.input_qlen] = qlenb
  feed_dict[m.input_qc] = qcb
  # answer
  feed_dict[m.input_ain] = ainb
  feed_dict[m.input_aou] = aoub
  feed_dict[m.input_alen] = alenb
  feed_dict[m.input_ac] = acb
  # memory
  feed_dict[m.input_mk] = mkb
  feed_dict[m.input_mv] = mvb
  feed_dict[m.input_mlen] = mlenb
  # movie
  feed_dict[m.input_movi] = mb
  # initial mder record
  feed_dict[m.dec_out_record] = dec_init_record
  # dropout
  feed_dict[m.keep_prob] = 0.8
  all_out = sess.run(m.out, feed_dict)
  loss = all_out[0]
  train_op = all_out[1]
  out_idx = all_out[2]
  set_loss += loss
  if(bi % 20 == 0 and bi > 0):
    if(bi % 100 == 0): print_batch(dset, qb, qlenb, qcb, ainb, aoub, alenb, acb, mb, mkb, mvb, mlenb, out_idx)
    print("\n------")
    print("batch %d, current loss %.4f, set loss: %.4f" % (bi, loss, set_loss / 20.0))
    set_loss = 0.0
    print("time cost: %.2f" % (time.time() - start_time))
    start_time = time.time()


