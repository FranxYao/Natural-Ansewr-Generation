




import numpy as np
from main_simple import *
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
from model_checklist import *
from utils import *

start_time = time.time()
# dset = Dataset()
dset = cPickle.load(open("../data/dataset.pkl", "rb"))
dset.build_remain(config)
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





# Supervised training
batch_num = 500
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

metrics_ret = metrics_full(aoub, qcb, out_idx, mkb, mvb, mlenb, dset, bi)

loss = all_out[0]
train_op = all_out[1]
out_idx = all_out[2]
all_logits = all_out[3]
dbg_attn_mkey = all_out[4]
dbg_attn_mval = all_out[5]
dbg_attn_q = all_out[6]
dbg_attn_out = all_out[7]


attn_visualize(qb, qlenb, mkb, mvb, mlenb, out_idx, dbg_attn_mkey, dbg_attn_mval, dbg_attn_q, dbg_attn_out, dset)
