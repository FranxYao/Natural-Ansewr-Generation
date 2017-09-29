# coding: utf-8

# ---- Main for FQA ----
# -- Francis Fu Yao 
# -- francis_yao@pku.edu.cn
# -- MON 29TH AUG 2017

import tensorflow as tf 
from data_utils_full import Dataset
from model import Model
import os
import cPickle
import time

GPU_NUM = 0
os.environ["CUDA_VISIBLE_DEVICES"] = str(GPU_NUM)
flags = tf.flags
flags.DEFINE_integer("vocab_size", 0, "")
flags.DEFINE_integer("vocab_norm_size", 0, "")
flags.DEFINE_integer("vocab_spec_size", 0, "")
flags.DEFINE_integer("max_q_len", 0, "")
flags.DEFINE_integer("max_a_len", 0, "")
flags.DEFINE_integer("max_s_len", 0, "")
flags.DEFINE_integer("state_size", 256, "")
flags.DEFINE_integer("epoch_num", 10, "")
flags.DEFINE_integer("batch_size", 128, "")
flags.DEFINE_boolean("is_train", True, "")
flags.DEFINE_integer("qclass_size", 10, "")
flags.DEFINE_integer("aclass_size", 3, "")
flags.DEFINE_integer("memkey_size", 21, "")
flags.DEFINE_integer("attn_hop_size", 2, "")
flags.DEFINE_integer("out_mode_gate", False, "")
flags.DEFINE_integer("gpu", GPU_NUM, "")
config = flags.FLAGS

def main():
  # read dataset
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
    mtest = Model(config, "test")
    mtest.build()
  print("\ntime to build: %.2f\n\n" % (time.time() - start_time))
  # train
  m.train(dset, mvalid, mtest)
  return 

if __name__ == "__main__":
  main()

