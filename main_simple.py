# coding: utf-8

# ---- Main for FQA ----
# -- Francis Fu Yao 
# -- francis_yao@pku.edu.cn
# -- MON 29TH AUG 2017

import tensorflow as tf 
from data_utils_full import Dataset
# from model import Model
from model_checklist import Model
import os
import cPickle
import time
GPU_NUM = 1
os.environ["CUDA_VISIBLE_DEVICES"] = str(GPU_NUM)
flags = tf.flags
flags.DEFINE_integer("vocab_size", 0, "")
flags.DEFINE_integer("vocab_norm_size", 0, "")
flags.DEFINE_integer("vocab_spec_size", 0, "")
flags.DEFINE_integer("max_q_len", 0, "")
flags.DEFINE_integer("max_a_len", 0, "")
flags.DEFINE_integer("max_s_len", 0, "")
flags.DEFINE_integer("state_size", 256, "")
flags.DEFINE_integer("epoch_num", 40, "")
flags.DEFINE_integer("batch_size", 128, "")
flags.DEFINE_boolean("is_train", True, "")
flags.DEFINE_boolean("is_kv", True, "")
flags.DEFINE_integer("qclass_size", 10, "")
flags.DEFINE_integer("aclass_size", 3, "")
flags.DEFINE_integer("memkey_size", 21, "")
flags.DEFINE_integer("attn_hop_size", 2, "")
flags.DEFINE_integer("out_mode_gate", False, "")
flags.DEFINE_integer("gpu", GPU_NUM, "")
flags.DEFINE_float("gpufrac", 1.00, "")
flags.DEFINE_integer("ncon_size", 10000, "")
flags.DEFINE_integer("conj_size", -1,"")
flags.DEFINE_integer("wiki_size", -1,"")
flags.DEFINE_string("config_name", "fqa", "")
flags.DEFINE_string("data_source", "ncon_conj", "")
flags.DEFINE_string("out_dir", "../output", "")
flags.DEFINE_string("data_dir", "../data/dataset.pkl", "")
config = flags.FLAGS

def main():
  # read dataset
  start_time = time.time()
  # dset = Dataset()
  dset = cPickle.load(open(config.data_dir, "rb"))
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
    mtest = Model(config, "test")
    mtest.build()
  print("\ntime to build: %.2f\n\n" % (time.time() - start_time))
  # train
  m.train(dset, mvalid, mtest)
  return 

if __name__ == "__main__":
  main()

