# coding: utf-8

# ---- FQA Model, simple version ----
# -- Francis Fu Yao 
# -- francis_yao@pku.edu.cn
# -- SUN 14TH AUG 2017

import tensorflow as tf
import numpy as np
import time
from sets import Set

class Model(object):
  def __init__(self, config):
    self.vocab_size = config.vocab_size
    self.vocab_norm_size = config.vocab_norm_size
    self.vocab_spec_size = config.vocab_spec_size
    self.max_q_len = config.max_q_len
    self.max_a_len = config.max_a_len
    self.max_s_len = config.max_s_len
    self.state_size = config.state_size
    self.epoch_num = config.epoch_num
    self.batch_size = config.batch_size
    self.is_train = config.is_train
    return 

  def build(self):
    # -- Placeholders
    self.input_q = tf.placeholder(dtype = tf.int32, shape = [None, self.max_q_len]) 
    self.input_qlen = tf.placeholder(dtype = tf.int32, shape = [None])
    self.input_a = tf.placeholder(dtype = tf.int32, shape = [None, self.max_a_len])
    self.input_alen = tf.placeholder(dtype = tf.int32, shape = [None])
    self.input_movie = tf.placeholder(dtype = tf.int32, shape = [None])
    self.input_s = tf.placeholder(dtype = tf.int32, shape = [self.max_s_len - 1, None])
    self.input_slen = tf.placeholder(dtype = tf.int32, shape = [None])
    self.targets = tf.placeholder(dtype = tf.int32, shape = [self.max_s_len - 1, None])
    self.keep_prob = tf.placeholder(dtype = tf.float32, shape = ())
    # self.is_train = tf.placeholder(dtype = tf.float32, shape = ())

    # -- Build embeddings:
    max_q_len = self.max_q_len
    max_a_len = self.max_a_len
    max_s_len = self.max_s_len
    norm_size = self.vocab_norm_size
    spec_size = self.vocab_spec_size
    state_size = self.state_size
    batch_size = tf.shape(self.input_q)[0]
    with tf.device("/cpu:0"):
      embeddings = tf.get_variable(name = "embedding", 
                                   shape = [self.vocab_size, state_size],
                                   dtype = tf.float32, 
                                   initializer = tf.random_uniform_initializer(0.0, 1.0))
      norm_emb, spec_emb = tf.split(embeddings, [norm_size + 1, spec_size], 0)
      print("norm embedding size: ", norm_emb.shape)  # [norm_size + 1, state_size]
      print("spec embedding size: ", spec_emb.shape)  # [spec_size, state_size]
    embed_q = tf.nn.embedding_lookup(embeddings, self.input_q)
    embed_a = tf.nn.embedding_lookup(embeddings, self.input_a)
    embed_m = tf.nn.embedding_lookup(embeddings, self.input_movie)
    embed_s = tf.nn.embedding_lookup(embeddings, self.input_s)
    print("embed_q shape: ", embed_q.shape) # [batch_size, max_q_len, state_size]
    print("embed_a shape: ", embed_a.shape) # [batch_size, max_a_len, state_size]
    print("embed_m shape: ", embed_m.shape) # [batch_size, state_size]
    print("embed_s shape: ", embed_s.shape) # [max_s_len - 1, batch_size, state_size]
    # print("embed_t shape: ", embed_t.shape) # [max_s_len - 1, batch_size, state_size]

    # -- Question encoding RNN, Q-A encoding
    encoder_cell = tf.contrib.rnn.LSTMCell(num_units = state_size)
    encoder_cell = tf.contrib.rnn.DropoutWrapper(encoder_cell, output_keep_prob = self.keep_prob)
    enc_init_state = encoder_cell.zero_state(batch_size = batch_size, dtype = tf.float32)
    (q_rnn_out, q_rnn_stt) = tf.nn.dynamic_rnn(cell = encoder_cell, 
                                               inputs = embed_q,
                                               sequence_length = self.input_qlen, 
                                               initial_state = enc_init_state)
    print("q_rnn_out size:", q_rnn_out.shape)  # [batch_size, max_q_len, state_size]
    # get last output 
    hot = tf.expand_dims(tf.cast(tf.one_hot(self.input_qlen - 1, max_q_len), tf.float32), 2) 
    print("hot shape: ", hot.shape) # [batch_size, max_q_len, 1]
    encoded_q = tf.reduce_sum(hot * q_rnn_out, 1)
    print("encoded_q shape: ", encoded_q.shape)  # [batch_size, state_size]
    # TB Test here ... -- GOOD

    # answer set encoding
    # mask padding: if pad, set corresponding vector to be 0
    a_mask = tf.expand_dims(tf.sequence_mask(self.input_alen, max_a_len, dtype = tf.float32), 2)
    print("mask shape: ", a_mask.shape) # [batch_size, max_a_len, 1]
    encoded_ans = tf.reduce_sum(a_mask * embed_a, 1)
    print("encoded_ans shape: ", encoded_ans.shape) # [batch_size, state_size]
    # TB test here ...  -- GOOD

    # all encoded:
    encoded_all = encoded_ans + encoded_q + embed_m
    print("encoded size: ", encoded_all.shape)

    # -- Attention
    # TB test here ... 
    # Note: here we use bilinear function as attention 
    # Note: if want multihop, just call this function multiple times 
    atten_mem_a = embed_a
    atten_mem_q = q_rnn_out
    def attention(query, stepi):
      alen = self.input_alen
      qlen = self.input_qlen
      with tf.variable_scope("attention") as atten_scope:
        if(stepi != 0): atten_scope.reuse_variables()
        # -- Attention on given answer (variable length)
        atten_w_a = tf.get_variable(name = "atten_w_a", 
                                  shape = [state_size, state_size],
                                  dtype = tf.float32,
                                  initializer = tf.random_normal_initializer())
        # attention energy at every timestep
        tmp_mem_a = tf.reshape(atten_mem_a, [-1, state_size])
        atten_mid_a = tf.reshape(tf.matmul(tmp_mem_a, atten_w_a), [batch_size, max_a_len, state_size])
        # query size: [batch_size, 1 -> max_a_len, state_size]
        atten_a_e = tf.reduce_sum(atten_mid_a * tf.expand_dims(query, 1), 2)
        # mask padding on attention energy
        atten_a_e = atten_a_e * tf.sequence_mask(alen, max_a_len, dtype = tf.float32)
        # softmax and sum  
        atten_a_e = tf.nn.softmax(atten_a_e)
        atten_a_out = tf.reduce_sum(tf.expand_dims(atten_a_e, 2) * atten_mem_a, 1)

        # -- Attention on given question (variable length)
        atten_w_q = tf.get_variable(name = "atten_w_q", 
                                  shape = [state_size, state_size],
                                  dtype = tf.float32,
                                  initializer = tf.random_normal_initializer())
        tmp_mem_q = tf.reshape(atten_mem_q, [-1, state_size])
        atten_mid_q = tf.reshape(tf.matmul(tmp_mem_q, atten_w_q), [batch_size, max_q_len, state_size])
        # query size: [batch_size, 1 -> max_a_len, state_size]  
        atten_q_e = tf.reduce_sum(atten_mid_q * tf.expand_dims(query, 1), 2)
        # mask padding
        atten_q_e = atten_q_e * tf.sequence_mask(qlen, max_q_len, dtype = tf.float32)
        # softmax and sum 
        atten_q_e = tf.nn.softmax(atten_q_e)
        atten_q_out = tf.reduce_sum(tf.expand_dims(atten_q_e, -1) * atten_mem_q, 1)

        # -- Output
        out = atten_a_out + atten_q_out + query
        return out

    # -- Prepare decoder
    decoder_cell = tf.contrib.rnn.LSTMCell(num_units = state_size)
    decoder_cell = tf.contrib.rnn.DropoutWrapper(decoder_cell, output_keep_prob = self.keep_prob)
    prev_h = (encoded_all, encoded_all)
    dec_s = tf.unstack(embed_s)
    out_idx = []
    loss_steps = []
    pred_size = norm_size + 1 + max_a_len + 1
    label_steps = tf.one_hot(self.targets, pred_size) # size: [time_step, batch_size, pred_size]
    label_steps = tf.unstack(label_steps)

    # -- Start decoding 
    with tf.variable_scope("decoder") as decoder_scope:
      for i in range(self.max_s_len - 1):
        if(i >= 1): decoder_scope.reuse_variables()
        if(i == 0): current_i = dec_s[0]
        else:
          # if self.is_train == 1, then current_i = dec_s[i], else current_i = prev_o
          # current_i = self.is_train * dec_s[i] + (1.0 - self.is_train) * prev_o
          if(self.is_train):
            current_i = dec_s[i]
          else:
            current_i = prev_o
        current_o, current_h = decoder_cell(current_i, prev_h)
        prev_h = current_h
        # attention on question and answer list words (i.e. memory)
        atten_o = attention(current_o, i)

        # -- Output projection to normal words
        output_norm_w = tf.get_variable(name = "output_norm_w", 
                                        shape = [state_size, norm_size + 1], 
                                        dtype = tf.float32,
                                        initializer = tf.random_normal_initializer())
        output_norm_b = tf.get_variable(name = "output_norm_b", 
                                        shape = [norm_size + 1], 
                                        dtype = tf.float32,
                                        initializer = tf.constant_initializer(0.0))
        atten_o_norm_e = tf.matmul(atten_o, output_norm_w) + output_norm_b

        # -- Energy funciton with special words (need mask)
        output_spec_w = tf.get_variable(name = "output_spec_w", 
                                        shape = [state_size, state_size], 
                                        dtype = tf.float32,
                                        initializer = tf.random_normal_initializer())
        atten_o_spec_w = tf.matmul(atten_o, output_spec_w)  # size: [batch_size, state_size]
        # concat movie name and answer as special 
        embed_spec = tf.concat([tf.expand_dims(embed_m, 1), embed_a], 1)  # size: [batch_size, max_a_len + 1, state_size]
        # atten_o_spec_w size: [batch_size, 1 -> max_a_len + 1, state_size]
        atten_o_spec_e = tf.reduce_sum(tf.expand_dims(atten_o_spec_w, 1) * embed_spec, 2)
        mask_spec = tf.sequence_mask(self.input_alen + 1, max_a_len + 1, dtype = tf.float32)
        atten_o_spec_e = atten_o_spec_e * mask_spec

        # -- Softmax & loss
        softmax_b = tf.get_variable(name = "softmax_b", 
                                    shape = [pred_size], # [norm_size + 1] + [movie] + ans
                                    dtype = tf.float32,
                                    initializer = tf.constant_initializer(0.0))
        logits = tf.concat([atten_o_norm_e, atten_o_spec_e], 1) # size: [batch_size, pred_size]
        logits = logits + softmax_b
        if(self.is_train):
          labels = label_steps[i]
          loss_steps.append(tf.nn.softmax_cross_entropy_with_logits(labels = labels, logits = logits))

        # -- Prepare next input for test 
        out_index = tf.argmax(logits, 1)
        out_idx.append(out_index)

        if(self.is_train == False):
          # out_choose_norm size: [batch_size, norm_size + 1], spec size: [batch_size, max_a_len + 1]
          out_choose_norm, out_choose_spec = tf.split(tf.one_hot(out_index, pred_size), [norm_size + 1, max_a_len + 1], 1)
          # choose normal words using tensorflow element-wise product broadcasting
          # after expand dims, out_choose_norm size: [     batch_size, norm_size + 1, 1 -> state_size]
          #                           norm_emb size: [1 -> batch_size, norm_size + 1,      state_size]
          o_norm = tf.reduce_sum(tf.expand_dims(out_choose_norm, 2) * tf.expand_dims(norm_emb, 0), 1)
          # out_choose_spec size: [batch_size, max_a_len + 1, 1 -> state_size]
          o_spec = tf.reduce_sum(tf.expand_dims(out_choose_spec, 2) * embed_spec, 1)
          prev_o = o_norm + o_spec
          # prev_o = tf.stop_gradient(prev_o)
          # prev_o = None    

    # -- Output
    out_idx = tf.transpose(tf.stack(out_idx)) # size: [batch_size, timesteps]
    # self.trainable = tf.trainable_variables()
    if(self.is_train == False):
      self.out = out_idx
      return

    # -- Loss and optimizer
    loss = tf.stack(loss_steps)
    loss = tf.transpose(loss)  # size: [batch_size, max_s_len - 1]
    # mask loss 
    loss_mask = tf.sequence_mask(self.input_slen, max_s_len - 1, tf.float32) # size: [batch_size, max_s_len]
    loss = tf.reduce_mean(loss_mask * loss)
    optimizer = tf.train.AdamOptimizer()
    # TB update: gradient clipping
    # grad_and_vars = optimizer.compute_gradients(loss)
    train_op = optimizer.minimize(loss)
    self.out = [loss ,train_op, out_idx]
    self.out_test = [loss, train_op, q_rnn_out, encoded_q, embed_a, encoded_ans]
    print("finished")
    return

  def test_model(self, dset):
    (qb, qlenb, ab, alenb, nb, sb, slenb, tb) = dset.get_next_batch("train", self.batch_size)
    feed_dict = dict()
    feed_dict[self.input_q] = qb
    feed_dict[self.input_qlen] = qlenb
    feed_dict[self.input_a] = ab
    feed_dict[self.input_alen] = alenb
    feed_dict[self.input_s] = sb
    feed_dict[self.input_slen] = slenb
    feed_dict[self.input_movie] = nb
    feed_dict[self.targets] = tb
    # feed_dict[self.is_train] = 1.0
    feed_dict[self.keep_prob] = 0.6
    loss, _, q_rnn_out, encoded_q, embed_a, encoded_ans = sess.run(self.out_test, feed_dict)
    return q_rnn_out, encoded_q, qlenb, embed_a, alenb, encoded_ans


  def train(self, dset, mvalid):
    print("start training ... ")
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    total_batches = dset.total_case["train"] / self.batch_size
    print("%d batches in total" % total_batches)
    for ei in range(self.epoch_num):
      fd = open("outx%d" % ei, "w")
      accuracy = 0.0
      cover = 0.0
      loss_accu = 0.0
      test_acc_accu = 0.0
      test_cov_accu = 0.0
      test_cnt = 0
      start_time = time.time()
      for bi in range(total_batches):
        (qb, qlenb, ab, alenb, nb, sb, slenb, tb) = dset.get_next_batch("train", self.batch_size)
        feed_dict = dict()
        feed_dict[self.input_q] = qb
        feed_dict[self.input_qlen] = qlenb
        feed_dict[self.input_a] = ab
        feed_dict[self.input_alen] = alenb
        feed_dict[self.input_s] = sb
        feed_dict[self.input_slen] = slenb
        feed_dict[self.input_movie] = nb
        feed_dict[self.targets] = tb
        # feed_dict[self.is_train] = 1.0
        feed_dict[self.keep_prob] = 1.0
        loss, _, out_idx = sess.run(self.out, feed_dict)
        loss_accu += loss
        accuracy_now, cover_now = batch_metrics(self, tb, slenb, out_idx)
        accuracy += accuracy_now
        cover += cover_now
        if(bi % 200 == 0 and bi > 0):
          print_ans(dset, qb, ab, nb, sb, tb, out_idx, self.batch_size)
          print("--------------------------------\n")
          print("batch %d, loss %.4f" % (bi, loss))
          print("accuracy: %.4f, cover: %.4f" % (accuracy_now, cover_now))
          print("accumulated accuracy: %.4f, cover: %.4f" % (accuracy / (bi + 1), cover / (bi + 1)))
          print("\n--------------------------------")
          test_acc, test_cov = test_valid(mvalid, sess, dset, ei, bi, fd)
          test_acc_accu += test_acc
          test_cov_accu += test_cov
          test_cnt += 1
          print("\n\ntime cost: %.4f" % (time.time() - start_time))
          start_time = time.time()
      print("\n*********************************")
      print("epoch %d finished" % ei)
      print("average loss: %.5f" % (loss_accu / total_batches))
      print("accumulated accuracy: %.4f, cover: %.4f" % (accuracy / (bi + 1), cover / (bi + 1)))
      print("test accumulated accuracy: %.4f, cover: %.4f" % (test_acc_accu / test_cnt, test_cov_accu / test_cnt))
      print("*********************************\n")
      fd.close()
    return 

def batch_metrics(model, tb, slenb, out_idx):
  tb = np.transpose(tb)
  positive = 0
  negative = 0
  spec_in = 0
  spec_total = 0
  for i in range(model.batch_size):
    spec_set_gold = []
    spec_set_pred = []
    for j in range(slenb[i] - 1):
      if(tb[i][j] == out_idx[i][j]):
        positive += 1
      else:
        negative += 1
      if(tb[i][j] > model.vocab_norm_size):
        spec_set_gold.append(tb[i][j])
      if(out_idx[i][j] > model.vocab_norm_size):
        spec_set_pred.append(out_idx[i][j])
    spec_set_gold = Set(spec_set_gold)
    spec_set_pred = Set(spec_set_pred)
    spec_in += len(spec_set_gold & spec_set_pred)
    spec_total += len(spec_set_gold)
  accuracy = float(positive) / (positive + negative)
  cover = float(spec_in) / (spec_total)
  return accuracy, cover

def test_valid(mvalid, sess, dset, epoch_num, batch_num, fd):
  (qb, qlenb, ab, alenb, nb, sb, slenb, tb) = dset.get_next_batch("valid", mvalid.batch_size)
  feed_dict = dict()
  feed_dict[mvalid.input_q] = qb
  feed_dict[mvalid.input_qlen] = qlenb
  feed_dict[mvalid.input_a] = ab
  feed_dict[mvalid.input_alen] = alenb
  feed_dict[mvalid.input_s] = sb
  feed_dict[mvalid.input_slen] = slenb
  feed_dict[mvalid.input_movie] = nb
  feed_dict[mvalid.targets] = tb
  # feed_dict[self.is_train] = 1.0
  feed_dict[mvalid.keep_prob] = 1.0
  out_idx = sess.run(mvalid.out, feed_dict)
  print("\n\n---- On validation set: ")
  qstr, astr, nstr, goldstr, predstr = print_ans(dset, qb, ab, nb, sb, tb, out_idx, mvalid.batch_size)
  accuracy, cover = batch_metrics(mvalid, tb, slenb, out_idx)
  print("---- accuracy: %.4f, cover: %.4f" % (accuracy, cover))
  if(epoch_num >= 0):
    fd.write("%dth batch\n" % batch_num)
    for qi, ai, ni, gi, pi in zip(qstr, astr, nstr, goldstr, predstr):
      fd.write("q: %s\na: %s\nn: %s\ng: %s\np: %s\n----\n" % (qi, ai, ni, gi, pi))
  return accuracy, cover

def print_ans(dset, qb, ab, nb, sb, tb, out_idx, batch_size):
  sb = np.transpose(sb)
  tb = np.transpose(tb)
  ob = out_idx
  qstr = []
  astr = []
  nstr = []
  goldstr = []
  predstr = []
  for i in range(batch_size):
    # print(qb[i])
    qwords = []
    for qi in qb[i]:
      qwords.append(dset.id2word[qi])
      if(dset.id2word[qi] == "_PADQ"): break
    if(i < 3): print("question: " + " ".join(qwords))
    qstr.append(" ".join(qwords))
    # print(ab[i])
    if(i < 3): print("answer: " + " ".join([dset.id2word[iw] for iw in ab[i]]))
    astr.append(" ".join([dset.id2word[iw] for iw in ab[i]]))
    if(i < 3): print("movie: %d, %s" % (nb[i], dset.id2word[nb[i]]))
    nstr.append(dset.id2word[nb[i]])
    # print(tb[i])
    twords = []
    for ti in tb[i]:
      if(ti <= dset.norm_size): 
        twords.append(dset.id2word[ti])
        if(dset.id2word[ti] == "_EOS"): break
      elif(ti == dset.norm_size + 1): twords.append(dset.id2word[nb[i]])
      else: twords.append(dset.id2word[ab[i][ti - dset.norm_size - 2]])
    if(i < 3): print("golden: " + " ".join(twords))
    goldstr.append(" ".join(twords))
    # print(ob[i])
    owords = []
    for oi in ob[i]:
      if(oi <= dset.norm_size): 
        owords.append(dset.id2word[oi])
        if(dset.id2word[oi] == "_EOS"): break
      elif(oi == dset.norm_size + 1): owords.append(dset.id2word[nb[i]])
      else: owords.append(dset.id2word[ab[i][oi - dset.norm_size - 2]])
    if(i < 3): print("predicted: " + " ".join(owords))
    predstr.append(" ".join(owords))
    if(i < 3): print("\n---------------------------------------\n")
  return qstr, astr, nstr, goldstr, predstr

def main():
  return 

if __name__ == "__main__":
  main()
