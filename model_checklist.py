# coding: utf-8

# ---- FQA Model, full version ----
# -- Francis Fu Yao 
# -- francis_yao@pku.edu.cn
# -- SUN 29TH AUG 2017

import tensorflow as tf
import numpy as np
import time
from sets import Set
from utils import metrics_full, write_test_batch

class Model(object):
  def __init__(self, config, name):
    self.gpu = config.gpu
    self.gpufrac = config.gpufrac
    self.config_name = config.config_name
    self.name = name
    self.is_kv = config.is_kv
    self.out_dir = config.out_dir
    self.vocab_size = config.vocab_size
    self.vocab_norm_size = config.vocab_norm_size
    self.vocab_spec_size = config.vocab_spec_size
    self.max_q_len = config.max_q_len
    self.max_a_len = config.max_a_len
    self.max_m_len = config.max_m_len
    self.state_size = config.state_size
    self.epoch_num = config.epoch_num
    self.batch_size = config.batch_size
    self.is_train = config.is_train
    self.qclass_size = config.qclass_size
    self.aclass_size = config.aclass_size
    self.memkey_size = config.memkey_size
    self.attn_hop_size = config.attn_hop_size
    self.out_mode_gate = config.out_mode_gate
    self.out = []
    self.out_test = []
    return 

  def build(self):
    print("Building the model ... ")
    # -- 0 Preparations
    state_size  = self.state_size
    max_q_len   = self.max_q_len
    max_a_len   = self.max_a_len
    max_m_len   = self.max_m_len
    vocab_size  = self.vocab_size
    norm_size   = self.vocab_norm_size
    spec_size   = self.vocab_spec_size
    qclass_size = self.qclass_size
    aclass_size = self.aclass_size
    memkey_size = self.memkey_size
    batch_size  = self.batch_size
    _GOOID = 2

    # -- 1 Placeholders
    # question
    self.input_q    = tf.placeholder(dtype = tf.int32, shape = [batch_size, max_q_len], name = "input_q") 
    self.input_qlen = tf.placeholder(dtype = tf.int32, shape = [batch_size], name = "input_qlen")
    self.input_qc   = tf.placeholder(dtype = tf.int32, shape = [batch_size], name = "input_qc")
    # memory
    self.input_mk   = tf.placeholder(dtype = tf.int32, shape = [batch_size, max_m_len], name = "input_mk")
    self.input_mv   = tf.placeholder(dtype = tf.int32, shape = [batch_size, max_m_len], name = "input_mv")
    self.input_mlen = tf.placeholder(dtype = tf.int32, shape = [batch_size], name = "input_mlen")
    # answer
    if(self.is_train):
      self.input_ain  = tf.placeholder(dtype = tf.int32, shape = [max_a_len, batch_size], name = "input_ain")
      self.input_aou  = tf.placeholder(dtype = tf.int32, shape = [max_a_len, batch_size], name = "input_aou")
      self.input_alen = tf.placeholder(dtype = tf.int32, shape = [batch_size], name = "input_alen")
    self.input_ac = tf.placeholder(dtype = tf.int32, shape = [batch_size], name = "input_ac")
    # movie
    self.input_movi = tf.placeholder(dtype = tf.int32, shape = [batch_size], name = "input_movie")
    self.out_record_key = tf.placeholder(dtype = tf.float32, 
                                         shape = [batch_size, max_a_len, state_size], 
                                         name = "out_record_key")
    self.out_record_val = tf.placeholder(dtype = tf.float32, 
                                         shape = [batch_size, max_a_len, state_size], 
                                         name = "out_record_val")
    self.keep_prob  = tf.placeholder(dtype = tf.float32, shape = (), name = "keep_prob")

    # -- 2 Build embeddings 
    print("\nBuilding embeddings ... ")
    with tf.device("/cpu:0"):
      embed_word   = tf.get_variable(name = "embed_word",
                                     shape = [vocab_size, state_size],
                                     dtype = tf.float32, 
                                     initializer = tf.random_uniform_initializer(0.0, 1.0))
      embed_qclass = tf.get_variable(name = "embed_qclass",
                                     shape = [qclass_size, state_size],
                                     dtype = tf.float32, 
                                     initializer = tf.random_uniform_initializer(0.0, 1.0))
      embed_aclass = tf.get_variable(name = "embed_aclass",
                                     shape = [aclass_size, state_size],
                                     dtype = tf.float32, 
                                     initializer = tf.random_uniform_initializer(0.0, 1.0))
      embed_memkey = tf.get_variable(name = "embed_memkey",
                                     shape = [memkey_size, state_size],
                                     dtype = tf.float32, 
                                     initializer = tf.random_uniform_initializer(0.0, 1.0))
      embed_norm_key = tf.get_variable(name = "embed_norm_key", shape = [state_size],
        dtype = tf.float32, initializer = tf.random_uniform_initializer(0.0, 1.0))
    embed_q     = tf.nn.embedding_lookup(embed_word,   self.input_q)
    embed_qc    = tf.nn.embedding_lookup(embed_qclass, self.input_qc)
    embed_mk    = tf.nn.embedding_lookup(embed_memkey, self.input_mk)
    embed_mv    = tf.nn.embedding_lookup(embed_word,   self.input_mv)
    embed_movi  = tf.nn.embedding_lookup(embed_word,   self.input_movi)
    embed_ac  = tf.nn.embedding_lookup(embed_aclass, self.input_ac)
    if(self.is_train):
      embed_ain = tf.nn.embedding_lookup(embed_word,   self.input_ain)
    print("embed_q shape: ", embed_q.shape)   # [batch_size, max_q_len, state_size]
    print("embed_qc shape: ", embed_qc.shape) # [batch_size, state_size]
    print("embed_mk shape: ", embed_mk.shape) # [batch_size, max_mem_len, state_size]
    print("embed_mv shape: ", embed_mv.shape) # [batch_size, max_mem_len, state_size]
    print("embed_movie shape: ", embed_movi.shape) # [batch_size, state_size]
    print("embed_ac shape: ", embed_ac.shape)   # [batch_size, state_size]
    if(self.is_train):
      print("embed_ain shape: ", embed_ain.shape) # [max_a_len, batch_size, state_size]

    # -- 3 Question encoding RNN 
    print("\nquestion encoding ... ")
    encoder_cell = tf.contrib.rnn.LSTMCell(num_units = state_size)
    encoder_cell = tf.contrib.rnn.DropoutWrapper(encoder_cell, output_keep_prob = self.keep_prob)
    # Note: add number of layers here in the future!
    enc_init_state = encoder_cell.zero_state(batch_size = batch_size, dtype = tf.float32)
    (q_rnn_out, q_rnn_stt) = tf.nn.dynamic_rnn(cell = encoder_cell, 
                                               inputs = embed_q,
                                               sequence_length = self.input_qlen, 
                                               initial_state = enc_init_state)
    # use the last hidden state as the question encoding
    encoded_q = q_rnn_stt.c + q_rnn_stt.h
    print("encoded_q size: ", encoded_q.shape)
    # assert(encoded_q.shape == [batch_size, state_size])

    # --  4 Memory encoding
    print("\nmemory encoding ... ")
    mem_mask = tf.sequence_mask(self.input_mlen, max_m_len, dtype = tf.float32) # size: [batch_size, max_m_len]
    print("memory mask shape:", mem_mask.shape)
    # assert(mem_mask.shape == (batch_size, max_m_len))
    encoded_mk = tf.reduce_sum(embed_mk * tf.expand_dims(mem_mask, 2), 1) # size: [batch_size, state_size]
    encoded_mv = tf.reduce_sum(embed_mv * tf.expand_dims(mem_mask, 2), 1) # size: [batch_size, state_size]
    print("encoded_mem shape:", encoded_mk.shape)
    # assert(encoded_mk.shape == (batch_size, state_size))

    # -- 5 Wrap up encoding
    print("\nwrapping up encoding ... ")
    all_encoded = tf.concat([encoded_q, encoded_mk, encoded_mv, embed_qc, embed_ac], 1)
    dec_cin_W = tf.get_variable(name = "dec_cin_W", 
                               shape = [state_size * 5, state_size],
                               dtype = tf.float32,
                               initializer = tf.random_normal_initializer())
    dec_cin_b = tf.get_variable(name = "dec_cin_b", 
                               shape = [state_size],
                               dtype = tf.float32,
                               initializer = tf.zeros_initializer())
    dec_state_c = tf.matmul(all_encoded, dec_cin_W) + dec_cin_b
    dec_hin_W = tf.get_variable(name = "dec_hin_W", 
                               shape = [state_size * 5, state_size],
                               dtype = tf.float32,
                               initializer = tf.random_normal_initializer())
    dec_hin_b = tf.get_variable(name = "dec_hin_b", 
                               shape = [state_size],
                               dtype = tf.float32,
                               initializer = tf.zeros_initializer())
    dec_state_h = tf.matmul(all_encoded, dec_hin_W) + dec_hin_b

    # -- 6 Attention
    # Note: we skip attention to question here 
    # Note: we use multi-hop attention here 
    def attention(query, memory_keys, memory_vals, mlen, max_mlen, reuse, attn_name):
      with tf.variable_scope(attn_name) as attn_scope:
        if(reuse): attn_scope.reuse_variables()
        q_size = int(query.shape[-1])
        attn_key_W_q = tf.get_variable(name = "attn_key_W_q",
                                       shape = [q_size, state_size],
                                       dtype = tf.float32,
                                       initializer = tf.random_normal_initializer())
        qr = tf.matmul(query, attn_key_W_q)
        m_size = int(memory_keys.shape[-1])
        attn_key_W_m = tf.get_variable(name = "attn_key_W_m",
                                       shape = [m_size, state_size],
                                       dtype = tf.float32,
                                       initializer = tf.random_normal_initializer())
        attn_key_W_m = tf.reshape(tf.tile(attn_key_W_m, [batch_size, 1]), [batch_size, m_size, state_size])
        mk = tf.matmul(memory_keys, attn_key_W_m)
        attn_v = tf.get_variable(name = "attn_v", dtype = tf.float32,
          initializer = np.array([1.0] * state_size).astype(np.float32))
        attn_v = tf.reshape(attn_v, [1, 1, state_size])
        qrmk = tf.nn.sigmoid(tf.expand_dims(qr, 1) + mk)
        attn_e_nomask = tf.reduce_sum(qrmk + attn_v, 2)
        # mask
        attn_e_masked = attn_e_nomask * tf.sequence_mask(mlen, max_mlen, dtype = tf.float32)
        attn_e = tf.nn.softmax(attn_e_masked)
        attn_o = tf.reduce_sum(tf.expand_dims(attn_e, 2) * memory_vals, 1)
      return attn_o, attn_e, attn_e_nomask

    # -- 7 Decoder
    decoder_cell = tf.contrib.rnn.LSTMCell(num_units = state_size)
    decoder_cell = tf.contrib.rnn.DropoutWrapper(decoder_cell, output_keep_prob = self.keep_prob)
    # Note: add number of layers here in the future!
    prev_h = tf.contrib.rnn.LSTMStateTuple(dec_state_c, dec_state_h)
    if(self.is_train):
      dec_in = tf.unstack(embed_ain)
      label_steps = tf.one_hot(self.input_aou, norm_size + max_m_len)
      label_steps = tf.unstack(label_steps)
    loss_steps = []
    out_idx = []
    all_logits = []
    out_record_key = self.out_record_key
    out_record_val = self.out_record_val
    mem_all = tf.concat((embed_mk, embed_mv), 2)
    attn_hist = tf.get_variable(name = "attn_hist", dtype = tf.float32, 
      initializer = np.zeros([batch_size, max_m_len]).astype(np.float32))
    # debug
    dbg_attn_hist = []
    dbg_mem_use = []
    dbg_mem_new = []
    dbg_attn_use_e = []
    dbg_attn_new_e = []
    dbg_attn_mkey = []
    dbg_attn_mval = []
    dbg_attn_outkey = []
    dbg_attn_outval = []
    dbg_attn_q = []
    # Decoding
    print("\nStart decoding ... ")
    with tf.variable_scope("decoder") as decoder_scope:
      for i in range(max_a_len):
        if(i == 0): 
          current_i = tf.nn.embedding_lookup(embed_word, np.array([_GOOID] * self.batch_size))
          attn_reuse = False
        else:
          decoder_scope.reuse_variables()
          attn_reuse = True
          if(self.is_train):
            current_i = dec_in[i]
          else:
            current_i = prev_o
        # attention query
        query = prev_h.h

        mem_use = mem_all * tf.expand_dims(attn_hist, 2)
        mem_new = mem_all * tf.expand_dims((1 - attn_hist), 2)
        embed_mk_use = embed_mk * tf.expand_dims(attn_hist, 2)
        embed_mk_new = embed_mk * tf.expand_dims((1 - attn_hist), 2)
        embed_mv_use = embed_mv * tf.expand_dims(attn_hist, 2)
        embed_mv_new = embed_mv * tf.expand_dims((1 - attn_hist), 2)
        dbg_mem_use.append(mem_use)
        dbg_mem_new.append(mem_new)

        # checklist
        # None KV
        # if(self.is_kv == False):
        #   # output attention to used memory, all
        #   attn_use_o, attn_use_e, attn_use_e_nomask = attention(query, mem_use, mem_use, 
        #     self.input_mlen, max_m_len, attn_reuse, "attention_use")
        #   dbg_attn_use_e.append(attn_use_e)
        #   # output attention to memory not use, all 
        #   attn_new_o, attn_new_e, attn_new_e_nomask = attention(query, mem_new, mem_new, 
        #     self.input_mlen, max_m_len, attn_reuse, "attention_new")
        #   dbg_attn_new_e.append(attn_new_e)
        # # KV 
        # else:
        #   # output attention to used memory, key
        #   attn_use_key_o, attn_use_key_e, _ = attention(query, embed_mk_use, embed_mv_use, 
        #     self.input_mlen, max_m_len, attn_reuse, "attention_key_use")
        #   # output attention to used memory, val
        #   attn_use_val_o, attn_use_val_e, _ = attention(query, embed_mv_use, embed_mv_use, 
        #     self.input_mlen, max_m_len, attn_reuse, "attention_val_use")
        #   # output attention to memory not use, key
        #   attn_new_key_o, attn_new_key_e, _ = attention(query, embed_mk_new, embed_mv_new, 
        #     self.input_mlen, max_m_len, attn_reuse, "attention_key_new")
        #   # output attention to memory not use, val
        #   attn_new_val_o, attn_new_val_e, _ = attention(query, embed_mv_new, embed_mv_new, 
        #     self.input_mlen, max_m_len, attn_reuse, "attention_val_new")

        # no checklist
        # output attention to keys directly
        attn_mkey_o, attn_mkey_e, _ = attention(query, embed_mk, embed_mv,
          self.input_mlen, max_m_len, attn_reuse, "attention_key")
        dbg_attn_mkey.append(attn_mkey_e)

        # output attention to values directly
        attn_mval_o, attn_mval_e, _ = attention(query, embed_mv, embed_mv,
          self.input_mlen, max_m_len, attn_reuse, "attention_val")
        dbg_attn_mval.append(attn_mval_e)

        # output attention to question 
        attn_q_o, attn_q_e, attn_q_e_nomask = attention(query, q_rnn_out, q_rnn_out, 
          self.input_qlen, max_q_len, attn_reuse, "attention_q")
        dbg_attn_q.append(attn_q_e)

        out_len = i * tf.ones([batch_size], tf.float32)
        # output attention to output record key
        attn_outkey_o, attn_outkey_e, _ = attention(query, out_record_key, out_record_val, 
          out_len, max_a_len, attn_reuse, "attention_outkey")
        dbg_attn_outkey.append(attn_outkey_e)

        # output attention to output record val
        attn_outval_o, attn_outval_e, _ = attention(query, out_record_val, out_record_val, 
          out_len, max_a_len, attn_reuse, "attention_oval")
        dbg_attn_outval.append(attn_outval_e)

        # checklist 
        # if(self.is_kv == False):
        #   dec_in_wrap = tf.concat([current_i, 
        #                            attn_use_o,
        #                            attn_new_o,
        #                            attn_q_o,
        #                            attn_out_o], 1)
        # else:
        #   dec_in_wrap = tf.concat([current_i, 
        #                            attn_use_key_o, attn_use_val_o,
        #                            attn_new_key_o, attn_new_val_o,
        #                            attn_q_o,
        #                            attn_out_o], 1)

        # no checklist
        # attention to key, value, question, and output record
        dec_in_wrap = tf.concat([current_i, 
                                 attn_mkey_o, attn_mval_o,
                                 attn_q_o, 
                                 attn_outkey_o, attn_outval_o], 1)

        # no output record, no question attention
        # dec_in_wrap = tf.concat([current_i, 
        #                          attn_use_o,
        #                          attn_new_o], 1)
        current_o, current_h = decoder_cell(dec_in_wrap, prev_h) 
        prev_h = current_h


        # output projection to normal word
        out_norm_W = tf.get_variable(name = "out_norm_W", 
                                     shape = [state_size, norm_size],
                                     dtype = tf.float32,
                                     initializer = tf.random_normal_initializer())
        out_norm_b = tf.get_variable(name = "out_norm_b",
                                     shape = [norm_size],
                                     dtype = tf.float32,
                                     initializer = tf.zeros_initializer())
        attn_o_norm_e = tf.matmul(current_o, out_norm_W) + out_norm_b

        # output projection to memory
        # Note: may alternate this with direct projection
        out_spec_W = tf.get_variable(name = "out_spec_W",
                                     shape = [state_size, 2 * state_size],
                                     dtype = tf.float32,
                                     initializer = tf.random_normal_initializer())
        attn_o_spec = tf.matmul(current_o, out_spec_W)
        spec_mem = tf.concat([embed_mk, embed_mv], 2)
        attn_o_spec_e = tf.reduce_sum(tf.expand_dims(attn_o_spec, 1) * spec_mem, 2)
        attn_o_spec_e = tf.sequence_mask(self.input_mlen, max_m_len, dtype = tf.float32) * attn_o_spec_e

        # Output mode choose gate
        if(self.out_mode_gate):
          out_mode_W = tf.get_variable(name = "out_mode_W", shape = [state_size, 1], 
            dtype = tf.float32, initializer = tf.random_normal_initializer())
          out_mode_b = tf.get_variable(name = "out_mode_b", shape = [1], 
            dtype = tf.float32, initializer = tf.random_normal_initializer())
          out_mode = tf.nn.sigmoid(tf.matmul(current_o, out_mode_W) + out_mode_b)
          attn_o_norm_e = attn_o_norm_e * out_mode # [batch_size, norm_size] [batch_size, 1]
          attn_o_spec_e = attn_o_spec_e * (1 - out_mode) # [batch_size, norm_size] [batch_size, 1]

        # softmax and logits
        logits = tf.concat([attn_o_norm_e, attn_o_spec_e], 1) # size: [batch_size, pred_size]

        # add to attention history 
        attn_hist += tf.slice(tf.nn.softmax(logits), [0, norm_size], [batch_size, max_m_len])
        dbg_attn_hist.append(attn_hist)

        # logits = logits + softmax_b
        if(self.is_train):
          loss_step = tf.nn.softmax_cross_entropy_with_logits(labels = label_steps[i], logits = logits)
          loss_steps.append(loss_step)

        # output index and get the next input
        all_logits.append(logits)
        out_index = tf.argmax(logits, 1)
        out_idx.append(out_index)
        out_choose_norm, out_choose_spec = tf.split(
          tf.one_hot(out_index, norm_size + max_m_len), [norm_size, max_m_len], 1)
        if(i == 0): 
          print("out_choose_norm size: ", out_choose_norm.shape)
        norm_emb = embed_word[:norm_size]
        o_norm = tf.nn.embedding_lookup(norm_emb, tf.cast(tf.argmax(out_choose_norm, 1), tf.int32))
        if(i == 0): 
          print("o_norm size: ", o_norm.shape)
        o_norm = tf.expand_dims(tf.reduce_sum(out_choose_norm, 1), 1) * o_norm
        if(i == 0): 
          print("o_norm size: ", o_norm.shape)
        # o_norm = tf.reduce_sum(tf.expand_dims(out_choose_norm, 2) * tf.expand_dims(norm_emb, 0), 1)
        o_spec = tf.reduce_sum(tf.expand_dims(out_choose_spec, 2) * embed_mv, 1)
        out_val = o_norm + o_spec
        # out_choose_norm size: [batch_size, norm_size]
        o_norm_key = tf.expand_dims(tf.reduce_sum(out_choose_norm, 1), 1) * tf.expand_dims(embed_norm_key, 0)
        o_spec_key = tf.reduce_sum(tf.expand_dims(out_choose_spec, 2) * embed_mk, 1)
        out_key = o_norm_key + o_spec_key
        if(i == 0): 
          print("o_norm_key size: ", o_norm_key.shape)

        if(self.is_train == False):
          prev_o = out_val

        # add to record
        out_new_key  = tf.expand_dims(tf.one_hot([i], max_a_len), 2) * tf.expand_dims(out_key, 1)
        out_new_val  = tf.expand_dims(tf.one_hot([i], max_a_len), 2) * tf.expand_dims(out_val, 1)
        # assert(dec_out_new.shape == (batch_size, max_a_len, state_size))
        if(i == 0): print("out_new_key shape:", out_new_key.shape)
        out_record_key = out_record_key + out_new_key
        out_record_val = out_record_key + out_new_val

    # -- 8 Output, loss and optimizer
    # debug:
    dbg_attn_hist = tf.stack(dbg_attn_hist)
    dbg_mem_use = tf.stack(dbg_mem_use)
    dbg_mem_new = tf.stack(dbg_mem_use)
    dbg_attn_use_e = tf.stack(dbg_attn_use_e)
    dbg_attn_new_e = tf.stack(dbg_attn_new_e)
    dbg_attn_mkey = tf.transpose(tf.stack(dbg_attn_mkey), [1, 0 ,2]) # [batch_size, timestep, max_m_len]
    dbg_attn_mval = tf.transpose(tf.stack(dbg_attn_mval), [1, 0 ,2]) 
    dbg_attn_q = tf.transpose(tf.stack(dbg_attn_q), [1, 0 ,2]) 
    dbg_attn_outkey = tf.transpose(tf.stack(dbg_attn_outkey), [1, 0 ,2])
    dbg_attn_outval = tf.transpose(tf.stack(dbg_attn_outval), [1, 0 ,2])
    print("\nCalculate loss ...")
    out_idx = tf.transpose(tf.stack(out_idx)) # size = [batch_size, max_a_len]
    all_logits = tf.transpose(tf.stack(all_logits), [1, 0, 2]) # size = [batch_size, max_a_len, norm_size + max_m_len]
    if(self.is_train == False):
      self.out = [out_idx]
      return 
    # loss
    loss = tf.stack(loss_steps)
    loss = tf.transpose(loss)  # size: [batch_size, max_s_len - 1]
    # mask loss 
    loss_mask = tf.sequence_mask(self.input_alen, max_a_len, tf.float32) # size: [batch_size, max_s_len]
    loss = tf.reduce_mean(loss_mask * loss)
    optimizer = tf.train.AdamOptimizer()
    train_op = optimizer.minimize(loss)
    self.out_train = [loss ,train_op, out_idx]
    self.out = [loss ,train_op, out_idx, all_logits]
    # self.out.extend([dbg_attn_hist, dbg_mem_use, dbg_mem_new, dbg_attn_use_e, dbg_attn_new_e])
    self.out.extend([dbg_attn_mkey, dbg_attn_mval, dbg_attn_q, dbg_attn_outkey, dbg_attn_outval])
    print("finished")
    return

  def train(self, dset, mvalid, mtest):
    print("\nstart training ... ")
    config = tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = self.gpufrac
    sess = tf.Session(config=config)
    sess.run(tf.global_variables_initializer())
    total_cases = dset.total_cases_train
    total_batches = total_cases / self.batch_size + 1
    print("%d cases in total" % total_cases)
    print("%d batches in total" % total_batches)
    print("self batchsize = %d" % self.batch_size)
    epoch_start_time = time.time()
    for ei in range(self.epoch_num):
      start_time = time.time()
      set_loss = 0.0
      for bi in range(total_batches):
        qb, qlenb, qcb, ainb, aoub, alenb, acb, mb, mkb, mvb, mlenb = dset.get_next_batch("train", self.batch_size)
        dec_init_record = np.zeros([self.batch_size, self.max_a_len, self.state_size])
        # print("dec_init_record shape:", dec_init_record.shape)
        feed_dict = dict()
        # question
        feed_dict[self.input_q] = qb
        feed_dict[self.input_qlen] = qlenb
        feed_dict[self.input_qc] = qcb
        # answer
        feed_dict[self.input_ain] = ainb
        feed_dict[self.input_aou] = aoub
        feed_dict[self.input_alen] = alenb
        feed_dict[self.input_ac] = acb
        # memory
        feed_dict[self.input_mk] = mkb
        feed_dict[self.input_mv] = mvb
        feed_dict[self.input_mlen] = mlenb
        # movie
        feed_dict[self.input_movi] = mb
        # initial decoder record
        feed_dict[self.out_record_key] = dec_init_record
        feed_dict[self.out_record_val] = dec_init_record
        # dropout
        feed_dict[self.keep_prob] = 0.8
        loss, _, out_idx = sess.run(self.out_train, feed_dict)
        set_loss += loss
        if(bi % 20 == 0 and bi > 0):
          # if(bi % 100 == 0): 
          #   print_batch(dset, qb, qlenb, qcb, ainb, aoub, alenb, acb, mb, mkb, mvb, mlenb, out_idx)
            # mvalid.test(dset, ei, sess)
            # return 
          print("\n------")
          print("model %s, epoch %d, batch %d, current loss %.4f, set loss: %.4f" % (self.config_name, ei, bi, loss, set_loss / 20.0))
          # metrics(aoub, out_idx, mkb, mvb, dset, bi)
          set_loss = 0.0
          print("time cost: %.2f" % (time.time() - start_time))
          start_time = time.time()
      print("----------------------------------")
      print("epoch %d finished! testing ... " % ei)
      mvalid.test(dset, ei, sess)
      mtest.test(dset, ei, sess)
      print("----------------------------------")
    print("total time cost: %.2f" % (time.time() - epoch_start_time))
    return 

  def test(self, dset, ei, sess):
    print("testing %s set" % self.name)
    if(self.name == "valid"):
      total_test_batch = dset.total_cases_valid / self.batch_size + 1
    else:
      total_test_batch = dset.total_cases_test / self.batch_size + 1
    redundancy += 0.0
    total_coverage_small += 0.0
    total_coverage_large_partial += 0.0
    total_coverage_large_perfect += 0.0
    enrichment += 0.0
    mem_cover += 0.0
    fd = open("%s/%s_%s_epoch%d_gpu%d.txt" % (self.out_dir, self.config_name, self.name, ei, self.gpu), "w")
    for bi in range(total_test_batch):
      qb, qlenb, qcb, ainb, aoub, alenb, acb, mb, mkb, mvb, mlenb = dset.get_next_batch(self.name, self.batch_size)
      dec_init_record = np.zeros([self.batch_size, self.max_a_len, self.state_size])
      # print("dec_init_record shape:", dec_init_record.shape)
      feed_dict = dict()
      # question
      feed_dict[self.input_q] = qb
      feed_dict[self.input_qlen] = qlenb
      feed_dict[self.input_qc] = qcb
      # note: not golden answer here 
      feed_dict[self.input_ac] = acb
      # memory
      feed_dict[self.input_mk] = mkb
      feed_dict[self.input_mv] = mvb
      feed_dict[self.input_mlen] = mlenb
      # movie
      feed_dict[self.input_movi] = mb
      # initial decoder record
      feed_dict[self.out_record_key] = dec_init_record
      feed_dict[self.out_record_val] = dec_init_record
      # dropout
      feed_dict[self.keep_prob] = 1.0
      out_idx = sess.run(self.out, feed_dict)[0]

      # coverage, coverage_large, coverage_perfect, cover_targets, covered, repeat_cnt, large_set_tags = metrics(aoub, 
      #   out_idx, mkb, mvb, dset, bi)
      # total_coverage += coverage
      # total_coverage_large += coverage_large
      # total_coverage_perfect += coverage_perfect
      # total_repeat += repeat_cnt

      metrics_ret = metrics_full(aoub, qcb, out_idx, mkb, mvb, mlenb, dset, bi)
      redundancy += metrics_ret[0]
      total_coverage_small += metrics_ret[1]
      total_coverage_large_partial += metrics_ret[1]
      total_coverage_large_perfect += metrics_ret[2]
      enrichment += metrics_ret[3]
      mem_cover += metrics_ret[4]

      write_test_batch(qb, qlenb, aoub, alenb, mb, mkb, mvb, mlenb, out_idx, 
        cover_targets, covered, dset, fd, large_set_tags)
    redundancy /= total_test_batch
    total_coverage_small /= total_test_batch
    total_coverage_large_partial /= total_test_batch
    total_coverage_large_perfect /= total_test_batch
    enrichment /= total_test_batch
    mem_cover /= total_test_batch
    print("redundancy: %.4f, enrichment: %.4f, mem_cover: %.4f" % (redundancy, enrichment, mem_cover))
    print("total coverage, small: %.4f, large_partial: %.4f, large_perfect: %.4f" % 
      (total_coverage_small, total_coverage_large_partial, total_coverage_large_perfect))
    return 

def print_batch(dset, qb, qlenb, qcb, ainb, aoub, alenb, acb, mb, mkb, mvb, mlenb, out_idx):
  ainb = ainb.T
  aoub = aoub.T
  for i in range(3):
    print("\ncase %d:" % i)
    case = {"q": qb[i], "qlen": qlenb[i], "c": qcb[i], 
            "ain": ainb[i], "aou": aoub[i], "alen": alenb[i], "ac": acb[i],
            "out_idx": out_idx[i],
            "m": mb[i],
            "mk": mkb[i], "mv": mvb[i], "mlen": mlenb[i]}
    show_case(dset, case)
  return

def show_case(dset, case):
  word2id = dset.word2id
  id2word = dset.id2word
  print("q, len = %d" % case["qlen"])
  # print(case["q"][:case["qlen"]])
  print("   ".join([id2word[i] for i in case["q"][: case["qlen"]]]))
  # print("ain, len = %d" % case["alen"])
  # print(case["ain"][: case["alen"] + 1])
  # print("   ".join([id2word[i] for i in case["ain"][: case["alen"] + 1]]))
  print("aou: ")
  # print(case["aou"][: case["alen"] + 1])
  print("   ".join([id2word[i] if i < dset.norm_word_cnt 
    else ("(loc %d -> " % (i - dset.norm_word_cnt)) + id2word[case["mv"][i - dset.norm_word_cnt]] + ")"
    for i in case["aou"][: case["alen"] + 1]]))
  print("out_idx: ")
  # print(case["out_idx"])
  print("   ".join([id2word[i] if i < dset.norm_word_cnt 
    else ("(loc %d -> " % (i - dset.norm_word_cnt)) + id2word[case["mv"][i - dset.norm_word_cnt]] + ")"
    for i in case["out_idx"]]))
  print("memory, len = %d" % case["mlen"])
  # print("k:", case["mk"][: case["mlen"]])
  # print("v:", case["mv"][: case["mlen"]])
  print([ (idx, dset.mem_id2labels[i[0]], id2word[i[1]]) 
    for idx, i in enumerate(zip(case["mk"][:case["mlen"]], case["mv"][:case["mlen"]])) ])
  print("qc: %s, m: %s, ac: %s" % (dset.id2qcls[case["c"]], id2word[case["m"]], dset.id2acls[case["ac"]]))
  return 






