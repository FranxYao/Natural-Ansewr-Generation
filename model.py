# coding: utf-8
# ---- FQA Model, full version ----
# -- Francis Fu Yao 
# -- francis_yao@pku.edu.cn
# -- SUN 29TH AUG 2017

import tensorflow as tf
import numpy as np
import time
from sets import Set

class Model(object):
  def __init__(self, config, name):
    self.gpu = config.gpu
    self.config_name = config.config_name
    self.name = name
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
    self.config_name = config.config_name
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
    batch_size = self.batch_size
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
    # Auxiliary
    self.dec_out_record = tf.placeholder(dtype = tf.float32, 
                                         shape = [batch_size, max_a_len, state_size], 
                                         name = "dec_out_record")
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
    # encoded_q = q_rnn_stt.c + q_rnn_stt.h
    encoded_q = q_rnn_stt.h
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
        attn_key_W_q = tf.get_variable(name = "attn_key_W_q",
                                       shape = [state_size, state_size],
                                       dtype = tf.float32,
                                       initializer = tf.random_normal_initializer())
        qr = tf.matmul(query, attn_key_W_q)
        attn_key_W_m = tf.get_variable(name = "attn_key_W_m",
                                       shape = [state_size, state_size],
                                       dtype = tf.float32,
                                       initializer = tf.random_normal_initializer())
        attn_key_W_m = tf.reshape(tf.tile(attn_key_W_m, [batch_size, 1]), [batch_size, state_size, state_size])
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

    # Attention to key
    # Justification: since the key is basically the class of the value, 
    #                it is more related to type of answer words
    #                i.e: question asking for "director" will more relatd to key "director"
    def attention_key(query, reuse):
      mlen = self.input_mlen
      with tf.variable_scope("attention_key") as attn_scope:
        if(reuse): attn_scope.reuse_variables()
        attn_key_W = tf.get_variable(name = "attn_key_W", 
                                     shape = [state_size, state_size],
                                     dtype = tf.float32,
                                     initializer = tf.random_normal_initializer())
        query = tf.matmul(query, attn_key_W)
        attn_e_nomask = tf.reduce_sum(tf.expand_dims(query, 1) * embed_mk, 2)  # Note: energy based on k
        # attn_e_nomask = tf.nn.sigmoid(attn_e_nomask)
        # assert(attn_e.shape == (batch_size, max_m_len))
        if(reuse != True): print("attn_e shape: ", attn_e_nomask.shape)
        attn_e = tf.nn.softmax(attn_e_nomask)
        # attn_e = tf.nn.softmax(attn_e_nomask * tf.sequence_mask(mlen, max_m_len, dtype = tf.float32))
        attn_o = tf.reduce_sum(tf.expand_dims(attn_e, 2) * embed_mv, 1) # Note: output based on v
        # assert(attn_o.shape == (batch_size, state_size))
        if(reuse != True): print("attn_o shape: ", attn_o.shape)
      return attn_o, attn_e, attn_e_nomask

    # Attention to value
    # Justification: the key is more relatied to the movie, as well as the type of answer sentences
    #                i.e: wiki sentences may attend to more words
    def attention_val(query, reuse):
      mlen = self.input_mlen
      with tf.variable_scope("attention_val") as attn_scope:
        if(reuse): attn_scope.reuse_variables()
        attn_val_W = tf.get_variable(name = "attn_val_W", 
                                     shape = [state_size, state_size],
                                     dtype = tf.float32,
                                     initializer = tf.random_normal_initializer())
        query = tf.matmul(query, attn_val_W)
        attn_e_nomask = tf.reduce_sum(tf.expand_dims(query, 1) * embed_mv, 2)  # Note: energy and output all based on v
        # attn_e_nomask = tf.nn.sigmoid(attn_e_nomask)
        # assert(attn_e.shape == (batch_size, max_m_len))
        if(reuse != True): print("attn_e shape: ", attn_e_nomask.shape)
        attn_e = tf.nn.softmax(attn_e_nomask)
        # attn_e = tf.nn.softmax(attn_e_nomask * tf.sequence_mask(mlen, max_m_len, dtype = tf.float32))
        attn_o = tf.reduce_sum(tf.expand_dims(attn_e, 2) * embed_mv, 1)
        # assert(attn_o.shape == (batch_size, state_size))
        if(reuse != True): print("attn_o shape: ", attn_o.shape)
      return attn_o, attn_e, attn_e_nomask

    # Attention to previous output
    # self.dec_out_record = tf.placeholder(dtype = tf.float32, 
    #                                      shape = [batch_size, max_a_len, state_size], 
    #                                      name = "dec_out_record")
    dec_out_record = self.dec_out_record
    def attention_out(query, stepi, reuse):
      alen = stepi * tf.ones([batch_size], tf.float32)
      with tf.variable_scope("attention_out") as attn_scope:
        if(reuse): attn_scope.reuse_variables()
        attn_out_W = tf.get_variable(name = "attn_out_W", 
                                     shape = [state_size, state_size],
                                     dtype = tf.float32,
                                     initializer = tf.random_normal_initializer())
        query = tf.matmul(query, attn_out_W)
        attn_e_nomask = tf.reduce_sum(tf.expand_dims(query, 1) * dec_out_record, 2)
        # attn_e_nomask = tf.nn.sigmoid(attn_e_nomask)
        if(reuse != True): 
          print("query shape: ", query.shape)
          print("dec_out_record shape: ", dec_out_record.shape)
        # assert(attn_e.shape == (batch_size, max_m_len))
        if(reuse != True): print("attn_e shape: ", attn_e_nomask.shape)
        attn_e = tf.nn.softmax(attn_e_nomask)
        # attn_e = tf.nn.softmax(attn_e_nomask * tf.sequence_mask(alen, max_a_len, dtype = tf.float32))
        attn_o = tf.reduce_sum(tf.expand_dims(attn_e, 2) * dec_out_record, 1)
        # assert(attn_o.shape == (batch_size, state_size))
        if(reuse != True): print("attn_o shape: ", attn_o.shape)
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
    # dec_out_record = self.dec_out_record
    # debug
    dbg_dec_o = []
    dbg_attn_mk_o = []
    dbg_attn_mk_e = []
    dbg_attn_mk_e_nomask = []
    dbg_attn_mv_o = []
    dbg_attn_mv_e = []
    dbg_attn_mv_e_nomask = []
    dbg_attn_out_o = []
    dbg_attn_out_e = []
    dbg_attn_out_e_nomask = []
    # dbg_attn_out_mo = []
    dbg_dec_out_record = []
    dbg_attn_o_norm_e = []
    dbg_attn_o_spec_e = []
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

        # attention to key
        attn_mk_o, attn_mk_e, attn_mk_e_nomask = attention(query, embed_mk, embed_mv, 
          self.input_mlen, max_m_len, attn_reuse, "attention_mkey")
        # attn_mk_o, attn_mk_e, attn_mk_e_nomask = attention_key(query, attn_reuse)
        # for ai in range(self.attn_hop_size - 1):
        #   attn_mk_o, attn_mk_e, attn_mk_e_nomask = attention_key(attn_mk_o, embed_qc, encoded_q, True)
        dbg_attn_mk_o.append(attn_mk_o)
        dbg_attn_mk_e.append(attn_mk_e)
        dbg_attn_mk_e_nomask.append(attn_mk_e_nomask)

        # attention to value 
        attn_mv_o, attn_mv_e, attn_mv_e_nomask = attention(query, embed_mv, embed_mv, 
          self.input_mlen, max_m_len, attn_reuse, "attention_mval")
        # attn_mv_o, attn_mv_e, attn_mv_e_nomask = attention_val(query, attn_reuse)
        # for ai in range(self.attn_hop_size - 1):
        #   attn_mv_o, attn_mv_e, attn_mv_e_nomask = attention_val(attn_mv_o, embed_movi, embed_ac, encoded_q, True)
        dbg_attn_mv_o.append(attn_mv_o)
        dbg_attn_mv_e.append(attn_mv_e)
        dbg_attn_mv_e_nomask.append(attn_mv_e_nomask)
        
        # attention to output record
        out_len = i * tf.ones([batch_size], tf.float32)
        attn_out_o, attn_out_e, attn_out_e_nomask = attention(query, dec_out_record, dec_out_record, 
          out_len, max_a_len, attn_reuse, "attention_out")
        # attn_out_o, attn_out_e, attn_out_e_nomask = attention_out(query, i, attn_reuse)
        # for ai in range(self.attn_hop_size - 1):
        #   attn_out_o, attn_out_e, attn_out_mo, attn_out_e_nomask = attention_out(attn_out_o, True, i)
        dbg_attn_out_o.append(attn_out_o)
        dbg_attn_out_e.append(attn_out_e)
        # dbg_attn_out_mo.append(attn_out_mo)
        dbg_attn_out_e_nomask.append(attn_out_e_nomask)

        # attention to question
        if(i == 0): print("q_rnn_out shape: ", q_rnn_out.shape)
        attn_q_o, attn_q_e, attn_q_e_nomask = attention(query, q_rnn_out, q_rnn_out, 
          self.input_qlen, max_q_len, attn_reuse, "attention_q")
        if(i == 0): print("attn_q_o shape: ", attn_q_o.shape)

        # call lstm cell
        dec_in_wrap = tf.concat([current_i, attn_mk_o, attn_mv_o, attn_out_o, attn_q_o], 1)
        current_o, current_h = decoder_cell(dec_in_wrap, prev_h) 
        prev_h = current_h
        dbg_dec_o.append(current_o)

        # add to record
        dec_out_new  = tf.expand_dims(tf.one_hot([i], max_a_len), 2) * tf.expand_dims(current_o, 1)
        # assert(dec_out_new.shape == (batch_size, max_a_len, state_size))
        if(i == 0): print("dec_out_new shape:", dec_out_new.shape)
        dec_out_record = dec_out_record + dec_out_new
        dbg_dec_out_record.append(dec_out_record)

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
        dbg_attn_o_norm_e.append(attn_o_norm_e)

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
        dbg_attn_o_spec_e.append(attn_o_spec_e)

        # Optional: output mode choose gate
        if(self.out_mode_gate):
          out_mode_W = tf.get_variable(name = "out_mode_W", shape = [state_size, 1], 
            dtype = tf.float32, initializer = tf.random_normal_initializer())
          out_mode_b = tf.get_variable(name = "out_mode_b", shape = [1], 
            dtype = tf.float32, initializer = tf.random_normal_initializer())
          out_mode = tf.nn.sigmoid(tf.matmul(current_o, out_mode_W) + out_mode_b)
          attn_o_norm_e = attn_o_norm_e * out_mode # [batch_size, norm_size] [batch_size, 1]
          attn_o_spec_e = attn_o_spec_e * (1 - out_mode) # [batch_size, norm_size] [batch_size, 1]

        # softmax and logits
        # softmax_b = tf.get_variable(name = "softmax_b", 
        #                             shape = [norm_size + max_m_len], # [norm_size + 1] + [movie] + ans
        #                             dtype = tf.float32,
        #                             initializer = tf.zeros_initializer())
        logits = tf.concat([attn_o_norm_e, attn_o_spec_e], 1) # size: [batch_size, pred_size]
        # logits = logits + softmax_b
        if(self.is_train):
          loss_step = tf.nn.softmax_cross_entropy_with_logits(labels = label_steps[i], logits = logits)
          loss_steps.append(loss_step)
        # output index and get the next input
        all_logits.append(logits)
        out_index = tf.argmax(logits, 1)
        out_idx.append(out_index)

        if(self.is_train == False):
          out_choose_norm, out_choose_spec = tf.split(tf.one_hot(out_index, norm_size + max_m_len), [norm_size, max_m_len], 1)
          norm_emb = embed_word[:norm_size]
          o_norm = tf.reduce_sum(tf.expand_dims(out_choose_norm, 2) * tf.expand_dims(norm_emb, 0), 1)
          o_spec = tf.reduce_sum(tf.expand_dims(out_choose_spec, 2) * embed_mv, 1)
          prev_o = o_norm + o_spec

    # -- 8 Output, loss and optimizer
    # debug:
    dbg_dec_o = tf.stack(dbg_dec_o)
    dbg_attn_mk_o = tf.stack(dbg_attn_mk_o)
    dbg_attn_mk_e = tf.stack(dbg_attn_mk_e)
    dbg_attn_mk_e_nomask = tf.stack(dbg_attn_mk_e_nomask)
    dbg_attn_mv_o = tf.stack(dbg_attn_mv_o)
    dbg_attn_mv_e = tf.stack(dbg_attn_mv_e)
    dbg_attn_mv_e_nomask = tf.stack(dbg_attn_mv_e_nomask)
    dbg_attn_out_o = tf.stack(dbg_attn_out_o)
    dbg_attn_out_e = tf.stack(dbg_attn_out_e)
    # dbg_attn_out_mo = tf.stack(dbg_attn_out_mo)
    dbg_attn_out_e_nomask = tf.stack(dbg_attn_out_e_nomask)
    dbg_dec_out_record = tf.stack(dbg_dec_out_record)
    dbg_attn_o_norm_e = tf.stack(dbg_attn_o_norm_e)
    dbg_attn_o_spec_e = tf.stack(dbg_attn_o_spec_e)
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
    self.out.append(dbg_dec_o)
    self.out.append(dbg_attn_mk_o)
    self.out.append(dbg_attn_mk_e)
    self.out.append(dbg_attn_mk_e_nomask)
    self.out.append(dbg_attn_mv_o)
    self.out.append(dbg_attn_mv_e)
    self.out.append(dbg_attn_mv_e_nomask)
    self.out.append(dbg_attn_out_o)
    self.out.append(dbg_attn_out_e)
    # self.out.append(dbg_attn_out_mo)
    self.out.append(dbg_attn_out_e_nomask)
    self.out.append(dbg_dec_out_record)
    self.out.append(dbg_attn_o_norm_e)
    self.out.append(dbg_attn_o_spec_e)
    print("finished")
    return

  def train(self, dset, mvalid, mtest):
    print("\nstart training ... ")
    sess = tf.Session()
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
        feed_dict[self.dec_out_record] = dec_init_record
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
          metrics(aoub, out_idx, mkb, mvb, dset, bi)
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
    # print("batch_size = %d" % self.batch_size)
    if(self.name == "valid"):
      total_test_batch = dset.total_cases_valid / self.batch_size + 1
    else:
      total_test_batch = dset.total_cases_test / self.batch_size + 1
    # print("%d test batches in total" % total_test_batch)
    total_coverage = 0
    total_coverage_large = 0
    total_coverage_perfect = 0
    total_repeat_cnt = 0
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
      feed_dict[self.dec_out_record] = dec_init_record
      # dropout
      feed_dict[self.keep_prob] = 1.0
      out_idx = sess.run(self.out, feed_dict)[0]
      coverage, coverage_large, coverage_perfect, cover_targets, covered, repeat_cnt, large_set_tags = metrics(aoub, 
        out_idx, mkb, mvb, dset, bi)
      total_repeat_cnt += repeat_cnt
      total_coverage += coverage
      total_coverage_large += coverage_large
      total_coverage_perfect += coverage_perfect
      write_test_batch(qb, qlenb, aoub, alenb, mb, mkb, mvb, mlenb, out_idx,
        cover_targets, covered, dset, fd, large_set_tags)
    total_coverage /= total_test_batch
    total_coverage_large /= total_test_batch
    total_coverage_perfect /= total_test_batch
    redundancy = total_repeat_cnt / (total_coverage_large * 100)
    print("total_coverage = %.2f, large: %.2f, perfect: %.2f, repeat: %d, redundancy: %.2f" % 
      (total_coverage, total_coverage_large, total_coverage_perfect, total_repeat_cnt, redundancy))
    return 
  






