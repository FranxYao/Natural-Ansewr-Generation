# ---- Utils for FQA
# -- Francis Fu Yao 
# -- francis_yao@pku.edu.cn
# -- WED 11TH OCT 2017

from sets import Set
from colored import fore, back, style
import numpy as np


def write_test_batch(qb, qlenb, aoub, alenb, mb, mkb, mvb, mlenb, out_idx, cover_targets, covered, dset, fd, large_set_tags):
  aoub = aoub.T
  ct_cnt = 0
  cv_cnt = 0
  for i in range(len(qb)):
    if(large_set_tags[i] == 0): continue
    qw = "   ".join(dset.id2word[idx] for idx in qb[i][: qlenb[i]])
    qw = "q:   "  + qw
    aouw = "   ".join(dset.id2word[idx] if idx < dset.norm_word_cnt else
      ("(loc %d -> " % (idx - dset.norm_word_cnt) + dset.id2word[mvb[i][idx - dset.norm_word_cnt]] + ")")
      for idx in aoub[i][: alenb[i]])
    aouw = "aou: " + aouw
    outw = "   ".join(dset.id2word[idx] if idx < dset.norm_word_cnt else
      ("(loc %d -> " % (idx - dset.norm_word_cnt) + dset.id2word[mvb[i][idx - dset.norm_word_cnt]] + ")")
      for idx in out_idx[i])
    outw = outw.split("_EOS")[0] + "_EOS"
    outw = "out: " + outw
    mw = " | ".join("%d, %s, %s" % (ii, dset.mem_id2labels[idx[0]], dset.id2word[idx[1]])
        for ii, idx in enumerate(zip(mkb[i][: mlenb[i]], mvb[i][: mlenb[i]])) )
    mw = "mm:  (" + mw + ")"
    ct = " , ".join(idx for idx in cover_targets[i])
    ct = "ct:   " + ct
    cv = " , ".join(idx for idx in covered[i])
    cv = "cv:   " + cv
    ct_cnt += len(cover_targets[i])
    cv_cnt += len(covered[i] & cover_targets[i])
    fd.write("%s\n%s\n%s\n%s\n%s\n%s\n--------\n\n" % (qw, aouw, outw, mw, ct, cv))
  fd.write("\n\ncover targets: %d, covered: %d\n\n\n" % (ct_cnt, cv_cnt))
  return 

qclass2memlebel = { "director":     ("ans_director", "directed_by"), 
                    "writer":       ("ans_writer", "written_by"), 
                    "actor":        ("ans_actor", "starred_actors"), 
                    "release_year": ("ans_release_year", "release_year"),
                    "language":     ("ans_language", "in_language"), 
                    "tag":          ("ans_tag", "has_tags"), 
                    "genre":        ("ans_genre", "has_genre"), 
                    "vote":         ("ans_vote", "has_imdb_votes"), 
                    "rating":       ("ans_rating", "has_imdb_rating"), 
                    "wiki":         ("") }

def find_targets(qc, mk, mv, mlen, dset):
  targets = []
  for i in range(mlen):
    if(dset.mem_id2labels[mk[i]] in qclass2memlebel[dset.id2qcls[qc]]): 
      targets.append(dset.id2word(mv[i]))
  targets = Set(targets)
  return targets

# bi: batch id
# redundancy: repeated word / word predicted (mis-retreived memory unit per effective memory unit) per case
# enrichment: new words covered except target word per case
# memory coverage: what percentage of memory words are covered? average over case
# target coverage: how many target words are covered(small/ large target set, partial/ perfect coverage)?
#   coverage_small:         what percentage of small target sets are covered?
#   coverage_large_partial: what percentage of words in larget target sets are covered?
#   coverage_large_perfect:   waht percentage of large target sets are perfectly covered?
# subject-object maintainance: the subject-object relationship should be maintained
def metrics_full(aoub, qcb, out_idx, mkb, mvb, mlenb, dset, bi):
  aoub = aoub.T
  batch_size = len(aoub)
  redundancy = 0.0
  small_target_cnt = 0
  large_target_cnt = 0
  target_coverage_small = 0.0
  target_coverage_large_partial = 0.0
  target_coverage_large_perfect = 0.0
  enrichment = 0.0
  mem_cover = 0.0
  for i in range(batch_size):
    targets = find_target(qcb[i], mkb[i], mvb[i], mlenb[i], dset)
    predicted = []
    alen = 0
    for j in range(dset.max_a_len):
      if(out_idx[i][j] == dset.word2id["_EOS"]): break
    alen = j + 1
    for j in range(alen):
      aid = out_idx[i][j]
      if(aid > dset.norm_word_cnt and dset.mem_id2labels[mkb[i][aid - dset.norm_word_cnt]] != "movie"):
        predicted.append(dset.id2word[mvb[i][aid - dset.norm_word_cnt]])
    predicted_clean = Set(predicted)
    # redundancy
    redundancy += len(predicted) - len(predicted_clean)
    # target coverage
    # small target set
    covered = predicted_clean & targets
    if(len(targets) == 1):
      small_target_cnt += 1
      if(targets[0] in predicted_clean): target_coverage_small += 1.0
      else: target_coverage_small += 0.0
    # large target set
    else:
      large_target_cnt += 1
      target_coverage_large_partial += float(len(covered)) / len(targets)
      if(len(covered) == len(targets)): target_coverage_large_perfect += 1.0
      else target_coverage_large_perfect += 0.0
    # enrichment 
    enrichment += len(predicted_clean - covered)
    # memory
    mem_cover += len(predicted_clean) / (len(Set(mvb[i])) - 1)
  redundancy /= batch_size
  target_coverage_small = float(target_coverage_small) / small_target_cnt
  target_coverage_large_partial = float(target_coverage_large_partial) / large_target_cnt
  target_coverage_large_perfect = float(target_coverage_large_perfect) / large_target_cnt
  enrichment /= batch_size
  mem_cover /= batch_size
  print("batch %d, redundancy: %.2f" % (bi, redundancy))
  print("target coverage, small: %.2f, large-partial: %.2f, large-total: %.2f" % 
    (target_coverage_small, target_coverage_large_partial, target_coverage_large_perfect))
  print("enrichment: %.2f, mem_cover: %.2f" % (enrichment, mem_cover))
  return (redundancy, target_coverage_small, target_coverage_large_partial,
    target_coverage_large_perfect, enrichment, mem_cover)

def metrics(aoub, out_idx, mkb, mvb, dset, bi):
  aoub = aoub.T
  target_cnt = 0
  cover_cnt = 0
  target_cnt_large = 0
  cover_cnt_large = 0
  perfect_cover_cnt = 0
  cover_targets = []
  covered = []
  repeat_cnt = 0
  large_set_tags = []
  for i in range(len(aoub)):
    target_set = Set()
    predict_set = []
    for j in aoub[i]:
      if(j >= dset.norm_word_cnt and dset.mem_id2labels[mkb[i][j - dset.norm_word_cnt]] != "movie"):
        target_set.add(dset.id2word[mvb[i][j - dset.norm_word_cnt]])
    for j in out_idx[i]:
      if(j >= dset.norm_word_cnt and dset.mem_id2labels[mkb[i][j - dset.norm_word_cnt]] != "movie"):
        predict_set.append(dset.id2word[mvb[i][j - dset.norm_word_cnt]])
    predict_len = len(predict_set)
    predict_set = Set(predict_set)
    repeat_cnt += predict_len - len(predict_set)
    # print("%d words in target set" % len(target_set))
    # print(target_set)
    # print("%d words in predict set" % len(predict_set))
    # print(predict_set)
    target_cnt += len(target_set)
    cover_cnt += len(target_set & predict_set)
    if(len(target_set) == len(predict_set)): perfect_cover_cnt += 1
    if(len(target_set) >= 2):
      target_cnt_large += len(target_set)
      cover_cnt_large += len(target_set & predict_set)
      large_set_tags.append(1)
    else:
      large_set_tags.append(0)
    cover_targets.append(target_set)
    covered.append(predict_set)
  coverage = float(cover_cnt) / target_cnt
  coverage_large = float(cover_cnt_large) / target_cnt_large if target_cnt_large != 0 else 0
  coverage_perfect = float(perfect_cover_cnt) / len(aoub)
  # print("batch %d, coverage: %.2f, coverage for large set: %.2f" % (bi, coverage, coverage_large))
  return coverage, coverage_large, coverage_perfect, cover_targets, covered, repeat_cnt, large_set_tags

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

# for the first 3 cases in batch, for each of the output timestep, output the attention
def attn_visualize(qb, qlenb, mkb, mvb, mlenb, out_idx, dbg_attn_mkey, dbg_attn_mval, dbg_attn_q, dbg_attn_out, dset):
  def show_attn(output, distribution, mlen):
    color_out = []
    for i in range(mlen):
      if(distribution[i] < 0.1):
        color_out.append(fore.WHITE + back.GREY_0 + output[i] + style.RESET)
      elif(0.1 <= distribution[i] < 0.3):
        color_out.append(fore.WHITE + back.PINK_1 + output[i] + style.RESET)
      elif(0.3 <= distribution[i] < 0.5):
        color_out.append(fore.WHITE + back.DEEP_PINK_1A + output[i] + style.RESET)
      else:
        color_out.append(fore.WHITE + back.RED + output[i] + style.RESET)
    print("   ".join(color_out))
    return
  word2id = dset.word2id
  id2word = dset.id2word
  for ii in range(15, 20):
    swords = []
    for i in out_idx[ii]:
      if(i < dset.norm_word_cnt): swords.append(id2word[i])
      else:
        tmpw = "(loc %d -> " % (i - dset.norm_word_cnt)
        tmpw += id2word[mvb[ii][i - dset.norm_word_cnt]]
        tmpw += ")"
        swords.append(tmpw)
    print("   ".join(swords))
    for j in range(len(out_idx[ii])):
      if(id2word[out_idx[ii][j]] == "_EOS"):
        olen = j + 1
        break
    for j in range(len(out_idx[ii])):
      # print the output and underline the current timestep
      print("timestep %d" % j)
      print("out_idx: ")
      swords = []
      for jj, i in enumerate(out_idx[ii][: olen]):
        if(i < dset.norm_word_cnt): 
          tmpw = id2word[i]
          if(jj == j): tmpw = fore.WHITE + back.BLUE + tmpw + style.RESET
          swords.append(tmpw)
        else:
          tmpw = "(loc %d -> " % (i - dset.norm_word_cnt)
          tmpw += id2word[mvb[ii][i - dset.norm_word_cnt]]
          tmpw += ")"
          if(jj == j): tmpw = fore.WHITE + back.BLUE + tmpw + style.RESET
          swords.append(tmpw)
      print("   ".join(swords))

      # attention to key
      print("attention to key: (id, weight)")
      kprint = []
      for idx, val in enumerate(dbg_attn_mkey[ii][j][: mlenb[ii]]):
        kprint.append("(%d, %.5f)" % (idx, val))
      print("  ".join(kprint))
      # memory key
      kwords = []
      kpos = np.argmax(dbg_attn_mkey[ii][j])
      for jj, i in enumerate(mkb[ii][: mlenb[ii]]):
        tmpw = "(" + str(jj) + ", " + dset.mem_id2labels[i] +  ")"
        # if(jj == kpos): tmpw = fore.WHITE + back.BLUE + tmpw + style.RESET
        kwords.append(tmpw)
      show_attn(kwords, dbg_attn_mkey[ii][j], mlenb[ii])
      # print(" ".join(kwords))

      # attention to val
      print("attention to val: (id, weight)")
      # print(dbg_attn_mval[ii][j])
      vprint = []
      for idx, val in enumerate(dbg_attn_mval[ii][j][: mlenb[ii]]):
        vprint.append("(%d, %.5f)" % (idx, val))
      print("  ".join(vprint))
      # memory val
      vwords = []
      vpos = np.argmax(dbg_attn_mval[ii][j])
      for jj, i in enumerate(mvb[ii][: mlenb[ii]]):
        tmpw = "(" + str(jj) + ", " + id2word[i] +  ")"
        # if(jj == vpos): tmpw = fore.WHITE + back.BLUE + tmpw + style.RESET
        vwords.append(tmpw)
      show_attn(vwords, dbg_attn_mval[ii][j], mlenb[ii])
      # print(" ".join(vwords))
      # print([ (idx, dset.id2word[i]) for idx, i in enumerate(mvb[ii][: mlenb[ii]]) ])

      # attention to question 
      print("attention to question: (id, weight)")
      # print(dbg_attn_q[ii][j])
      qprint = []
      for idx, val in enumerate(dbg_attn_q[ii][j][: qlenb[ii]]):
        qprint.append("(%d, %.5f)" % (idx, val))
      print("  ".join(qprint))
      # question
      qwords = []
      qpos = np.argmax(dbg_attn_q[ii][j])
      for jj, i in enumerate(qb[ii][: qlenb[ii]]):
        tmpw = id2word[i]
        # if(jj == qpos): tmpw = fore.WHITE + back.BLUE + tmpw + style.RESET
        qwords.append(tmpw)
      show_attn(qwords, dbg_attn_q[ii][j], qlenb[ii])
      # print("   ".join(qwords))
      # print("   ".join([id2word[i] for i in qb[ii][: qlenb[ii]]]))

      # attention to output record
      print("attention to output record: (id, weight)")
      # print(dbg_attn_out[ii][j])
      oprint = []
      for idx, val in enumerate(dbg_attn_out[ii][j][: olen]):
        oprint.append("(%d, %.5f)" % (idx, val))
      print("  ".join(oprint))
      # output
      swords = []
      opos = np.argmax(dbg_attn_out[ii][j])
      for jj, i in enumerate(out_idx[ii][: olen]):
        if(i < dset.norm_word_cnt): 
          tmpw = id2word[i]
          # if(jj == opos): tmpw = fore.WHITE + back.BLUE + tmpw + style.RESET
          swords.append(tmpw)
        else:
          tmpw = "(loc %d -> " % (i - dset.norm_word_cnt)
          tmpw += id2word[mvb[ii][i - dset.norm_word_cnt]]
          tmpw += ")"
          # if(jj == opos): tmpw = fore.WHITE + back.BLUE + tmpw + style.RESET
          swords.append(tmpw)
      show_attn(swords, dbg_attn_out[ii][j], olen)
      # print("   ".join(swords))
      # print("   ".join(swords))
      print("\n\n")
      if(id2word[out_idx[ii][j]] == "_EOS"): break
  return


