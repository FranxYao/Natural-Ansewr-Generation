# coding: utf-8

# ---- Data Utilities for Full Version of FQA ----
# -- Francis F. Yao
# -- francis_yao@pku.edu.cn
# -- FRI 25TH AUG 2017

# Source of training set:
# 1 None conjunction golde# attention query
#   - answer sentences restricted to what is asked
#   - pattern generated
#   - sentence variety restricted to patterns
# 2 Conjunction golden: 
#   - answer sentences conjuncted with other related information
#   - pattern generated
#   - sentence variety restricted to patterns, but more than 1
# 3 Answer wiki sentences: 
#   - answer sentences from wiki
#   - more sentence variety than 2
# 4 Normal wiki sentences: 
#   - movie related sentences from wiki, for general purpose language model
#   - more sentence variety than 3

import numpy as np
import random as rd
from sets import Set
import re
from acora import AcoraBuilder
from collections import Counter
import cPickle
import time
import os

MAX_A_LEN = 20
MAX_M_LEN = 10
DATA_SOURCE = "ncon"

class Dataset(object):
  def __init__(self):
    # Intermediate data structure
    self.entities = None
    self.movie_names = None
    self.poly_movie_names = None # Some movie names are polyseme
    self.movie_sent = None
    self.movie_kb = None
    self.movie_canddt_ans = None # candidate answers
    self.movie_ans_conj = None   # conjuncted answer sentences 
    self.qapairs = dict()
    self.case_set_ncon = dict()
    self.case_set_conj = dict()
    self.case_set_wiki = dict()
    self.wiki_sent_set = []
    self.id_set_ncon = dict()
    self.id_set_conj = dict()
    self.id_set_wiki = dict()
    self.id_wik_sent =[]
    
    # data structure used 
    self.conj_ratio = 1.0
    self.wiki_ratio = 1.0
    self.word_count = None
    self.word2id = None
    self.id2word = None
    self.total_words = 0
    self.norm_word_cnt = 0
    self.spec_word_cnt = 0
    self.max_q_len = 11
    self.max_a_len = 45
    self.max_aw_len = 0
    self.max_kb_len = 0
    self.max_mem_len = 66
    self.mem_labels2id = dict()
    self.mem_id2labels = dict()
    self.id2qcls = None
    self.qcls2id = None
    self.acls2id = None
    self.id2acls = None
    self.batches = None
    self.batch_pointer = None
    self.total_cases_train = 0
    self.total_cases_valid = 0
    self.total_cases_test = 0

    # Auxiliary 
    self.directors = None
    self.writers = None
    self.actors = None
    self.release_years = None
    self.languages = None
    self.tags = None
    self.genres = None
    self.votes = None
    self.ratings = None
    self.record_special = None # the union of all the above
    self.person_names = None
    self.wiki_only_minor = None
    #
    self.pipeline()
    return 

  def read_entities(self):
    # Input: entities.txt
    print("reading entities ... ")
    with open("../data/entities.txt", "ro") as fd:
      lines = fd.readlines()
    entities = []
    for l in lines:
      entities.append(l.strip())
    # Output: entities
    print("%d entities in total" % len(entities))
    # Output
    self.entities = Set(entities)
    return 

  def read_wiki(self):
    # input: wiki.txt, entities
    print("reading wiki ... ")
    with open("../data/wiki.txt", "ro") as fd:
      lines = fd.readlines()
    movie_sent = dict()
    new_movie = True
    p = re.compile(" \(.*\)")
    movie_attr_set = Set()
    entities = self.entities
    for l in lines:
      if(new_movie == True):
        movie_name = l[2:-1]
        movie_attr = p.search(movie_name)
        movie_attr = movie_attr.group()[2:-1] if movie_attr is not None else None
        if(movie_attr is not None): movie_attr_set.add(movie_attr)
        movie_name = re.sub(p, "", movie_name)
        if(movie_name not in entities):
          print("movie name %s not in entities" % movie_name)
        else:
          movie_sent[movie_name] = []
          if((movie_attr is not None) and (movie_attr != "film")): 
            tmp_disp_sent = "%s is a %s" % ("It" if rd.random() < 0.5 else movie_name, movie_attr)
            movie_sent[movie_name].append(tmp_disp_sent)
            # print("adding disp sent: %s" % tmp_disp_sent)
        new_movie = False
      elif(l == "\n"): new_movie = True
      else:
        movie_sent[movie_name].append(l[2:-1])
    # with open("attr_from_wiki.txt", "w") as fd:
    #   for att in movie_attr_set:
    #     fd.write("%s\n" % att)
    # Output
    self.movie_sent = movie_sent
    self.movie_names = movie_sent.keys()
    print("%d movies in wiki" % len(movie_sent))
    return

  def read_kb(self):
    # Input: wiki_entities_kb.txt
    print("\nreading kb ... ")
    entities = self.entities
    relations = ["directed_by", "written_by", "starred_actors", 
      "release_year", "in_language", "has_tags", "has_plot", "has_genre", "has_imdb_votes", "has_imdb_rating"]
    movie_kb = dict()
    with open("../data/wiki_entities_kb.txt", "ro") as fd:
      lines = fd.readlines()
    new_movie = True
    multiple_names = Set()
    for l in lines:
      if(l != "\n"):
        l = l[2:-1]
        for r in relations:
          if r in l:
            movie_name, obj = l.split(" " + r + " ")
            if(new_movie == True and movie_name in movie_kb): multiple_names.add(movie_name)
            new_movie = False
            if(movie_name not in entities):
              print("movie name %s not in entities" % movie_name)
            if(movie_kb.get(movie_name) == None): movie_kb[movie_name] = []
            movie_kb[movie_name].append((r, obj))
      else: new_movie = True
    # Output: 
    self.movie_kb = movie_kb
    print("%d movies in kb" % len(movie_kb))
    print("%d of them are polysemes" % len(multiple_names))
    # with open("multiple_movie_name.txt", "w") as fd:
    #   for mv in multiple_names: fd.write("%s\n" % mv)
    self.poly_movie_names = list(multiple_names)
    return

  # Read candidate answer
  def read_canddt_ans(self, filename):
    entities = self.entities
    movie_canddt_ans = dict()
    with open(filename, "ro") as fd:
      lines = fd.readlines()
    movie_name = str()
    for l in lines:
      if(l == "\n"): pass
      elif(l[0] == "1" and l[1] == " "): # movie name
        movie_name = l[2:-1]
        if(movie_name not in entities): print("movie name %s not in entities" % movie_name)
        if(movie_name not in movie_canddt_ans): movie_canddt_ans[movie_name] = []
      else: # sentences
        if("was the year" in l): 
          cdd_obj = []
          lset = []
          for rel, obj in self.movie_kb[movie_name]: 
            if(rel == "release_year"):
              cdd_obj.append(obj) 
          for obj in cdd_obj: lset.append(l.replace("was the year" , obj + " was the year"))
          for li in lset: movie_canddt_ans[movie_name].append(li[2:-1])
        else: movie_canddt_ans[movie_name].append(l[2:-1])
    # Output: 
    return movie_canddt_ans

  def read_ans_sent(self):
    self.movie_canddt_ans = self.read_canddt_ans("../data/templates=all_conj=0.0_coref=0.0.txt")
    self.movie_ans_conj = self.read_canddt_ans("../data/templates=all_conj=0.5_coref=0.8.txt")
    return

  def read_questions(self):
    # -- Build AC Automata
    print("\nreading questions ...")
    print("building AC Automata")
    start_time = time.time()
    movie_with_comma = []
    total_movies = Set(self.movie_names)
    for m in total_movies:
      if("," in m): movie_with_comma.append(m)
    builder = AcoraBuilder()
    for m in movie_with_comma:
      builder.add(m)
    ac = builder.build()
    print("time expense: %.4f" % (time.time() - start_time))
    # Out: ac

    # -- Read qapairs and filter * to film questions
    ques_p = re.compile("(what|which) (is a )?(film|movie)")
    ques_p2 = re.compile("what does .* in")
    ques_p3 = re.compile("name a film|give a few words|words describe|the writer of")
    ques_p4 = re.compile("who (are the actors|stars|acted) in")
    ques_p5 = re.compile("the director of|the film .*starred")
    for sname in ["train", "dev", "test"]:
      to_film_cnt = 0
      total_cnt = 0
      with open("../data/wiki-entities_qa_%s.txt" % sname) as fd:
        lines = fd.readlines()
      qapairs = []
      for l in lines:
        total_cnt += 1
        q, a = [li.strip() for li in l.split("\t")]
        q = q[2:]
        q = q[:-1] + " ?" if q[-1] == "?" else q
        if(ac.findall(a) != []): 
          to_film_cnt += 1
          continue
        a_list = [ai.strip() for ai in a.split(",")]
        in_set_cnt = 0
        for ai in a_list:
          if(ai in total_movies): in_set_cnt += 1
        if(in_set_cnt >= 0.6 * len(a_list)): 
          to_film_cnt += 1
          continue
        if(ques_p.findall(q) != [] or ques_p2.findall(q) != [] or ques_p3.findall(q) != []
          or ques_p4.findall(q) != [] or ques_p5.findall(q) != []):
          to_film_cnt += 1
          continue
        qapairs.append([q, a])
      self.qapairs[sname] = qapairs
      print("%s set, %d total questions, %d questions ask about film" % (sname, total_cnt, to_film_cnt))
    # Output: self.qapairs
    print("questions ask about film are filtered")
    return

  def build_vocab(self):
    print("\nbuilding vocabulary ... ")
    # -- Build AC Automata
    print("building AC Automata")
    start_time = time.time()
    delims = [" ", ",", ".", '"', "?", "!" , "(", ")", ":", "'", ";", "/", "—", "’", "*", "[", "]"]
    ent_with_space = []
    for e in self.entities:
      if(e == "good movie"): continue
      if(e == "bad movie"): continue
      # if(" " in e): ent_with_space.append(e)
      for dl in delims:
        if(dl in e): ent_with_space.append(e)
    builder = AcoraBuilder()
    for e in ent_with_space:
      builder.add(e)
    ac = builder.build()
    print("time expense: %.4f" % (time.time() - start_time))

    # -- Sentence split algorithm
    def sent_split(s):
      # find special words
      if(s[-1] in [".", "?", "!"]): s = s[:-1] + " " + s[-1]
      spe_words_s = ac.findall(s)
      spe_words_s.sort(key = lambda x: x[1])
      # out: spe_words_s

      locs = []
      prevsl, prevel = 0, 0
      # process locations
      # detect and handle weird cases
      for w, sl in spe_words_s:
        # If one special word covers another, choose the longer one
        # e.g: Lon Chaney v.s Lon Chaney Jr.
        el = sl + len(w)
        if(sl < prevsl):
          print("weird case 0:")
          print(s)
          print(spe_words_s)
        if(sl >= prevsl and sl < prevel): 
          if(el <= prevel): continue # the first covers the second
          elif(sl == prevsl): #  and el > prevel
            # print("weird case E:")
            # print((prevsl, prevel)) # Bug here TBC ... 
            # print(s)
            # print(spe_words_s)
            locs.remove((prevsl, prevel)) # the second covers the first, abandon the first
            prevloc = locs[-1] if len(locs) >= 1 else (0, 0)
            prevsl, prevel = prevloc
          else: # two special words overlap, all abandon
            locs.remove((prevsl, prevel))
            prevloc = locs[-1] if len(locs) >= 1 else (0, 0)
            prevsl, prevel = prevloc
            continue
            # print("weird case A:")
            # print(s)
            # print(spe_words_s)
        if(sl > 0 and s[sl - 1] not in delims and re.search("\w|[-]", s[sl - 1]) != None):
          continue # the beginning of this special word is covered by normal 
        if(el < len(s) and s[el] not in delims and re.search("\w|[-]", s[el]) != None):
          continue # the end of this special word is covered by normal 
        locs.append((sl, el))
        prevsl = sl
        prevel = sl + len(w)
      # out: locs i.e. locations of special words

      def delims_add_space(s):
        for dl in delims: s = s.replace(dl, " " + dl + " ")
        return s
      segments = []
      prevel = 0
      # find segments
      # only detect weird cases, do not handle them
      for sl, el in locs:
        if((prevel > sl) or (sl > 0 and s[sl - 1] not in delims) 
          or (el < len(s) and s[el] not in delims)): 
          if(prevel > sl): print("B", prevel, sl)
          elif(sl > 0 and s[sl - 1] not in delims): 
            if(ord(s[sl - 1]) >= 128): s = s.replace(s[sl - 1], " ")
            elif(s[sl - 1] == "$"): continue
            else: 
              print("weird case:")
              print(s)
              print(spe_words_s)
              print("C", s[sl - 1], sl)
          else: 
            if(ord(s[el]) >= 128): s = s.replace(s[el], " ")
            else: 
              print("weird case:")
              print(s)
              print(spe_words_s)
              print("D", s[el], el)
        # common_seg = s[prevel:sl].replace(",", " ,").split()
        common_seg = delims_add_space(s[prevel:sl]).split()
        special_seg = s[sl:el]
        segments.extend(common_seg)
        segments.append(special_seg)
        prevel = el
      # common_seg = s[prevel:].replace(",", " ,").split()
      common_seg = delims_add_space(s[prevel:]).split()
      segments.extend(common_seg)
      return segments

    # -- Process sentences
    print("\nbuilding vocabulary, start to process sentences ... ")
    start_time = time.time()
    all_words = []
    # start to process sentences
    new_sent = dict()
    new_canddt_ans = dict()
    new_ans_conj = dict()
    print("movie sent:")
    tmpi = 0
    for m in self.movie_sent:
      new_sent[m] = []
      for s in self.movie_sent[m]:
        tmp_words = sent_split(s)
        new_sent[m].append(tmp_words)
        all_words.extend(tmp_words)
      tmpi += 1
      # if(tmpi == 10): break
    print("movie_canddt_ans:")
    tmpi = 0
    for m in self.movie_canddt_ans:
      new_canddt_ans[m] = []
      for s in self.movie_canddt_ans[m]:
        tmp_words = sent_split(s)
        new_canddt_ans[m].append(tmp_words)
        all_words.extend(tmp_words)
      tmpi += 1
      # if(tmpi == 10): break
    tmpi = 0
    print("movie_ans_conj")
    for m in self.movie_ans_conj:
      new_ans_conj[m] = []
      for s in self.movie_ans_conj[m]:
        tmp_words = sent_split(s)
        new_ans_conj[m].append(tmp_words)
        all_words.extend(tmp_words)
      tmpi += 1
      # if(tmpi == 10): break
    # Output: 
    self.movie_sent = new_sent
    self.movie_canddt_ans = new_canddt_ans
    self.movie_ans_conj = new_ans_conj
    print("time expense: %.4f" % (time.time() - start_time))

    # -- Process kb
    # Note: plot is abandoned
    directors = []
    writers = []
    actors = []
    release_years= []
    languages = []
    tags = []
    genres = []
    votes = []
    ratings = []
    new_kb = dict()
    print("\nbuilding vocabulary, start to process kb ... ")
    start_time = time.time()
    # ti = 1
    for m in self.movie_kb:
      new_kb[m] = []
      for r, obj in self.movie_kb[m]:
        if(r == "directed_by"):
          tmpset = Set([d.strip() for d in sent_split(obj)])
          # if("," in tmpset): tmpset.remove(",")
          tmpset = tmpset - Set(delims)
          tmp_dircts = list(tmpset)
          # if(ti <= 10): 
          #   print tmp_dircts
          #   ti += 1
          directors.extend(tmp_dircts)
          new_kb[m].append((r, tmp_dircts))
          all_words.extend(tmp_dircts)
        elif(r == "written_by"):
          tmpset = Set([w.strip() for w in sent_split(obj)])
          # if("," in tmpset): tmpset.remove(",")
          tmpset = tmpset - Set(delims)
          tmp_writers = list(tmpset)
          # tmp_writers = [w.strip() for w in sent_split(obj)]
          writers.extend(tmp_writers)
          new_kb[m].append((r, tmp_writers))
          all_words.extend(tmp_writers)
        elif(r == "starred_actors"):
          tmpset = Set([a.strip() for a in sent_split(obj)])
          # if("," in tmpset): tmpset.remove(",")
          tmpset = tmpset - Set(delims)
          tmp_actors = list(tmpset)
          # tmp_actors = [a.strip() for a in sent_split(obj)]
          actors.extend(tmp_actors)
          new_kb[m].append((r, tmp_actors))
          all_words.extend(tmp_actors)
        elif(r == "release_year"):
          tmpset = Set([y.strip() for y in sent_split(obj)])
          # if("," in tmpset): tmpset.remove(",")
          tmpset = tmpset - Set(delims)
          tmp_years = list(tmpset)
          # tmp_years = [y.strip() for y in sent_split(obj)]
          if(len(tmp_years) > 1): print("movie %s has multiple release year" % m)
          release_years += tmp_years
          new_kb[m].append((r, tmp_years))
          all_words.extend(tmp_years)
        elif(r == "in_language"):
          tmpset = Set([l.strip() for l in sent_split(obj)])
          # if("," in tmpset): tmpset.remove(",")
          tmpset = tmpset - Set(delims)
          tmp_langs = list(tmpset)
          # tmp_langs = [l.strip() for l in sent_split(obj)]
          languages += tmp_langs
          new_kb[m].append((r, tmp_langs))
          all_words.extend(tmp_langs)
        elif(r == "has_tags"):
          tmpset = Set([t.strip() for t in sent_split(obj)])
          # if("," in tmpset): tmpset.remove(",")
          tmpset = tmpset - Set(delims)
          tmp_tags = list(tmpset)
          # tmp_tags = [t.strip() for t in sent_split(obj)]
          tags += tmp_tags
          new_kb[m].append((r, tmp_tags))
          all_words.extend(tmp_tags)
        elif(r == "has_genre"):
          tmpset = Set([g.strip() for g in sent_split(obj)])
          # if("," in tmpset): tmpset.remove(",")
          tmpset = tmpset - Set(delims)
          tmp_genres = list(tmpset)
          # tmp_genres = [g.strip() for g in sent_split(obj)]
          genres += tmp_genres
          new_kb[m].append((r, tmp_genres))
          all_words.extend(tmp_genres)
        elif(r == "has_imdb_votes"):
          tmpset = Set([v.strip() for v in sent_split(obj)])
          # if("," in tmpset): tmpset.remove(",")
          tmpset = tmpset - Set(delims)
          tmp_votes = list(tmpset)
          # tmp_votes = [v.strip() for v in sent_split(obj)]
          votes += tmp_votes
          new_kb[m].append((r, tmp_votes))
          all_words.extend(tmp_votes)
        elif(r == "has_imdb_rating"): # "has_imdb_rating"
          tmpset = Set([rt.strip() for rt in sent_split(obj)])
          # if("," in tmpset): tmpset.remove(",")
          tmpset = tmpset - Set(delims)
          tmp_rating = list(tmpset)
          # tmp_rating = [r.strip() for r in sent_split(obj)]
          ratings += tmp_rating
          new_kb[m].append((r, tmp_rating))
          all_words.extend(tmp_rating)
        else: pass # abandon plot
    # Output: directors, writers, actors, release_years, languages, tags, genres, votes, and rating
    print("time expense: %.4f" % (time.time() - start_time))
    self.directors = Counter(directors)
    print("%d directors" % len(self.directors))
    self.writers = Counter(writers)
    print("%d writers" % len(self.writers))
    self.actors = Counter(actors)
    print("%d actors" % len(self.actors))
    self.release_years = Counter(release_years)
    print("%d release_years" % len(self.release_years))
    self.languages = Counter(languages)
    print("%d languages" % len(self.languages))
    self.tags = Counter(tags)
    print("%d tags" % len(self.tags))
    self.genres = Counter(genres)
    print("%d genres" % len(self.genres))
    self.votes = Counter(votes)
    print("%d votes" % len(self.votes))
    self.ratings = Counter(ratings)
    print("%d ratings" % len(self.ratings))
    total_special = len(self.directors) + len(self.writers) + len(self.actors) + len(self.release_years)
    total_special += len(self.languages) + len(self.tags) + len(self.genres) + len(self.votes) + len(self.ratings)
    total_special += len(self.movie_names)
    self.record_special = self.directors.keys() + self.writers.keys() + self.actors.keys() + self.release_years.keys()
    self.record_special += self.languages.keys() + self.tags.keys() + self.genres.keys() + self.votes.keys()
    self.record_special += self.ratings.keys() + self.movie_names
    print("%d special words in total ... but %d entities, why?" % (total_special, len(self.entities)))
    # Output: new_kb
    self.movie_kb = new_kb

    # -- Process questions
    print("\nbuilding vocabulary, start to process questions ... ")
    # 1. find answer set 
    self.q_find_a_sent()

    # 2. tokenize questions
    for sname in ["train", "test", "dev"]:
      for case in self.case_set_ncon[sname]: 
        case["q"] = sent_split(case["q"])
        all_words.extend(case["q"])
      for case in self.case_set_conj[sname]: 
        case["q"] = sent_split(case["q"])
        all_words.extend(case["q"])
      for case in self.case_set_wiki[sname]: 
        case["q"] = sent_split(case["q"])
        all_words.extend(case["q"])
    # 3. split quesiton into q-a pairs
    self.extend_questions()

    # 4. enrich candidate answer words
    self.enrich_wiki_conj_ans()

    # 5. add kb triples 
    self.ans_add_kb()

    # Output: all_words
    all_words = Counter(all_words)
    self.word_count = all_words
    return 

  def q_find_a_sent(self):
    # -- Build AC Automata for finding movie names
    print("building AC Automata ... ")
    start_time = time.time()
    total_movies = self.movie_names
    builder = AcoraBuilder()
    for m in total_movies:
      builder.add(m)
    ac = builder.build()
    print("time expense: %.4f" % (time.time() - start_time))

    def q_classification(q):
      if(" acted" in q or " actor" in q or " star" in q): return "actor"
      elif(" direct" in q): return "director"
      elif(" writ" in q or "wrot" in q or "author" in q or "script" in q): return "writer"
      elif("release" in q): return "release_year"
      elif("topic" in q or "describ" in q or " term" in q): return "tag"
      elif("type" in q or "genre" in q or "kind" in q or "sort" in q): return "genre"
      elif("language" in q): return "language"
      elif("how popular" in q or "famous" in q): return "vote"
      elif("think of" in q or "rate" in q or "rating" in q or "consider" in q or "any good" in q or "popular opinion"): return "rating"
      else: return "unknown"

    # -- Find answer for questions
    case_set_ncon = dict()
    case_set_conj = dict()
    case_set_wiki = dict()
    for sname in ["train", "test", "dev"]:
      case_set_ncon[sname] = []
      case_set_conj[sname] = []
      case_set_wiki[sname] = []
      no_movie_name = 0
      many_answer_cases = 0
      many_answer_act = 0
      many_answer_dir = 0
      many_answer_wri = 0
      many_answer_oth = 0
      no_answer_case = 0
      not_cover_nconj = 0
      not_cover_conj = 0
      not_cover_wiki = 0
      for q, a in self.qapairs[sname]:
        # -- Find answer in no conjunction set
        movie_name = ac.findall(q)
        a_set = [ai.strip() for ai in a.split(",")]
        q_class = q_classification(q)
        if(q_class == "unknown"): print("an unknown question class", q)
        if(len(movie_name) == 0): # cannot find movie name in q, or too many movie names
          print("weird question cannot find movie name:", q, movie_name)  
          movie_name = None
        elif(len(movie_name) >= 2):
          mvs = [mv[0] for mv in movie_name]
          lmv = mvs[0]
          for mv in mvs: lmv = mv if(len(mv) > len(lmv)) else lmv
          all_in_one = True
          for mv in mvs: all_in_one = False if(mv in lmv == False) else all_in_one
          movie_name = lmv if(all_in_one == True) else None
        else:
          movie_name = movie_name[0][0]
          # Note: since no answer word has "," within, we can split the answers using ","
        if(movie_name != None):
          case = dict()
          case["q"] = q
          case["m"] = movie_name
          case["aw"] = list(a_set)
          case["a"] = []
          case["c"] = q_class
          for s in self.movie_canddt_ans[movie_name]:
            ans_in_set = 0
            for ai in a_set:
              if ai in s:
                ans_in_set += 1
            if((len(a_set) == 1 and ans_in_set == 1) or (len(a_set) >= 2 and ans_in_set >= 2)):
              if(len(a_set) != ans_in_set): not_cover_nconj += 1
              case["a"].append(s)
          if(len(case["a"]) == 0):
            print("weird no answer: ", q, a, case["a"])
            no_answer_case += 1
          elif(len(case["a"]) >= 2): 
            # Reasons for many answer: 
            # 1. director/ actor/ writer ambiguation -- use question class to choose the right one 
            # 2. multiple ways to answer indeed -- accept all of them 
            # 3. noise in plot -- accept, but count it 
            # question classification:
            aact = []
            adir = []
            awri = []
            aoth = []
            for ai in case["a"]:
              astr = " ".join(ai)
              if(" actor" in astr or " acted" in astr or " star" in astr): aact.append(ai)
              elif(" direct" in astr): adir.append(ai)
              elif(" writ" in astr or " wrot" in astr or " author" in astr or " script" in astr): awri.append(ai)
              else: aoth.append(ai)
            # if("acted" in q or "actor" in q or "star" in q): 
            if(q_class == "actor"):
              if(len(aact) != 0): 
                case["a"] = aact
                case_set_ncon[sname].append(case)
              else: 
                print("q %s, class %s, cannot find answer set" % (q, q_class))
                print("act:", aact)
                print("dir:", adir)
                print("wri:", awri)
                print("oth:", aoth)
                many_answer_act += 1
            # elif("direct" in q): 
            elif(q_class == "director"):
              if(len(adir) != 0): 
                case["a"] = adir
                case_set_ncon[sname].append(case)
              else:
                print("q %s, class %s, cannot find answer set" % (q, q_class))
                print("act:", aact)
                print("dir:", adir)
                print("wri:", awri)
                print("oth:", aoth)
                many_answer_dir += 1
            # elif("writ" in q or "wrot" in q or "author" in q or "script" in q): 
            elif(q_class == "writer"):
              if(len(awri) != 0): 
                case["a"] = awri
                case_set_ncon[sname].append(case)
              else:
                print("q %s, class %s, cannot find answer set" % (q, q_class))
                print("act:", aact)
                print("dir:", adir)
                print("wri:", awri)
                print("oth:", aoth)
                many_answer_wri += 1
            # elif("release" in q): # release year, do not need to filter answer 
            elif(q_class == "release_year"):
              case_set_ncon[sname].append(case)
            # elif("topic" in q or "describ" in q or "term" in q): # tag, do not need to filter answer 
            elif(q_class == "tag"):
              case_set_ncon[sname].append(case)
            # elif("type" in q or "genre" in q or "kind" in q or "sort" in q): # genre, do not need to filter answer 
            elif(q_class == "genre"):
              case_set_ncon[sname].append(case)
            # elif("language" in q): # language, do not need to filter answer 
            elif(q_class == "language"):
              case_set_ncon[sname].append(case)
            # elif("popular" in q or "famous" in q): # vote, do not need to filter answer 
            elif(q_class == "vote"):
              case_set_ncon[sname].append(case)
            # elif("think of" in q or "rate" in q or "rating" in q or "consider" in q): # rating, do not need to filter answer 
            elif(q_class =="rating"):
              case_set_ncon[sname].append(case)
              # if("good" in case.ans_word): print(case.q, case.ans_word, case["a"])
            else: 
              print("question cannot find class: ", q)
              print("a_set:", a_set)
              print("case.a", case["a"])
              many_answer_oth += 1
            # print("weird many answers: ", q, a, case["a"])
            many_answer_cases += 1
          else: case_set_ncon[sname].append(case)
        else:
          no_movie_name += 1
        # Output: case_set_ncon[sname]

        # -- Find answer in conjunction set -- TBC -- GOOD
        case = dict()
        case["q"] = q
        case["m"] = movie_name
        case["aw"] = list(a_set)
        case["a"] = []
        case["c"] = q_class
        for s in self.movie_ans_conj[movie_name]:
          ans_in_set = 0
          for ai in a_set:
            if ai in s:
              ans_in_set += 1
          if((len(a_set) == 1 and ans_in_set == 1) or (len(a_set) >= 2 and ans_in_set >= 2)):
            if(len(a_set) != ans_in_set): not_cover_conj += 1
            case["a"].append(s)
        if(len(case["a"]) != 0): case_set_conj[sname].append(case)
        # Output: case_set_conj

        # -- Find answer in raw wiki sentences -- TBC -- GOOD
        case = dict()
        case["q"] = q
        case["m"] = movie_name
        case["aw"] = list(a_set)
        case["a"] = []
        case["c"] = q_class
        for s in self.movie_sent[movie_name]:
          ans_in_set = 0
          for ai in a_set:
            if ai in s:
              ans_in_set += 1
          if((len(a_set) == 1 and ans_in_set == 1) or (len(a_set) >= 2 and ans_in_set >= 2)):
            if(len(a_set) != ans_in_set): not_cover_wiki += 1
            case["a"].append(s)
        if(len(case["a"]) != 0): case_set_wiki[sname].append(case)
        # Output: case_set_wiki
      print("%s set, %d questions pairs in non_conjunction set " % (sname, len(case_set_ncon[sname])))
      print("%d no movie name, %d no answer(abandoned), %d multiple answers(resolved)" % 
        (no_movie_name, no_answer_case, many_answer_cases))
      print("%d act, %d dir, %d wri, %d oth" % (many_answer_act, many_answer_dir, many_answer_wri, many_answer_oth))
      print("...")
      print("%d cases in conjunction set" % len(case_set_conj[sname]))
      print("...")
      print("%d cases in wiki set" % len(case_set_wiki[sname]))
      print("")
    print("cases that answer not fully covered in answer: nconj %d, conj %d, wiki %d" % 
      (not_cover_nconj, not_cover_conj, not_cover_wiki))
    print("Note: cases from conjunction and wiki will not be used")

    # -- Randomly shuffle and abandon some cases in conj and wiki 
    print("shuffling the cases ... ")
    for sname in ["train", "test", "dev"]:
      rd.shuffle(case_set_ncon[sname])
      self.case_set_ncon[sname] = case_set_ncon[sname]
      rd.shuffle(case_set_conj[sname])
      tmp_len = len(case_set_conj[sname])
      self.case_set_conj[sname] = case_set_conj[sname][: int(tmp_len * self.conj_ratio)]
      rd.shuffle(case_set_wiki[sname])
      tmp_len = len(case_set_wiki[sname])
      self.case_set_wiki[sname] = case_set_wiki[sname][: int(tmp_len * self.wiki_ratio)]
    return

  def extend_questions(self):
    print("\nextending questions ... ")
    new_set_ncon = dict()
    new_set_conj = dict()
    new_set_wiki = dict()
    for sname in ["train", "test", "dev"]:
      new_set_ncon[sname] = []
      new_set_conj[sname] = []
      new_set_wiki[sname] = []
      for case in self.case_set_ncon[sname]:
        if(len(case["a"]) == 1): 
          newcase = dict(case)
          newcase["a"] = newcase["a"][0]
          new_set_ncon[sname].append(newcase)
        else:
          for ai in case["a"]:
            newcase = {"q": case["q"], "aw": case["aw"], "m": case["m"], "c": case["c"], "a": ai}
            new_set_ncon[sname].append(newcase)
      for case in self.case_set_conj[sname]:
        if(len(case["a"]) == 1): 
          newcase = dict(case)
          newcase["a"] = newcase["a"][0]
          new_set_conj[sname].append(newcase)
        else:
          for ai in case["a"]:
            newcase = {"q": case["q"], "aw": case["aw"], "m": case["m"], "c": case["c"], "a": ai}
            new_set_conj[sname].append(newcase)
      for case in self.case_set_wiki[sname]:
        if(len(case["a"]) == 1): 
          newcase = dict(case)
          newcase["a"] = newcase["a"][0]
          new_set_wiki[sname].append(newcase)
        else:
          for ai in case["a"]:
            newcase = {"q": case["q"], "aw": case["aw"], "m": case["m"], "c": case["c"], "a": ai}
            new_set_wiki[sname].append(newcase)
      # Output:
      rd.shuffle(new_set_ncon[sname])
      self.case_set_ncon[sname] = new_set_ncon[sname]
      rd.shuffle(new_set_conj[sname])
      self.case_set_conj[sname] = new_set_conj[sname]
      rd.shuffle(new_set_wiki[sname])
      self.case_set_wiki[sname] = new_set_wiki[sname]
    return 

  def enrich_wiki_conj_ans(self):
    record_special = Set(self.record_special) - Set(self.tags)
    tmpi = 0
    enriched = []
    for sname in ["train", "test", "dev"]:
      for cset in [self.case_set_ncon, self.case_set_conj, self.case_set_wiki]:
        for case in cset[sname]:
          # print(case["aw"], case["m"])
          tmp_wset = Set(case["aw"] + [case["m"]])
          for a in case["a"]:
            if((a in record_special) and (a not in tmp_wset)):
              case["aw"].append(a)
              enriched.append(a)
      # for case in self.case_set_conj[sname]:
      #   tmp_wset = Set(case["aw"] + [case["m"]])
      #   for a in case["a"]:
      #     if((a in record_special) and (a not in tmp_wset)):
      #       case["aw"].append(a)
    enriched = Counter(enriched)
    # with open("enriched", "w") as fd:
    #   for e, i in enriched.most_common(): fd.write("%s: %d\n" % (e, i))
    return 

  # -- Add kb entries to dataset 
  def ans_add_kb(self):
    rel_map = { "actor": "starred_actors", "director": "directed_by", "writer": "written_by", "release_year": 
                "release_year", "language": "in_language", "tag": "has_tags", "genre": "has_genre", 
                "vote": "has_imdb_votes", "rating": "has_imdb_rating"}
    for sname in ["train", "test", "dev"]:
      for cset, cname in [(self.case_set_ncon, "ncon"), (self.case_set_conj, "conj"), (self.case_set_wiki, "wiki")]:
        print("\nset %s/ %s" % (sname, cname))
        no_kb = 0
        case_abandon = []
        for case in cset[sname]:
          case["kbe"] = None
          for rel, obj in self.movie_kb[case["m"]]:
            if(rel_map[case["c"]] == rel):
              tmpkbe = []
              for obji in obj: tmpkbe.append((rel, obji))
              case["kbe"] = tmpkbe
              break
          if(case["kbe"] == None):
            if(rel_map[case["c"]] == "has_imdb_rating"):
              if(case["aw"][0] in self.ratings): case["kbe"] = [(rel, case["aw"][0])]
              else: 
                # print("\nA: ")
                # print("question %s cannot find a kb entry" % case["q"])
                # print("class: %s" % case["c"])
                # print(case["aw"])
                case_abandon.append(case) 
                no_kb += 1
            elif(rel_map[case["c"]] == "has_imdb_votes"):
              if(case["aw"][0] in self.votes): case["kbe"] = [(rel, case["aw"][0])]
              # if(len(case["aw"]) == 1): case["kbe"] = (rel, case["aw"][0])
              else:
                print("\nB:")
                print("question %s cannot find a kb entry" % case["q"])
                print("class: %s" % case["c"])
                print(case["aw"])
                no_kb += 1
            else:
              print("\nC:")
              print("question %s cannot find a kb entry" % case["q"])
              print("class: %s - %s" % (case["c"], rel_map[case["c"]]))
              print(self.movie_kb[case["m"]])
              no_kb += 1
        print("%d cases abandoned" % len(case_abandon))
        for case in case_abandon:
          cset[sname].remove(case)
        print("%d questions cannot find a kb entry" % no_kb)  
    return 

  # Transform all words to id
  def all_to_id(self):
    print("\nTransforming all words to id ... ")
    # -- Count word2id
    self.count_person_name()
    self.word_set_diff()
    word2id = {"_PAD": 0, "_UNK": 1, "_GOO": 2, "_EOS": 3}
    id2word = dict()
    for w in word2id: id2word[word2id[w]] = w

    # -- Split the vocabulary
    i = len(word2id)
    for w in self.word_count:
      if(w not in self.person_names and w not in self.wiki_only_minor):
        word2id[w] = i
        id2word[i] = w
        i += 1
    norm_word_cnt = i
    for w in self.word_count:
      if(w in self.person_names):
        word2id[w] = i
        id2word[i] = w
        i += 1
    spec_word_cnt = i - norm_word_cnt
    print("%d normal words, %d special words" % (norm_word_cnt, spec_word_cnt))
    # Output: word2id, id2word
    # Note: structure of word2id: [ PAD-EOS | normal_word | person_names ] (wiki_only_minor is omitted)
    self.word2id = word2id
    self.id2word = id2word
    self.total_words = i
    self.norm_word_cnt = norm_word_cnt
    self.spec_word_cnt = spec_word_cnt

    # -- Transform words to id
    all_sets = []
    new_sets = []
    for sname in ["train", "dev", "test"]:
      for aset in [self.case_set_ncon, self.case_set_conj, self.case_set_wiki]: all_sets.append(aset[sname])
    all_sets.append(self.wiki_sent_set)
    for st in all_sets:
      nst = []
      for case in st:
        newcase = dict()
        newcase["q"] = [word2id[w] for w in case["q"]]
        if(case["m"] not in word2id): 
          print case
          continue
        newcase["m"] = word2id[case["m"]]
        newcase["c"] = case["c"]
        newcase["a"] = [word2id[w] if w in word2id else word2id["_UNK"] for w in case["a"]]
        newcase["aw"] = [word2id[w] for w in case["aw"]]
        newcase["kbe"] = [(rel, word2id[obj]) for rel, obj in case["kbe"]]
        nst.append(newcase)
      new_sets.append(nst)
    # Output:
    for i, sname in enumerate(["train", "dev", "test"]):
      self.id_set_ncon[sname] = new_sets[3 * i]
      self.id_set_conj[sname] = new_sets[3 * i + 1]
      self.id_set_wiki[sname] = new_sets[3 * i + 2]
    self.id_wik_sent = new_sets[-1]
    return 

  def format_wiki(self):
    # -- format sentences from wiki to case 
    print("\nformatting wiki sentences ... ")
    record_special = Set(self.record_special)
    wiki_sent_set = []
    for m in self.movie_sent:
      for s in self.movie_sent[m]:
        case = {"q": [], "m": m, "c": "wiki", "a": s, "aw": [], "kbe": []}
        tmpaw = Set()
        tmpkbe = Set()
        for w in case["a"]:
          for rel, obj in self.movie_kb[m]:
            for obji in obj:
              if(w == obji):
                tmpaw.add(w)
                tmpkbe.add((rel, obji))
          if(w in record_special):
            tmpaw.add(w)
        if(len(tmpaw)!= 0):
          case["aw"] = list(tmpaw)
          case["kbe"] = list(tmpkbe)
        wiki_sent_set.append(case)
    # Output: wiki_sent_set
    print("%d cases in wiki sentences" % len(wiki_sent_set))
    self.wiki_sent_set = wiki_sent_set
    return 

  # delete long sentences and rebuild the dictionary 
  def delete_long_sent(self): 
    # -- Sentence length statistics
    print("\ndeleting cases with sentence too long ... ")
    all_sets = []
    new_sets = []
    new_word = []
    for sname in ["train", "dev", "test"]:
      for aset in [self.case_set_ncon, self.case_set_conj, self.case_set_wiki]: all_sets.append(aset[sname])
    all_sets.append(self.wiki_sent_set)
    q_len_set = []
    a_len_set = []
    aw_len_set = []
    kb_len_set = []
    sent_deleted = 0
    for st in all_sets:
      q_len = []
      a_len = []
      aw_len = []
      kb_len = []
      ns = []
      for case in st:
        if(len(case["a"]) >= 45): 
          sent_deleted += 1
          continue
        q_len.append(len(case["q"]))
        a_len.append(len(case["a"]))
        aw_len.append(len(case["aw"]))
        kb_len.append(len(case["kbe"]))
        new_word.extend(case["q"])
        new_word.extend(case["a"])
        new_word.extend(case["aw"])
        new_word.append(case["c"])
        new_word.append(case["m"])
        new_word.extend([obj for rel, obj in case["kbe"]])
        ns.append(case)
      q_len_set.append(q_len)
      a_len_set.append(a_len)
      aw_len_set.append(aw_len)
      kb_len_set.append(kb_len)
      new_sets.append(ns)
    # Output: new case set
    for i, sname in enumerate(["train", "dev", "test"]):
      self.case_set_ncon[sname] = new_sets[3 * i]
      self.case_set_conj[sname] = new_sets[3 * i + 1]
      self.case_set_wiki[sname] = new_sets[3 * i + 2]
    self.wiki_sent_set = new_sets[-1]
    print("%d sentences deleted" % sent_deleted)
    print("train[ncon, conj, wiki], dev[ncon, conj, wiki], test[ncon, conj, wiki], wikisent")
    print("max q len in different sets: ")
    print([np.max(s) for s in q_len_set])
    print("max a len in different sets: ")
    print([np.max(s) for s in a_len_set])
    print("max aw len in different sets: ")
    print([np.max(s) for s in aw_len_set])
    print("max kb len in different sets: ")
    print([np.max(s) for s in kb_len_set])
    self.max_aw_len = np.max([np.max(s) for s in aw_len_set])
    self.max_kb_len = np.max([np.max(s) for s in kb_len_set])
    # Output: new vocabulary
    new_word = Counter(new_word)
    print("\nprevious %d words, %d remained" % (len(self.word_count), len(new_word)))
    self.word_count = new_word
    return

  def format_all(self):
    print("\nFormatting all cases ... ")
    start_time = time.time()
  
    # -- Preparation
    # prepare quesion class 
    question_class = {"director": 0, "writer": 1, "actor": 2, "release_year": 3, 
      "language": 4, "tag": 5, "genre": 6, "vote": 7, "rating": 8, "wiki": 9}
    self.qcls2id = question_class
    self.id2qcls = dict()
    for ql in self.qcls2id: self.id2qcls[self.qcls2id[ql]] = ql
    # prepare answer class
    ans_class = {"ncon": 0, "conj": 1, "wiki": 2}
    self.acls2id = ans_class
    self.id2acls = dict()
    for al in self.acls2id: self.id2acls[self.acls2id[al]] = al
    # prepare word
    word2id = self.word2id
    id2word = self.id2word
    # prepare relation 
    rels = ["directed_by", "written_by", "starred_actors", "release_year", "in_language", 
            "has_tags", "has_genre", "has_imdb_votes", "has_imdb_rating"]
    mem_labels = {"_pad":0, "movie": 1}
    i = 2
    for r in rels:
      mem_labels[r] = i
      i += 1
    for ql in question_class:
      mem_labels["ans_" + ql] = i
      i += 1
    self.mem_labels2id = mem_labels
    for lb in mem_labels: self.mem_id2labels[mem_labels[lb]] = lb

    ans_word_set = { "director": self.directors, 
                     "writer": self.writers,
                     "actor": self.actors,
                     "release_year": self.release_years, 
                     "language": self.languages,
                     "tag": self.tags,
                     "genre": self.genres, 
                     "vote": self.votes,
                     "rating": self.ratings }
    self.one_ans_cls = 0
    self.no_ans_cls = 0
    self.many_ans_cls_q = 0
    self.many_ans_cls_n = 0
    def find_ans_class(ansid, qcls):
      answ = id2word[ansid]
      cls_set = []
      for ac in ans_word_set:
        if(answ in ans_word_set[ac]): cls_set.append(ac)
      if(len(cls_set) == 1): 
        self.one_ans_cls += 1
        return "ans_" + cls_set[0]
      elif(len(cls_set) == 0): 
        self.no_ans_cls += 1
        return "ans_wiki"
      else:
        if(qcls in cls_set): 
          self.many_ans_cls_q += 1
          return "ans_" + qcls
        else: 
          self.many_ans_cls_n += 1
          return "ans_" + cls_set[0]
      return 

    # -- Pad sentences and Transform to NN input format 
    all_sets = []
    for sname in ["train", "dev", "test"]:
      for aset in [self.id_set_ncon, self.id_set_conj, self.id_set_wiki]: all_sets.append(aset[sname])
    all_sets.append(self.id_wik_sent)
    not_in_mem = 0
    spec_word = []
    for st in all_sets:
      for case in st:
        # question
        new_case = dict(case)
        new_case["qlen"] = len(new_case["q"])
        while(len(new_case["q"]) < self.max_q_len): new_case["q"].append(word2id["_PAD"])
        # question class
        new_case["c"] = question_class[new_case["c"]]
        # answer
        new_case["alen"] = len(new_case["a"]) + 1
        while(len(new_case["a"]) < self.max_a_len): new_case["a"].append(word2id["_EOS"])
        new_case["aou"] = new_case["a"]
        new_case["ain"] = list([word2id["_GOO"]] + new_case["a"][:-1])
        # del new_case["a"]
        # memory 
        mem_set = [(mem_labels["movie"], new_case["m"])]
        for awi in new_case["aw"]: 
          # find out answer word class
          tmp_ans_cls = find_ans_class(awi, self.id2qcls[new_case["c"]])
          mem_set.append((mem_labels[tmp_ans_cls], awi))
        for rel, obj in new_case["kbe"]: mem_set.append((mem_labels[rel], obj))
        rd.shuffle(mem_set)
        # link answer word to memory 
        for awi in range(self.max_a_len):
          aw = new_case["aou"][awi]
          if(aw in [mi[1] for mi in mem_set]): # a word in memory 
            # find related memory entry
            rel_mem = []
            for loci, mi in enumerate(mem_set):
              if(mi[1] == aw): rel_mem.append(loci)
            # randomly choose one location
            if(len(rel_mem) != 0): new_case["aou"][awi] = self.norm_word_cnt + rel_mem[rd.randint(0, len(rel_mem) - 1)]
            else:
              print("\nweird cases cannot find answer!")
          else:
            if(aw >= self.norm_word_cnt): # a word not in memory and not in normal, put in memory 
              not_in_mem += 1
              mem_set.append((mem_labels["ans_wiki"], aw))
              new_case["aou"][awi] = self.norm_word_cnt + len(mem_set) - 1
              spec_word.append(id2word[aw])
            else: pass
        new_case["mm"] = mem_set
        new_case["mlen"] = len(mem_set)
        case["q"] = new_case["q"]
        case["qlen"] = new_case["qlen"]
        case["ain"] = new_case["ain"]
        case["aou"] = new_case["aou"]
        case["alen"] = new_case["alen"]
        case["c"] = new_case["c"]
        case["mm"] = new_case["mm"]
        while(len(case["mm"]) < self.max_mem_len): case["mm"].append((mem_labels["_pad"], word2id["_PAD"]))
        case["mk"] = []
        case["mv"] = []
        for k, v in case["mm"]:
          case["mk"].append(k)
          case["mv"].append(v)
        case["mlen"] = new_case["mlen"]
        del case["a"]
        del case["kbe"]
        del case["aw"]
        del case["mm"]
    print("%d words not in memory!" % not_in_mem)
    spec_word = Set(spec_word)
    print("%d answer one calss, %d no class, %d multiple classes(resolved in q), %d multiple classes unresolved" %
      (self.one_ans_cls, self.no_ans_cls, self.many_ans_cls_q, self.many_ans_cls_n))
    print("they are: ", list(spec_word), "and all added to memory")
    print("time cost %.2f" % (time.time() - start_time))
    del self.one_ans_cls
    del self.no_ans_cls
    del self.many_ans_cls_q
    del self.many_ans_cls_n

    # -- Add answer source information
    for sname in ["train", "dev", "test"]:
      for aname, aset in [("ncon", self.id_set_ncon), ("conj", self.id_set_conj), ("wiki", self.id_set_wiki)]: 
        for case in aset[sname]: case["ac"] = ans_class[aname]
    for case in self.id_wik_sent: case["ac"] = ans_class["wiki"]
    for sname in ["train", "dev", "test"]:
      for aname, aset in [("ncon", self.case_set_ncon), ("conj", self.case_set_conj), ("wiki", self.case_set_wiki)]: 
        for case in aset[sname]: case["ac"] = aname
    for case in self.wiki_sent_set: case["ac"] = "wiki"
    return 

  def delete_long_ans_mem(self):
    all_sets = []
    for sname in ["train", "dev", "test"]:
      for aname, aset in [("ncon", self.id_set_ncon), ("conj", self.id_set_conj), ("wiki", self.id_set_wiki)]: 
        all_sets.append(aset[sname])
    all_sets.append(self.id_wik_sent)
    all_news = []
    for st in all_sets:
      newset = []
      for case in st:
        if(case["alen"] > MAX_A_LEN or case["mlen"] > MAX_M_LEN):
          continue
        else:
          case["ain"] = case["ain"][: MAX_A_LEN]
          case["aou"] = case["aou"][: MAX_A_LEN]
          case["mk"] = case["mk"][: MAX_M_LEN]
          case["mv"] = case["mv"][: MAX_M_LEN]
          newset.append(case)
      all_news.append(newset)
    i = 0
    for sname in ["train", "dev", "test"]:
      for aname, aset in [("ncon", self.id_set_ncon), ("conj", self.id_set_conj), ("wiki", self.id_set_wiki)]: 
        aset[sname] = all_news[i]
        i += 1
    self.id_wik_sent = all_news[-1]
    return 

  def build_batches(self, source = DATA_SOURCE):
    print("\nBuilding batches ... ")
    # -- Build batches 
    start_time = time.time()
    if(source == "all"):
      batches_set = { "valid":  self.id_set_ncon["dev"] + 
                                self.id_set_conj["dev"] + 
                                self.id_set_wiki["dev"],
                      "test":   self.id_set_ncon["test"] + 
                                self.id_set_conj["test"] +
                                self.id_set_wiki["test"],
                      "train":  self.id_set_ncon["train"] + 
                                self.id_set_conj["train"] + 
                                self.id_set_wiki["train"] + 
                                self.id_wik_sent }
    elif(source == "ncon"):
      batches_set = { "valid":  self.id_set_ncon["dev"],
                      "test":   self.id_set_ncon["test"],
                      "train":  self.id_set_ncon["train"] }
    else:
      raise ValueError("Invalid batch source!")
    batches = dict()
    for sname in batches_set:
      batches[sname] = {"q": [], "qlen": [], "c": [], 
                        "ain": [], "aou": [], "alen": [], "ac": [],
                        "m": [],
                        "mk": [], "mv": [], "mlen": []}
      rd.shuffle(batches_set[sname])
      for case in batches_set[sname]:
        for ci in case: batches[sname][ci].append(case[ci])
      for ci in batches[sname]: batches[sname][ci] = np.array(batches[sname][ci])
    # Output:
    self.batches = batches
    batch_pointer = dict()
    for ci in self.batches:
      batch_pointer[ci] = 0
    self.batch_pointer = batch_pointer

    # -- Statistics
    self.total_cases_train = len(batches["train"]["q"])
    self.total_cases_valid = len(batches["valid"]["q"])
    self.total_cases_test = len(batches["test"]["q"])
    print("total cases: %d in train, %d in valid, %d in test" % 
      (self.total_cases_train, self.total_cases_valid, self.total_cases_test))
    print("train set, %d non-conjunction, %d conjunction, %d answer wiki, %d normal wiki" % 
      (len(self.id_set_ncon["train"]), len(self.id_set_conj["train"]), 
       len(self.id_set_wiki["train"]), len(self.id_wik_sent)))
    print("time cost: %.2f" % (time.time() - start_time))
    self.max_a_len = MAX_A_LEN
    self.max_mem_len = MAX_M_LEN
    return 

  # Source of training set:
  # - None conjunction golden: answer sentences restricted to what is asked, pattern generated
  # - Conjunction golden: answer sentences conjuncted with other related information, pattern generated
  # - Answer wiki sentences: answer sentences from wiki with more sentence variety
  # - Normal wiki sentences: movie related sentences from wiki, for general purpose language model
  def get_next_batch(self, sname, batch_size):
    # print("in get next batch, batch_size = %d" % batch_size)
    ptr = self.batch_pointer[sname]
    # print("ptr = %d, %d cases in batch" % (ptr, len(self.batches[sname]["q"])))
    ptr = ptr if ptr + batch_size < len(self.batches[sname]["q"]) else len(self.batches[sname]["q"]) - batch_size
    self.batch_pointer[sname] = ptr
    qb = self.batches[sname]["q"][ptr: ptr + batch_size]
    qlenb = self.batches[sname]["qlen"][ptr: ptr + batch_size]
    qcb = self.batches[sname]["c"][ptr: ptr + batch_size]
    ainb = self.batches[sname]["ain"][ptr: ptr + batch_size]
    aoub = self.batches[sname]["aou"][ptr: ptr + batch_size]
    alenb = self.batches[sname]["alen"][ptr: ptr + batch_size]
    acb = self.batches[sname]["ac"][ptr: ptr + batch_size]
    mb = self.batches[sname]["m"][ptr: ptr + batch_size]
    mkb = self.batches[sname]["mk"][ptr: ptr + batch_size]
    mvb = self.batches[sname]["mv"][ptr: ptr + batch_size]
    mlenb = self.batches[sname]["mlen"][ptr: ptr + batch_size]
    self.batch_pointer[sname] = (self.batch_pointer[sname] + batch_size) % len(self.batches[sname]["q"])
    # print("transposing, before", ainb.shape)
    ainb = np.transpose(np.array(ainb))
    aoub = np.transpose(np.array(aoub))
    # print("transposing, after", ainb.shape)
    # test 
    # for mkbi in mkb:
    #   for mkbii in mkbi: 
    #     if(mkbii > 62402): print("hit!")
    # for mi in mb:
    #   if(mi > 62402): print("hit!")
    return  qb, qlenb, qcb, ainb, aoub, alenb, acb, mb, mkb, mvb, mlenb

  def test_batch(self, sname):
    qb, qlenb, qcb, ainb, aoub, alenb, acb, mb, mkb, mvb, mlenb = self.get_next_batch(sname, 3)
    ainb = np.transpose(ainb)
    aoub = np.transpose(aoub)
    for i in range(3):
      print("\ncase %d:" % i)
      case = {"q": qb[i], "qlen": qlenb[i], "c": qcb[i], 
              "ain": ainb[i], "aou": aoub[i], "alen": alenb[i], "ac": acb[i],
              "m": mb[i],
              "mk": mkb[i], "mv": mvb[i], "mlen": mlenb[i]}
      self.show_case(case)
    return

  def show_case(self, case):
    word2id = self.word2id
    id2word = self.id2word
    print("q, len = %d" % case["qlen"])
    print(case["q"][:case["qlen"]])
    print("   ".join([id2word[i] for i in case["q"][: case["qlen"]]]))
    print("ain, len = %d" % case["alen"])
    print(case["ain"][: case["alen"]])
    print("   ".join([id2word[i] for i in case["ain"][: case["alen"] + 1]]))
    print("aou: ")
    print(case["aou"][: case["alen"]])
    print("   ".join([id2word[i] if i < self.norm_word_cnt 
      else ("(loc %d -> " % (i - self.norm_word_cnt)) + id2word[case["mv"][i - self.norm_word_cnt]] + ")"
      for i in case["aou"][: case["alen"]]]))
    print("memory, len = %d" % case["mlen"])
    print("k:", case["mk"][: case["mlen"]])
    print("v:", case["mv"][: case["mlen"]])
    print([ (idx, self.mem_id2labels[i[0]], id2word[i[1]]) 
      for idx, i in enumerate(zip(case["mk"][:case["mlen"]], case["mv"][:case["mlen"]])) ])
    print("qc: %s, m: %s, ac: %s" % (self.id2qcls[case["c"]], id2word[case["m"]], self.id2acls[case["ac"]]))
    return 

  def count_person_name(self):
    print("\nCounting person names ... ")
    dir_names = Set(self.directors.keys())
    act_names = Set(self.actors.keys())
    wrt_names = Set(self.writers.keys())
    mov_names = Set(self.movie_names)
    print("%d director names" % len(dir_names))
    print("%d actor names" % len(act_names))
    print("%d writer names" % len(wrt_names))
    print("%d movie names" % len(mov_names))
    print("%d special names in total" % len(dir_names | act_names | wrt_names | mov_names))
    self.person_names = dir_names | act_names | wrt_names | mov_names
    return 

  def word_set_diff(self):
    print("\nCounting word set differences ... ")
    normal_word = []
    for m in self.movie_canddt_ans:
      for s in self.movie_canddt_ans[m]: normal_word.extend(s)
    normal_word_cnt = Counter(normal_word)
    normal_word = Set(normal_word)
    wiki_word = []
    for m in self.movie_sent:
      for s in self.movie_sent[m]: wiki_word.extend(s)
    wiki_word_cnt = Counter(wiki_word)
    wiki_word = Set(wiki_word)
    ques_word = []
    for sname in ["train", "test", "dev"]:
      for case in self.case_set_ncon[sname]: ques_word.extend(case["q"])
    ques_word_cnt = Counter(ques_word)
    ques_word = Set(ques_word)
    print("%d words in total" % len(normal_word | wiki_word | ques_word))
    print("%d normal word, %d question word, %d wiki word" % (len(normal_word), len(ques_word), len(wiki_word)))
    print("%d words only occur in wiki ..." % len(wiki_word - (normal_word | ques_word)))
    wiki_only = wiki_word - (normal_word | ques_word)
    wiki_only_cnt = 0
    wiki_only_minor = []
    for w in wiki_only:
      if(wiki_word_cnt[w] <= 10): 
        wiki_only_cnt += 1
        wiki_only_minor.append(w)
    print("... %d of them occur less than 10 times!" % wiki_only_cnt)
    self.wiki_only_minor = Set(wiki_only_minor)
    return 

  def clear(self):
    del self.entities
    del self.movie_names
    del self.poly_movie_names
    del self.movie_sent
    del self.movie_kb
    del self.movie_canddt_ans
    del self.movie_ans_conj
    del self.qapairs
    del self.case_set_ncon
    del self.case_set_conj
    del self.case_set_wiki
    del self.wiki_sent_set
    del self.id_set_ncon
    del self.id_set_conj
    del self.id_set_wiki
    del self.id_wik_sent
    return 

  def pipeline(self):
    start_time = time.time()
    self.read_entities()
    self.read_wiki()
    self.read_kb()
    self.read_ans_sent()
    self.read_questions()
    self.build_vocab()
    self.format_wiki()
    self.delete_long_sent()
    self.all_to_id()
    print("len of word2id: %d" % len(self.word2id))
    # self.format_all()
    # self.delete_long_ans_mem()
    # self.build_batches()
    # self.clear()
    print("%.2f seconds in total" % (time.time() - start_time))
    return 

  def build_remain(self):
    self.format_all()
    self.delete_long_ans_mem()
    self.build_batches()
    self.clear()
    return

def build_dataset():
  return 

def main():
  data_path = "../data/dataset.pkl"  
  if not os.path.exists(data_path):
    print("Building from scratch ... ")
    dset = Dataset()
    print("\nStoring dataset ... ")
    cPickle.dump(dset, open(data_path, "wb"))
  else:
    print("loading data ... ")
    start_time = time.time()
    dset = cPickle.load(open(data_path, "rb"))
    print("data loading finished, time consume %.4f\n" % (time.time() - start_time))
  return 

if __name__ == "__main__":
  main()
