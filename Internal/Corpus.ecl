/*##############################################################################
## HPCC SYSTEMS software Copyright (C) 2019 HPCC Systems.  All rights reserved.
############################################################################## */
IMPORT $.^.Types;
IMPORT Std.str;
IMPORT Std.System.Thorlib;

nNodes := Thorlib.nodes();
Sentence := Types.Sentence;
SentInfo := Types.SentInfo;
WordInfo := Types.WordInfo;
t_Sentence := Types.t_Sentence;
t_Word := Types.t_Word;
t_Vector := Types.t_Vector;
WordList := Types.WordList;
TrainingDat := Types.TrainingDat;
t_WordId := Types.t_WordId;
t_SentId := Types.t_SentId;
NegativeRec := Types.NegativeRec;

UNSIGNED4 RAND_MAX := 4294967295; 
// Intermediate format for the training pairs stored as text
TrainingPairT := RECORD
  t_Word main;
  t_Word context;
END;
/**
  * Analyze a corpus of sentences to support vectorization activities.
  * <p>Tokenizes sentences into words, provides a Vocabulary of unique words, and
  * supports conversion of sentences into training data.
  * @param wordNGrams The maximum sized NGram to generate.  1 indicates unigrams only.
  *                   2 indicates unigrams and bigrams. 3 indicates uni, bi, and trigrams.
  *                   Defaults to 1 (unigrams only).
  * @param discThreshold Discard threshold.  Words with frequency greater than or equal to this number
  *                   are probabilistically discarded based on their frequency.  Words with
  *                   frequencies below this threshold are never discarded (Default .0001).
  * @param minOccurs Words that occur less than this number of times in the corpus are eliminated
  *                  (Default 5).
  *                  Words with very few occurrences in the corpus may not get properly trained due
  *                  to lack of context.
  * @param dropoutK The number of NGrams to drop from a sentence (per Sent2Vec paper).  Default 3.
  */
EXPORT Corpus(DATASET(Sentence) sentences_in=DATASET([], Sentence),
              UNSIGNED4 wordNGrams = 1,
              REAL4 discThreshold = .0001,
              UNSIGNED4 minOccurs = 5,
              UNSIGNED4 dropoutK = 3) := MODULE
  /**
    * Return a dataset of Sentences, distributed evenly.
    */
  EXPORT sentences := DISTRIBUTE(sentences_in, sentId);

  /**
    * Produce a series of nGrams from the set of words in a sentence.
    * For example, if the parameter ngrams is three, it will produce the set of
    * Bigrams (i.e. 2grams) as well as the set of Trigrams (i.e. 3grams).
    * Ngrams are formatted as _Word1_Word2_Word3.  Given the sentence ['the', 'quick',
    * 'brown', 'fox'] and ngrams set to 3, it will return: ['_the_quick', '_quick_brown',
    * '_brown_fox', '_the_quick_brown', '_quick_brown_fox'].
    */
  EXPORT SET OF VARSTRING getNGrams(SET OF VARSTRING words, UNSIGNED4 ngrams, UNSIGNED4 dropoutk = 0) := EMBED(C++)
    #include <string>
    #body
    uint16_t maxWords = 1000;  // A sentence shouldn't have more than 1000 words.
    char * pos = (char *) words;
    char * endpos = pos + lenWords;
    char ** wordlist = (char**)rtlMalloc(maxWords * sizeof(char *));
    uint16_t numWords = 0;
    uint32_t outSize = 0;
    // Extract the list of words from the input
    while (pos < endpos && numWords < maxWords)
    {
      wordlist[numWords] = pos;
      numWords++;
      pos += strlen(pos) + 1;
    }
    uint16_t n, i, j, inWords;
    inWords = numWords; // Save original word count
    // Decide which words' Ngrams should be dropped based on dropout K parameter.
    // Note:  I'm not sure of the logic behind dropping K words' Ngrams.  This
    // biases against Ngrams in short sentences.  The authors of Sent2vec, however
    // found it helpful and found it to work better than randomly dropping at a given
    // rate, so it is implemented according to their paper.
    bool * discardTokens = (bool *)rtlMalloc(inWords * sizeof(bool));
    for (i = 0; i < inWords; i++)
      discardTokens[i] = false;
    uint16_t numDiscard = 0;
    while ((numDiscard <= dropoutk) && (inWords - numDiscard > 2))
    {
      uint32_t r = rand() % inWords;
      if (!discardTokens[r])
      {
        discardTokens[r] = true;
        numDiscard ++;
      }
    }
    uint32_t maxOutbuff = 10000; // Safety maximum size of ngrams
    char * outBuff =  (char *) rtlMalloc(maxOutbuff);
    char * outPos = outBuff;
    // Produce 2-grams up to n-grams
    for (n = 2; n <= ngrams; n++)
    {
      // Loop through the input words
      for (i = 0; i < inWords - (n-1); i++)
      {
        // Skip the word if it was chosen for Ngram discard
        if (discardTokens[i])
          continue;
        char * ngram = (char *) rtlMalloc(1000);
        uint32_t size = strlen(wordlist[i]) + 1;
        ngram[0] = '_';
        strcpy(ngram + 1, wordlist[i]);
        // Form n-gram from the input word, plus subsequent words
        for (j = 1; j < n; j++)
        {
          char * w = wordlist[i + j];
          ngram[size++] = '_';
          strcpy(ngram + size, w);
          size += strlen(w);
        }
        ngram[size++] = 0; // add 1 for the null terminator
        // Bail out if we've reached the maxOutbuff size so we don't overflow.
        if (outPos + size >= outBuff + maxOutbuff)
        {
          rtlFree(ngram);
          break;
        }
        // Copy the ngram into the output buffer
        strcpy(outPos, ngram);
        outPos += size;
        outSize += strlen(ngram) + 1;
        rtlFree(ngram);
      }
    }
    rtlFree(wordlist);
    rtlFree(discardTokens);
    __result = (void *) outBuff;
    __lenResult = outSize;
    __isAllResult = FALSE;
  ENDEMBED;
  /**
    * Convert a sentence to a list of words (including n-grams if requested).
    * Strip out punctuation, cleanup whitespace, and split the words.
    */
  EXPORT DATASET(WordList) sent2wordList(DATASET(Sentence) sent) := FUNCTION
    WordList makeWordList(Sentence s) := TRANSFORM
      noPunctuation := Str.SubstituteIncluded(s.text, '[./*!:;,%^()@#$"-+&]', ' ');
      lessThan := Str.FindReplace(noPunctuation, '<', ' < ');
      greatThan := Str.FindReplace(lessThan, '>', ' > ');
      trimmed := Str.CleanSpaces(greatThan);
      toLower := Str.ToLowerCase(trimmed);
      words := Str.SplitWords(toLower, ' ');
      SELF.words := IF(wordNGrams > 1, words + getNGrams(words, wordNGrams, dropoutK), words);
      SELF.sentId := s.sentId;
    END;

    wLists := PROJECT(sent, makeWordList(LEFT), LOCAL);
    RETURN wLists;
  END;

  /**
    * Each sentence transformed to a word list.
    */
  EXPORT DATASET(WordList) tokenizedSent := sent2wordList(sentences);

  /**
    * Find unique vocabulary words, and remove any rare words that occur
    * less than minOccurs times in the corpus.
    */
  SHARED DATASET(WordInfo) Vocabulary0 := FUNCTION
    words := NORMALIZE(tokenizedSent, COUNT(LEFT.words), TRANSFORM(WordInfo, SELF.wordId := 0, SELF.text := LEFT.words[COUNTER]));
    wordsD := DISTRIBUTE(words, HASH32(text));
    wordsS := SORT(wordsD, text, LOCAL);
    wordsUnique := ROLLUP(wordsS, TRANSFORM(WordInfo, SELF.occurs := LEFT.occurs + RIGHT.occurs,
                                                    SELF := LEFT), text, LOCAL);
    // Eliminate any words that occur less than minOccurs times in the corpus.  Assign
    // ids to the remaining words
    vocab := PROJECT(wordsUnique(occurs >= minOccurs), TRANSFORM(WordInfo, SELF.wordId := COUNTER, SELF := LEFT));
    RETURN vocab;
  END;

  /**
    * The size of the vocabulary
    */
  EXPORT VocabSize := COUNT(Vocabulary0);

  /**
    * The number of words in the full corpus.
    */
  wcTable := TABLE(Vocabulary0, {wc := SUM(GROUP, occurs)}, FEW);
  EXPORT wordCount := wcTable[1].wc;

  /**
    * Each word is assigned a discard probability based on its frequency
    * in the corpus.  Very common words are sub-sampled (by discarding some
    * occurrences) to avoid over-training of those common words and speed up
    * training
    */
  WordInfo assignDiscardProb(WordInfo w) := TRANSFORM
    // discThreshold = No discards below this frequency.
    // freq = occurs / wordCount
    // q = MIN[1, SQRT(discThreshold / freq) + discThreshold / freq]
    // probDiscard = 1 - q
    REAL freq := w.occurs / wordCount;
    REAL ifreq := discThreshold / freq;
    REAL q := MIN(1, SQRT(ifreq) + ifreq);
    SELF.pdisc := 1 - q;
    SELF := w;
  END;

  /**
    * The set of all the unique words in the corpus.
    * Note: returned vocabulary is distributed by HASH32(text),
    * and sorted by text.
    */
  EXPORT Vocabulary := PROJECT(Vocabulary0, assignDiscardProb(LEFT), LOCAL);

  SHARED expSentence := RECORD
    t_SentId sentId;
    UNSIGNED ord;
    t_Word word;
  END;
  SHARED expSentenceId := RECORD
    t_SentId sentId;
    UNSIGNED ord;
    t_WordId wordId;
    REAL4 pdisc;
  END;
  EXPORT wordIdList := RECORD
    UNSIGNED4 sentId;
    UNSIGNED2 ord;
    SET OF t_WordId wordIds;
    REAL4 pdisc;
  END;
  /**
    * Convert a list of textual words making up a sentence to a set of ids representing
    * each word's wordId in the vocabulary.
    */
  EXPORT DATASET(wordIdList) WordList2WordIds(DATASET(WordList) sent) := FUNCTION
    expSent := NORMALIZE(sent, COUNT(LEFT.words), TRANSFORM(expSentence,
                            SELF.sentId := LEFT.sentId,
                            SELF.ord := COUNTER,
                            SELF.word := LEFT.words[COUNTER]));
    expSentId := JOIN(expSent, Vocabulary, LEFT.word = RIGHT.text, 
                          TRANSFORM(expSentenceId,
                            SELF.wordId := RIGHT.wordId,
                            SELF.pdisc := RIGHT.pdisc,
                            SELF := LEFT), LOOKUP);
    expSentId_S := SORT(expSentId, sentId, ord, LOCAL);
    widl0 := PROJECT(expSentId_S, TRANSFORM(wordIdList,
                                    SELF.wordIds := [LEFT.wordId],
                                    SELF := LEFT), LOCAL);
    widl := ROLLUP(widl0, TRANSFORM(wordIdList,
                                      SELF.wordIds := LEFT.wordIds + RIGHT.wordIds,
                                      SELF := LEFT), sentId, LOCAL);
    RETURN widl;
  END;


  /**
    * Generate training data based on the corpus.  Each record is a main word
    * and a set of context words (words that occur with that word in a sentence).
    */
  EXPORT DATASET(TrainingDat) GetTraining := FUNCTION
    wordIds := WordList2WordIds(tokenizedSent);
    // Create pairs of main and context (i.e. all words except main).
    // While we're at it, probabilistically skip each main word based on it's discard probability.
    tdat := NORMALIZE(wordIds, COUNT(LEFT.wordIds), TRANSFORM(TrainingDat,
                                SELF.main := IF(RANDOM() / RAND_MAX < LEFT.pdisc,
                                                    SKIP,
                                                    LEFT.wordIds[COUNTER]),
                                SELF.context := LEFT.wordIds[1..COUNTER-1] +
                                                  LEFT.wordIds[COUNTER+1..COUNT(LEFT.wordIds)]));
    RETURN tdat;
  END;
  /**
    * Working Record for producing the negatives table (see below).
    */
  SHARED negRec0 := RECORD
    REAL4 prob;
  END;
  /**
    * Negatives Table is a record containing the discard probability of each word
    * in the vocabulary as a single SET, indexed by the wordId.
    * It is not currently used but is left here for possible future use.
    */
  EXPORT NegativesTable := FUNCTION
    v0 := SORT(Vocabulary, wordId);
    v1 := PROJECT(v0, TRANSFORM(negRec0,
                        SELF.prob := SQRT(LEFT.occurs)), LOCAL);
    maxProbTab := TABLE(v1, {maxP := MAX(GROUP, prob)});
    maxProb := maxProbTab[1].maxP;
    v2 := PROJECT(v1, TRANSFORM(NegativeRec,
                            SELF.probs := [LEFT.prob / maxProb], SELF.nodeId := 0), LOCAL);
    v3 := ROLLUP(v2, TRUE, TRANSFORM(RECORDOF(LEFT),
                                SELF.probs := LEFT.probs + RIGHT.probs, SELF.nodeId := 0));
    v4 := NORMALIZE(v3, nNodes, TRANSFORM(RECORDOF(LEFT),
                            SELF.nodeId := COUNTER, SELF := LEFT));
    v := DISTRIBUTE(v4, nodeId);
    RETURN v;
  END;
END;