/*##############################################################################
## HPCC SYSTEMS software Copyright (C) 2019 HPCC Systems.  All rights reserved.
############################################################################## */

/**
  * Unit Test for the Corpus Subsystem
  */

IMPORT $.^.Types;
IMPORT $.^.internal AS int;

Word := Types.Word;
Sentence := Types.Sentence;

sentences := DATASET([{1, 'i [painted] my house white'},
                      {2, 'i painted my car "black"'},
                      {3, ' you painted your  Bike "yellow" '},
                      {4, 'you colored your car green'},
                      {5, '/he painted his fence, blue'},
                      {6, 'the sky is "blue"'},
                      {7, 'trees are green'},
                      {8, 'newspapers are black / white'},
                      {9, 'the  SUN     is  yellow '}], Sentence);

// Set the discard threshold high, since there are very few words in our vocab
//corp := int.Corpus(sentences, discThreshold := .01, wordNGrams := 1);
corp := int.Corpus(sentences, discThreshold := .1, wordNGrams := 2);

vocab := corp.vocabulary;
vocabSize := corp.vocabSize;
wordCount := corp.wordCount;

OUTPUT(vocabSize, NAMED('VocabSize'));
OUTPUT(vocab, ALL, NAMED('Vocabulary'));
OUTPUT(wordCount, NAMED('CorpusWordCount'));
sentLenRec := RECORD
  UNSIGNED len;
END;
sentLens := DATASET([{1}, {2}, {3}, {4}, {5}, {6}, {7}, {8}, {9}, {10}, {11}, {12}], sentLenRec);



trainDat := corp.GetTraining;

OUTPUT(trainDat[..5000], ALL, NAMED('TrainingData'));
OUTPUT(COUNT(trainDat), NAMED('TrainingCount'));
SET OF STRING s := ['ABC', 'DEF', 'GHIJ', 'KLM', 'NOPQ', 'RST', 'UVWX', 'YZ'];

SET OF VARSTRING s2 := corp.getNGrams(s, 4, 3);

OUTPUT(s2, NAMED('Ngrams'));

