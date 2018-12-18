/*##############################################################################
## HPCC SYSTEMS software Copyright (C) 2019 HPCC Systems.  All rights reserved.
############################################################################## */
IMPORT $.^ AS TV;
IMPORT TV.Types;
/**
  * Unit test for the top-level SentenceVectors interface.
  * Note that without a large corpus, the results are fairly meaningless.
  * This test calls all of the public attributes, and shows that the results
  * are returned.  For a true live test see the MVR project that uses a
  * real corpus.
  */
Sentence := Types.Sentence;
Word := Types.Word;
WordExt := Types.WordExt;
t_Word := Types.t_Word;
typeWord := Types.t_ModRecType.word;
t_Vector := Types.t_Vector;


SV := TV.SentenceVectors(vecLen := 20, numEpochs := 10, minOccurs := 1,
              wordNGrams := 2, negSamples := 3, learningRate := .4, discardThreshold := .1,
              saveSentences := TRUE);

edTestRec := RECORD
  STRING str1;
  STRING str2;
  UNSIGNED4 dist := 0;
END;


sentences := DATASET([{1, 'i [painted] my house white'},
                      {2, 'i painted my car "black"'},
                      {3, ' you painted your  Bike "yellow" '},
                      {4, 'you colored your car green'},
                      {5, 'he painted his fence, blue'},
                      {6, 'the sky is "blue"'},
                      {7, 'trees are green'},
                      {8, 'newspapers are black / white'},
                      {9, 'the  SUN     is  yellow '}], Sentence);

model := SV.GetModel(sentences);

OUTPUT(model[..5000], ALL, NAMED('Model'));
tempRec := RECORD
  STRING text;
  UNSIGNED len;
  t_Vector vec;
END;
tempOut := PROJECT(model(typ = typeWord), TRANSFORM(tempRec, SELF.text := LEFT.text,
                      SELF.len := LENGTH(LEFT.text), SELF.vec := LEFT.vec));
OUTPUT(tempOut, NAMED('tempOut'));

trainStats := SV.GetTrainStats(model);

OUTPUT(trainStats, NAMED('TrainStats'));
mapRec := RECORD
  t_Word orig;
  t_Word mapsto := '';
END;
testMap := DATASET([{'panted'},
                    {'th'},
                    {'bue'},
                    {'wspapers'},
                    {'sin'},
                    {'tres'},
                    {'trees'}], mapRec);

testWords := DATASET([{1, 'i'}, {1, 'panted'}, {1, 'th'}, {1, 'house'},
                    {2, 'tres'}, {2, 'are'}, {2, 'green'}], WordExt);


words := DATASET([{1,'i'}, {2,'painted'}, {3,'my'}, {4,'house'}, {5, 'white'}], Word);

wVecs := SV.GetWordVectors(model, words);

OUTPUT(wVecs, NAMED('WordVectors'));

sentVecs := SV.GetSentVectors(model, sentences);

OUTPUT(sentVecs, NAMED('SentVectors'));

closeWords := SV.ClosestWords(model, words, 5);

OUTPUT(closeWords, NAMED('ClosestWords'));

testSent := DATASET([{1, 'I painted MY    car white'},
                      {2, 'You PAINTED / your house white'},
                      {3, 'I painted My, car black'},
                      {4, 'I panted my cr black'}], Sentence);
closeSentences := SV.ClosestSentences(model, testSent, 3);
OUTPUT(closeSentences, NAMED('ClosestSentences'));

leastSim := SV.LeastSimilarWords(model, DATASET([{1,'white'}, {2,'black'},
                                          {3,'yellow'}, {4,'green'}, {5, 'i'}, {6,'blue'}], word));

OUTPUT(leastSim, NAMED('LeastSimilarWords'));

testVec1 := sentVecs[1].vec;
testVec2 := sentVecs[2].vec;

sim := SV.Similarity(testVec1, testVec2);

OUTPUT(sim, NAMED('Similarity'));

analogy := SV.WordAnalogy(model, 'i', 'painted', 'you');

OUTPUT(analogy, NAMED('Analogy'));



