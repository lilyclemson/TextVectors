/*##############################################################################
## HPCC SYSTEMS software Copyright (C) 2019 HPCC Systems.  All rights reserved.
############################################################################## */
IMPORT $.^ AS TV;
IMPORT TV.Types;
IMPORT $.^.internal AS int;
IMPORT Std.system.Thorlib;
IMPORT int.svUtils AS Utils;
IMPORT Std.Str;

t_TextId := Types.t_TextId;
t_Sentence := Types.t_Sentence;
t_Vector := Types.t_Vector;
t_Word := Types.t_Word;
t_WordId := Types.t_WordId;
t_SentId := Types.t_SentId;
TextMod := Types.TextMod;
Vector := Types.Vector;
Word := Types.Word;
WordInfo := Types.WordInfo;
WordList := Types.WordList;
Sentence := Types.Sentence;
SentInfo := Types.SentInfo;
SliceExt := Types.SliceExt;
TrainStats := Types.TrainStats;
t_ModRecType := Types.t_ModRecType;
Closest := Types.Closest;
WordExt := Types.WordExt;
typeWord := Types.t_ModRecType.word;

mappingRec := RECORD
  t_Word orig;
  t_Word new;
END;
/**
  * Unit test for the top-level SentenceVectors interface.
  * Note that without a large corpus, the results are fairly meaningless.
  * This test calls all of the public attributes, and shows that the results
  * are returned.  For a true live test see the MVR project that uses a
  * real corpus.
  */




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

vecLen := 20;
numEpochs := 10;
minOccurs := 1;
wordNGrams := 2; 
negSamples := 3; 
learningRate := .4; 
discardThreshold := .1;
saveSentences := TRUE;
vecLen=100;
trainToLoss := .05;
batchSize:=0;
dropoutK := 3;
noProgressEpochs := 1;
maxTextDistance := 3;
maxNumDistance := 9;
corp := int.Corpus(sentences, wordNGrams, discardThreshold, minOccurs, dropoutK);
vocabulary := corp.Vocabulary;
trainDat := corp.GetTraining;
trainCount := COUNT(trainDat);
// OUTPUT(trainCount); // working
vocabSize := corp.vocabSize;
nnShape := [vocabSize, vecLen, vocabSize];
calConst := 25;
// If the default batchSize is zero (default -- auto), automatically calculate
// a reasonable value.
// nWeights = # of weights = 2 * vocabSize * vecLen
// ud = Update Density = # of weights per batch / # of weights = 2 * vecLen * (1 + negSamp) * batchSize * nNodes /
//      (2 * veclen * vocabSize) = (1 + negSamp) * batchSize * nNodes / vocabSize
// We want to adjust batchSize such that ud = calConst.
// batchSize = calConst * vocabSize / ((1+ negSamp) * nNodes)
node := Thorlib.node();
nNodes := Thorlib.nodes();
batchSizeCalc := (UNSIGNED4)(calConst * vocabSize) / ((1 + negSamples) * nNodes);
batchSizeAdj := IF(batchSize = 0, batchSizeCalc, batchSize);
// Set up the neural network and do stochastic gradient descent to train
// the network.
nn := int.SGD(nnShape, trainToLoss, numEpochs, batchSizeAdj, learningRate, negSamples, noProgressEpochs);
// OUTPUT(nn);// working
finalWeights := nn.Train_Dupl(trainDat);
OUTPUT(finalWeights);// ISSUE ****

computeWordVectors(DATASET(SliceExt) slices, DATASET(WordInfo) words,
																SET OF UNSIGNED shape) := FUNCTION
    w := int.Weights(shape);  // Module to manage weights
    // As an optimization, we only convert half of the weight slices
    // since the word vectors are always in the first half
    firstHalf := ROUNDUP(w.nSlices / 2);
    allWeights := w.slices2Linear(slices(sliceId <= firstHalf ));
    // Extract the vectors for each word.  Words should be evenly distributed at this point.
    WordInfo getVectors(WordInfo wrd) := TRANSFORM
      // The weights that form the word vector are the layer 1 weights.  j is the wordId
      // and i is the term number of the word vector.  The terms of the word vector are
      // contiguous, so we only need to know the start and end indexes.
      startIndx := w.toFlatIndex(1, wrd.wordId, 1);
      endIndx := startIndx + shape[2] - 1;
      SELF.vec := Utils.normalizeVector(allWeights.weights[startIndx .. endIndx]);
      SELF := wrd;
    END;
    wordsWithVecs := PROJECT(words, getVectors(LEFT), LOCAL);
    RETURN wordsWithVecs;
END;
// Now extract the final weights for the first layer as the word vectors
wVecs := computeWordVectors(finalWeights, vocabulary, nnShape);
// OUTPUT(wVecs);



DATASET(TextMod) makeWordModel(DATASET(WordInfo) words) := FUNCTION
    modOut := PROJECT(words, TRANSFORM(TextMod, SELF.typ := Types.t_ModRecType.Word,
                          SELF.id := LEFT.wordId, SELF := LEFT), LOCAL);
    RETURN modOut;
  END;
// And produce the word portion of the model.
wMod := makeWordModel(wVecs);  
//OUTPUT(wMod);

DATASET(mappingRec) findClosestMatch(DATASET(TextMod) mod, DATASET(mappingRec) words) := FUNCTION
    distRec := RECORD(mappingRec)
      UNSIGNED4 eDist;
    END;
    wordsN := words(Utils.isNumeric(orig));
    wordsT := words(Utils.isNumeric(orig)=FALSE);
    d0T := JOIN(wordsT, mod(Utils.isNumeric(text)=FALSE), TRUE,
                      TRANSFORM(distRec, SELF.eDist := Str.EditDistance(LEFT.orig, RIGHT.text),
                      SELF.new := RIGHT.text, SELF := LEFT), ALL);
    d0N := JOIN(wordsN, mod(Utils.isNumeric(text)), TRUE,
                      TRANSFORM(distRec, SELF.eDist := Utils.numDistance(LEFT.orig, RIGHT.text),
                      SELF.new := RIGHT.text, SELF := LEFT), ALL);
    d0TF := d0T(eDist <= maxTextDistance);
    d0NF := d0N(eDist <= maxNumDistance);
    d0 := d0TF + d0NF;
    d0D := DISTRIBUTE(d0, HASH32(orig));
    d1 := SORT(d0D, orig, eDist, LOCAL);
    d2 := DEDUP(d1, orig);
    d3 := PROJECT(d2, mappingRec, LOCAL);
    out := DISTRIBUTE(d3, HASH32(orig));
    RETURN out;
  END;


mapWords(DATASET(TextMod) mod, DATASET(WordExt) allWords) := FUNCTION
    // First find unique words to avoid repeating expensive mapping operation
    //  Allwords should be distributed by HASH32(text) at this point
    //  mod should also be distributed by HASH32(text).
    allWordsD := DISTRIBUTE(allWords, HASH32(text));
    modD := DISTRIBUTE(mod, HASH32(text));
    allWordsS := SORT(allWordsD, text, LOCAL);
    // Find unique
    words := DEDUP(allWordsS, text, LOCAL);
    // Join the unique words with the model.  If a word is not in the model, find the closest
    // word to it in typographic edit distance.
    wordsM0 := JOIN(words, modD, LEFT.text = RIGHT.text,
                TRANSFORM(mappingRec, SELF.orig := LEFT.text,
                  SELF.new := RIGHT.text),
               LEFT OUTER, LOCAL);
    missing := wordsM0(LENGTH(new) = 0);
    found := findClosestMatch(mod, missing);
    good := wordsM0(LENGTH(new) != 0);
    wordsM := good + found;
    // Now map the changes back into a new version of allWords
    mapped := JOIN(allWordsD, wordsM, LEFT.text = RIGHT.orig,
                      TRANSFORM(WordExt, SELF.text := RIGHT.new,
                                  SELF := LEFT), LOCAL);
    // Re DISTRIBUTE since some of the text may have changed.
    RETURN DISTRIBUTE(mapped, HASH32(text));
  END;

SentInfo sent2vector(DATASET(TextMod) mod, DATASET(Sentence) sent, UNSIGNED2 vecLen,
                          BOOLEAN mapMissingWords = TRUE) := FUNCTION
    // Should ultimately optimize with C++
    corp := int.Corpus(wordNGrams := wordNGrams);
    // Get the tokenized sentence
    wl := corp.sent2wordList(sent);
    wordExt getWords(WordList w, UNSIGNED c) := TRANSFORM
      SELF.sentId := w.sentId;
      SELF.text := w.words[c];
    END;
    // Create a separate record for each word
    allWords := NORMALIZE(wl, COUNT(LEFT.words), getWords(LEFT, COUNTER));
    allWordsD := DISTRIBUTE(allWords, HASH32(text));
    modD := DISTRIBUTE(mod, HASH32(text));
    
    // If requested, map any words not in the model to the closest approximation
    allWordsM := IF(mapMissingWords, mapWords(modD, allWordsD), allWordsD);
    // Get the word vectors for each word from the model
    sentWords0 := JOIN(allWordsM, modD, LEFT.text = RIGHT.text,
                        TRANSFORM(SentInfo, SELF.sentId := LEFT.sentId,
                            SELF.vec := RIGHT.vec,
                            SELF := LEFT),
                        LOCAL);
    // Redistribute by sentence id
    sentWords := SORT(DISTRIBUTE(sentWords0, sentId), sentId, LOCAL);
    SentInfo doRollup(sentInfo lr, SentInfo rr) := TRANSFORM
      SELF.vec := lr.vec + rr.vec;
      SELF.sentId := lr.sentId;
      SELF.text := '';
    END;
    // Accumulate all of the word vectors into a single long vector
    sentOut0 := ROLLUP(sentWords, doRollup(LEFT, RIGHT), sentId, LOCAL);
    // Now reduce the concatenated word vectors to a single word vector, and restore the original
    // sentence text.
    sentD := DISTRIBUTE(sent, sentId);
    sentOut := JOIN(sentOut0, sentD, LEFT.sentId = RIGHT.sentId, TRANSFORM(RECORDOF(LEFT),
                              SELF.vec := Utils.calcSentVector(LEFT.vec, vecLen),
                              SELF.text := RIGHT.text,
                              SELF := RIGHT), LOCAL);
    RETURN sentOut;
  END;

// Calculate a vector for each sentence in the corpus to produce the sentence portion
// of the model
sVecs := sent2vector(wMod, corp.Sentences, vecLen, mapMissingWords := FALSE);
//OUTPUT(sVecs);


DATASET(TextMod) makeSentModel(DATASET(SentInfo) sent) := FUNCTION
    modOut := PROJECT(sent, TRANSFORM(TextMod, SELF.typ := Types.t_ModRecType.Sentence,
                          SELF.id := LEFT.sentId, SELF := LEFT), LOCAL);
    RETURN modOut;
  END;
sMod := makeSentModel(sVecs);
//OUTPUT(sMod);
// Concatenate the two portions, unless saveSentences is FALSE.
mod := IF(saveSentences, wMod + sMod, wMod);
// OUTPUT(mod[..500]);





// model := SV.GetModel(sentences);

// OUTPUT(model[..5000], ALL, NAMED('Model'));

/*
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

*/