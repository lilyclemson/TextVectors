/*##############################################################################
## HPCC SYSTEMS software Copyright (C) 2019 HPCC Systems.  All rights reserved.
############################################################################## */
IMPORT $.^ AS TV;
IMPORT TV.Types;
IMPORT $.^.internal AS int;
IMPORT Std.system.Thorlib;
IMPORT int.svUtils AS Utils;
IMPORT Std.Str;
IMPORT STD;

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
vocabSize := corp.vocabSize;
nnShape := [vocabSize, vecLen, vocabSize];
calConst := 25;
node := Thorlib.node();
nNodes := Thorlib.nodes();
batchSizeCalc := (UNSIGNED4)(calConst * vocabSize) / ((1 + negSamples) * nNodes);
batchSizeAdj := IF(batchSize = 0, batchSizeCalc, batchSize);
nn := int.SGD(nnShape, trainToLoss, numEpochs, batchSizeAdj, learningRate, negSamples, noProgressEpochs);


shape:= nnShape; 
// trainToLoss:=.05; 
// numEpochs:=0; 
miniBatchSize:=batchSizeAdj; 
lr:=learningRate; 
negSamp:=negSamples;
// noProgressEpochs:=5 ;
w := int.Weights(shape);       
OUTPUT(w);


// Initialize the weights to random values
initWeights := w.initWeights;
// Get the size of each slice of the weights (i.e. number of weights)
sliceSize := w.sliceSize;
// Copy the weights to all nodes as SliceExt records.
initWeightsExt := w.toSliceExt(initWeights);
    // Get the size of the local segment of the training data.
trainSize := TABLE(trainDat, {cnt := COUNT(GROUP)}, LOCAL)[1].cnt;
// OUTPUT(trainSize);

slice := Types.slice;
CSlice := Types.CSlice;
TrainingDat := Types.TrainingDat;
NegRec := Types.NegativeRec;

RandomizeSamples(DATASET(trainingDat) dIn) := FUNCTION
    tempRec := RECORD(trainingDat)
      UNSIGNED4 rnd;
    END;
    d0 := PROJECT(dIn, TRANSFORM(tempRec, SELF.rnd := RANDOM(), SELF := LEFT), LOCAL);
    d1 := SORT(d0, rnd, LOCAL);
    dOut := PROJECT(d1, trainingDat);
    return dOut;
  END;

SET OF REAL8 addWeights(SET OF REAL8 w1, SET OF REAL8 w2, UNSIGNED4 numweights) := EMBED(C++)
#body
__lenResult = (size32_t) (numweights * sizeof(double));
double *wout = (double*) rtlMalloc(__lenResult);
__isAllResult = false;
__result = (void *) wout;
double *ww1 = (double *) w1;
double *ww2 = (double *) w2;
for (uint32_t i = 0; i < numweights; i++)
{
    wout[i] = ww1[i] + ww2[i];
}
ENDEMBED;

calcProgress(REAL avgLoss) := FUNCTION
    // Scale loss into range [0,1].  The highest non-spurious loss should be .5.
    // The lowest interesting loss is trainToLoss, because that's where we stop.
    // Protect the bounds to make sure we don't get a loss > .5.
    avL := MIN(.5, avgLoss);
    // Scaled Loss at .5 should be 1, and at trainToLoss should be 0.
    scaledL(REAL4 x) := MAX(0, (x - trainToLoss) / (.5 - trainToLoss));
    // The multiplier is heuristic since we don't know the actual exponential
    // This has the effect of moving up on the logarithm curve, where the slope is shallower.
    logMultiplier := 150;
    logL(REAL4 x) := LN(1 + logMultiplier * scaledL(x));
    // Scaled Log Loss [0,1]. But reduce the range to below 15.
    // That is because the first part of the log reduction curve is nearly vertical
    // and we can't compensate for it.  ScaledLL will be .99 until we get down to
    // 15.
    minLinear := .15;
    scaledLL(REAL4 x) := MIN(1, (LogL(x)) / logL(minLinear));
    linLoss := scaledLL(avL);
    // Always report some progress (.01) even when we're above trainToLoss * 2.
    // Square the progress to account for the reducing effect of Learning Rate which
    // is reduced with progress.
    altProgress := (.5 - avL) * .1;
    progress := MAX(.01 , POWER(1 - linLoss, 2), altProgress);
    return progress;
  END;

rollUpdates(DATASET(SliceExt) inWeights, DATASET(SliceExt) updates) := FUNCTION
    combined := SORT(inWeights+updates, sliceId, LOCAL);
    SliceExt doRollup(SliceExt l, SliceExt r) := TRANSFORM
      SELF.weights := addWeights(l.weights, r.weights, w.sliceSize);
      SELF.loss := l.loss + r.loss;
      // To avoid premature stopping, use the max loss for all nodes and
      // the max epoch of minimum loss.
      SELF.minLoss := MAX(l.minLoss, r.minLoss);
      SELF.minEpoch := MAX(l.minEpoch, r.minEpoch);
      SELF := l;
    END;
    outWeights := ROLLUP(combined, doRollup(LEFT, RIGHT), sliceId, LOCAL);
    outWeightsD := w.distributeAllSlices(outWeights);
    outWeightsS := SORT(outWeightsD, sliceId, LOCAL);
    RETURN outWeightsS;
  END;

rollUpdatesC(DATASET(SliceExt) inWeights, DATASET(CSlice) updates) := FUNCTION
    uncompUpdates := w.decompressWeights(updates);
    RETURN rollUpdates(inWeights, uncompUpdates);
  END;

BOOLEAN isConverged(DATASET(SliceExt) slices, UNSIGNED c, UNSIGNED trainSize) := FUNCTION
    isStalled := c - slices(nodeId = node AND sliceId = 1)[1].minEpoch > noProgressEpochs;
    isFinished := numEpochs > 0 AND c > numEpochs;
    firstW := slices(nodeId = node AND sliceId = 1)[1]; 
    loss := firstW.loss;
    avgLoss := loss / (trainSize * (1 + negSamp));
    isConverged := IF(loss > 0, avgLoss < trainToLoss, FALSE);
    RETURN isStalled OR isFinished OR isConverged;
  END;
  /**
    * Determines whether an epoch is complete by comparing the records processed
    * with the training set size.
    */
BOOLEAN isEpochDone(DATASET(SliceExt) slices, UNSIGNED4 trainSize) := FUNCTION
    batchPos := slices(nodeId = node AND sliceId = 1)[1].batchPos;
    //batchPos := ASSERT(slices(nodeId = node AND sliceId = 1), FALSE, 'batchPos = ' + batchPos)[1].batchPos;
    done := batchPos >= trainSize;
    RETURN done;
  END;

COMPRESS_UPDATES := IF(nNodes > 1, TRUE, FALSE);
LR_Progress_Factor := .75;
Batch_Size_Prog_Factor := .75;

// DATASET(sliceExt) doEpoch(DATASET(sliceExt) inWeights, UNSIGNED epochNum) := FUNCTION
    inWeights := InitWeightsExt;
    R_train := RandomizeSamples(trainDat); // Local operation
    // OUTPUT(R_train);
    zWeights := PROJECT(inWeights, TRANSFORM(RECORDOF(LEFT),
                        SELF.loss := 0,
                        SELF.batchPos := 1,
                        SELF := LEFT), LOCAL);
    epochNum := 1;
    noProgress := epochNum - zWeights[1].minEpoch - 1;
    maxNoProgress := MAX(zWeights[1].maxNoProg, noProgress);
    epochLR := IF(maxNoProgress > 0, lr * POWER(LR_Progress_Factor, maxNoProgress), lr);
    epochBatchSize := (UNSIGNED4)IF(maxNoProgress > 0, miniBatchSize * POWER(Batch_Size_Prog_Factor, maxNoProgress),
                        miniBatchSize);
    batchSize1 := MIN(trainSize, epochBatchSize);
    // LOOP for each mini-batch

    // DATASET(sliceExt) doBatch(DATASET(sliceExt) inWeights2, UNSIGNED batchNum) := FUNCTION
    // Walk through the randomized samples taking batchSize at a time.
    inWeights2 := zWeights;
    batchNum := 1;
    firstW := inWeights2(nodeId = node AND sliceId = 1)[1];
    batchPos := firstW.batchPos;
    loss := firstW.loss;
    B_train := CHOOSEN(R_train, batchSize1, batchPos, LOCAL);
    nTrainRecs := TABLE(B_train, {cnt := COUNT(GROUP)}, LOCAL)[node + 1].cnt;
    // OUTPUT(nTrainRecs);
    // Do the gradient descent, and return the weight updates
    // Decrease learning rate as we proceed so that we can better converge.
    avgLoss := loss / (nNodes * batchSize1 * batchNum * (1 + negSamp));
    progress := calcProgress(avgLoss);
    adjLR := MAX((1 - progress), .1) * epochLR;
    // Train the neural network and get updated weights.
    tempPrint := Std.System.Log.addWorkunitInformation('batchPos = ' + batchPos +
            ', rtrain = ' + COUNT(R_train) +
            ', weightSlots = ' +  w.nWeightSlots +
            ', sliceSize = ' + sliceSize + ', inWeights2 = ' + COUNT(inWeights2) + ', nWeights = ' + COUNT(inWeights2[1].weights));
    //DATASET(sliceExt) wUpdates := WHEN(int.svTrainNN(inWeights2, B_train, sliceSize, w.nWeightSlots,
    //      shape[1], shape[2], nTrainRecs, adjLR, negSamp), tempPrint); // C++
    DATASET(sliceExt) wUpdates := int.svTrainNN(inWeights2, B_train, sliceSize, w.nWeightSlots,
            shape[1], shape[2], nTrainRecs, adjLR, negSamp); // C++
    
    // Distribute the updates by sliceId for rollup.  Compress the updates if needed.
    wUpdatesC := w.compressWeights(wUpdates);
    OUTPUT(wUpdatesC, NAMED('wUpdatesC'));
    OUTPUT(wUpdatesC(SIZEOF(cweights) > 0));
    wUpdatesDC := DISTRIBUTE(wUpdatesC(COUNT(cweights) > 0), sliceId);
     OUTPUT(wUpdatesDC);
    wUpdatesD := DISTRIBUTE(wUpdates, sliceId);
    // Now apply the updates on the nodes assigned by sliceId, and then re-replicate to all nodes.
    newWeightsC := rollUpdatesC(inWeights2(sliceId % nNodes = node), wUpdatesDC);
    newWeightsN := rollUpdates(inWeights2(sliceId % nNodes = node), wUpdatesD);
    newWeights0 := IF(COMPRESS_UPDATES, newWeightsC, newWeightsN);
    // Continue the loop with replicated weights.
    newWeights1 := PROJECT(newWeights0, TRANSFORM(RECORDOF(LEFT),
                                                    SELF.batchPos := batchPos + nTrainRecs,
                                                    SELF := LEFT), LOCAL);
    firstW2 := newWeights1(nodeId = node AND sliceId = 1)[1];
    loss2 := firstW2.loss;
    status := Std.System.Log.addWorkunitInformation('Status: Initial Loss = ' +
                ROUND(loss2 / (nNodes * batchSize * (1 + negSamp)), 6));
    newWeights := IF(epochNum = 1 AND batchNum = 1, WHEN(newWeights1, status), newWeights1);
    // RETURN newWeights;
    // END;
// epochWeights0 := LOOP(zWeights, TRUE, NOT isEpochDone(ROWS(LEFT), trainSize) , doBatch(ROWS(LEFT), COUNTER));
// OUTPUT(epochWeights0); // ISSUE **



//epochWeights0 := LOOP(zWeights, nBatches, doBatch(ROWS(LEFT), COUNTER));
// Mark the loss information in each slice.
// isBest(SliceExt rec) := rec.loss < rec.minLoss;
// epochWeights := PROJECT(epochWeights0, TRANSFORM(RECORDOF(LEFT),
//                             SELF.minLoss := IF(isBest(LEFT), LEFT.loss, LEFT.minLoss),
//                             SELF.minEpoch := IF(isBest(LEFT), epochNum, LEFT.minEpoch),
//                             SELF.maxNoProg := maxNoProgress,
//                             SELF := LEFT), LOCAL);
// firstW := epochWeights(nodeId = node AND sliceId = 1)[1];
// loss := firstW.loss;
// avgLoss := loss / (COUNT(trainDat) * (1 + negSamp));
// progress := calcProgress(avgLoss);
// adjLR := MAX((1 - progress), .1) * epochLR;
// minEpoch := firstW.minEpoch;
// status := Std.System.Log.addWorkunitInformation('Status: ' + 'Epoch = ' + epochNum +
//                         ', Progress = ' + (DECIMAL5_2) (progress * 100) +
//                         '%, Loss = ' + (DECIMAL6_6)avgLoss +
//                         ', minEpoch = ' + minEpoch +
//                         ', LR = ' + (DECIMAL6_6)adjLR +
//                         ', batchSize = ' + batchSize);
// RETURN WHEN(epochWeights, status);
// END;

// finalWeights := LOOP(InitWeightsExt, TRUE, NOT isConverged(ROWS(LEFT), COUNTER, COUNT(trainDat)), doEpoch(ROWS(LEFT), COUNTER));
// firstW := finalWeights(nodeId = node AND sliceId = 1)[1];
// loss := firstW.loss;
// OUTPUT(firstW);