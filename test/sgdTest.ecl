/*##############################################################################
## HPCC SYSTEMS software Copyright (C) 2019 HPCC Systems.  All rights reserved.
############################################################################## */
#option ('clusterSize', 5);

/**
  * Unit test for the Synchronous Gradient Descent (SGD) subsystem
  */
IMPORT $.^.Types;
IMPORT $.^.internal AS int;

Slice := Types.slice;
SliceExt := Types.sliceExt;
TrainingDat := Types.TrainingDat;


shape := [10, 5, 10];

w := int.Weights(shape);

mySGD := int.SGD(shape, 10000, 10, .5, 1);

train := DATASET([{1,[2,3,4]},{2,[1,5,7]},{8,[2,3,9]}], TrainingDat);
//train := DATASET([{1,2},{1,3}], TrainingPair);

slices := w.initWeights;

SE := w.toSliceExt(slices);


//wNew := mySGD.getWeightUpdates(SE, train, w.sliceSize, 10, 5, 100, .1);

OUTPUT(slices, ALL, NAMED('OrigWeights'));
OUTPUT(SE, ALL, NAMED('OrigWeightsExt'));

sliceSize := w.sliceSize;
OUTPUT(sliceSize, NAMED('sliceSize'));

//wNew := mySGD.getWeightUpdates(SE, train, w.sliceSize, 10, 5, 100, .1);
wNew := mySGD.Train_Dupl(train);
wCount := COUNT(wNew[1].weights);
OUTPUT(wCount, NAMED('OutWeightCount'));
OUTPUT(wNew, NAMED('NewWeights'));

wNew2 := mySGD.Train(train);
OUTPUT(wNew2, NAMED('NewWeights2'));

temprec := RECORD
  UNSIGNED sliceId;
  SET OF REAL8 wi;
  SET OF REAL8 wo;
END;
testrec := RECORD
 UNSIGNED wNum;
 REAL8 wi;
 REAL8 wo;
 BOOLEAN error := 0;
END;

t1 := JOIN(slices, wNew, LEFT.sliceId = RIGHT.sliceId, TRANSFORM(temprec,
                                            SELF.wi := LEFT.weights, SELF.wo := RIGHT.weights,
                                            SELF := LEFT), LOCAL);

t2 := NORMALIZE(t1, sliceSize, TRANSFORM(testrec, SELF.wNum := COUNTER + (LEFT.sliceId - 1) * sliceSize,
                      SELF.wi := LEFT.wi[COUNTER], SELF.wo := LEFT.wo[COUNTER],
                      SELF.error := (SELF.wi != SELF.wo)));
OUTPUT(t2, ALL, NAMED('WeightsCompare'));