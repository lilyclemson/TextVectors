/*##############################################################################
## HPCC SYSTEMS software Copyright (C) 2019 HPCC Systems.  All rights reserved.
############################################################################## */
#option ('clusterSize', 3);
/**
  * Unit test for the Weights subsystem
  */
IMPORT $.^.Types;
IMPORT $.^.internal as int;
IMPORT std.System.Thorlib;

Word := Types.Word;
WordInfo := Types.WordInfo;
sliceExt := Types.sliceExt;
CSlice := Types.CSlice;
slice := Types.slice;

nNodes := Thorlib.nodes();

w := int.weights([100, 10, 100]);

indxTests := DATASET([{1,1,1},
                  {2, 1, 1},
                  {1, 100, 10},
                  {2, 10, 100},
                  {1, 1, 10},
                  {1, 100, 1},
                  {2, 10, 1}], Types.wIndex);
indxTestOut := RECORD (Types.wIndex)
  UNSIGNED4 flatIndx;
  UNSIGNED2 newL;
  UNSIGNED4 newJ;
  UNSIGNED4 newI;
  BOOLEAN error;
END;

indxTestOut doIndxTest(Types.wIndex t) := TRANSFORM
  SELF.flatIndx := w.toFlatIndex(t.l, t.j, t.i);
  newIndx := w.fromFlatIndex(SELF.flatIndx);
  SELF.newL := newIndx.l;
  SELF.newJ := newIndx.j;
  SELF.newI := newIndx.I;
  SELF.error := SELF.newL != t.l OR SELF.newJ != t.j OR SELF.newI != t.i;
  SELF := t;
END;
indxResult := PROJECT(indxTests, doIndxTest(LEFT));

OUTPUT(indxResult, NAMED('IndexTest'));


OUTPUT(nNodes, NAMED('NodeCount'));

shape := RECORD
  SET OF UNSIGNED s;
END;
sliceSizeTests := DATASET([{[100, 10, 100]},
                    {[73, 10, 73]},
                    {[27, 5, 27]},
                    {[100000, 100, 100000]},
                    {[101, 11, 147]},
                    {[111201, 201, 111201]}], shape);
sliceSizeOut := RECORD(shape)
  UNSIGNED nWeights;
  UNSIGNED slicesPerNode;
  UNSIGNED sliceSize;
  UNSIGNED nSlices;
  UNSIGNED nWeightSlots;
END;

sliceSizeOut doSliceSizeTest(shape s) := TRANSFORM
  SELF.s := s.s;
  w := int.weights(s.s);
  SELF.nWeights := w.nWeights;
  SELF.slicesPerNode := w.slicesPerNode;
  SELF.sliceSize := w.sliceSize;
  SELF.nSlices := w.nSlices;
  SELF.nWeightSlots := w.nWeightSlots;
END;

sliceSizeResult := PROJECT(sliceSizeTests, doSliceSizeTest(LEFT));

OUTPUT(sliceSizeResult, NAMED('SliceSize'));

// Init Weights test

w2 := int.weights([73, 10, 73]);


slices2 := w2.initWeights;

OUTPUT(slices2, ALL, NAMED('InitialSlices'));

w3 := int.weights([27, 5, 27]);

slices3 := w3.initWeights;
slices3E := w3.toSliceExt(slices3);



OUTPUT(w3.sliceSize, NAMED('sliceSize3'));
OUTPUT(slices3, ALL, NAMED('slices3'));
OUTPUT(slices3E, ALL, NAMED('slices3E'));

testslices := DATASET([{1, 1, 0,0, 0, 0, 0, [0, .1, 0, 0, .2, 0, 0, .3]}, {1, 2, 0,0,0,0,0, [.1, 0, 0, 0, .2, .3, 0, 0]},
                        {1, 3, 0,0,0, 0,0,[0,0,0,0,.1,0,0,0]}], SliceExt);

CSlice compress(SliceExt s) := TRANSFORM
  SELF.cweights := w3.compressOne(s.weights, 8);
  SELF := s;
END;
cslices := PROJECT(testslices, compress(LEFT), LOCAL);

SliceExt decomp(CSlice cs) := TRANSFORM
  SELF.weights := w3.decompressOne(cs.cweights, 8);
  SELF := cs;
END;
slices := PROJECT(cslices, decomp(LEFT), LOCAL);
OUTPUT(cslices, ALL, NAMED('Compressed'));
OUTPUT(slices, ALL, NAMED('Decompressed'));

linear := w3.slices2Linear(slices);
OUTPUT(linear, NAMED('Linear'));
