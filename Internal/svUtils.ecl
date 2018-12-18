/*##############################################################################
## HPCC SYSTEMS software Copyright (C) 2019 HPCC Systems.  All rights reserved.
############################################################################## */
IMPORT $.^.Types;

t_Vector := Types.t_Vector;
/**
  * Various utility functions used by TextVectors
  */
EXPORT svUtils := MODULE
  /**
    * Normalize a vector by dividing by the its length to create a unit vector
    */ 
  EXPORT t_Vector normalizeVector(t_Vector vec) := EMBED(C++)
    double * dVec = (double *) vec;
    __lenResult = lenVec;
    __isAllResult = FALSE;
    double * result = (double *)rtlMalloc(__lenResult);
    __result = (void *) result;
    uint16_t vecLen = lenVec / sizeof(double);
    uint16_t i;
    double norm = 0.0;
    for (i = 0; i < vecLen; i++)
    {
      double cell = dVec[i];
      norm += cell * cell;
    }
    norm = sqrt(norm);
    if (norm < 1e-8)
      // Avoid scaling vectors with tiny magnitude.  Leave the values at close to zero.
      norm = 1;
    for (i = 0; i < vecLen; i++)
    {
      result[i] = dVec[i] / norm;
    }
  ENDEMBED;

  /**
    * Calculate a Sentence Vector by taking the average of the word vectors for all words
    * in the sentence.
    * @param wordvecs A concatenated set of vectors for all the words in the sentence.
    * @param veclen The length of each word vector and the resulting sentence vector.
    */
  EXPORT t_Vector calcSentVector(t_Vector wordvecs, UNSIGNED2 veclen) := EMBED(C++)
    #include <assert.h>
    #body
    const double * items = (double *) wordvecs;
    assert(veclen > 0);
    assert(lenWordvecs % (veclen * sizeof(double)) == 0);
    uint32_t nItems = lenWordvecs / (veclen * sizeof(double));
    __lenResult = veclen * sizeof(double);
    __isAllResult = false;
    uint16_t item, i;
    uint32_t indx;
    double norm;
    double * result = (double *) rtlMalloc(__lenResult);
    __result = (void *) result;
    // Initialize result to zero
    for (i = 0; i < veclen; i++)
      result[i] = 0.0;
    // Loop over the word vector items
    for (item = 0; item < nItems; item++)
    {
      // Now accumulate the sum of word vectors in result
      // Word vectors are unit vectors, so no normalization of them is needed
      // We don't need to average them because we are going to normalize them at the
      // end, and normalize(SUM(wordVecs)) is equivalent to normalize(AVG(wordVecs))
      for (i = 0; i < veclen; i++)
      {
        indx = item * veclen + i;
        result[i] += items[indx];
      }
    }
    // Normalize the result
    norm = 0;
    for (i = 0; i < veclen; i++)
    {
      double cell = result[i];
      norm += cell * cell;
    }
    norm = sqrt(norm);
    if (norm < 1e-8)
      // Avoid scaling vectors with tiny magnitude.  Leave the values at close to zero.
      norm = 1;
    for (i = 0; i < veclen; i++)
    {
      result[i] = result[i] / norm;
    }
  ENDEMBED;
  /**
    * Cosine similarity
    *
    * a and b are unit vectors.  Theta is the angle between vectors.
    * Cosine similarity is Cos(theta).
    *
    * Cos(theta) = (a . b) / (L2Norm(a) * L2Norm(b))
    *
    * Note: a . b = L2Norm(a) * L2Norm(b) * Cos(theta)
    * Since we assume the inputs to be unit vectors, the norms will be 1.
    * We therefore simplify the calculation to a . b.
    */
  EXPORT REAL8 cosineSim(t_Vector a_in, t_Vector b_in, UNSIGNED4 veclen) := EMBED(C++)
    #body
    double * a = (double*)a_in;
    double * b = (double*)b_in;
    double adotb = 0;
    for (uint32_t i = 0; i < veclen; i++)
    {
      adotb += a[i] * b[i];
    }
    return adotb;;
  ENDEMBED;
  /**
    * Returns TRUE if a string represents a number (integer).  Otherwise FALSE.
    */
  EXPORT BOOLEAN isNumeric(STRING instr) := EMBED(C++)
    #body
    uint32_t slen = lenInstr;
    for (uint i = 0; i < slen; i++)
    {
      if (!isdigit(instr[i]))
        return false;
    }
    return true;
  ENDEMBED;
  /**
    * Calculates the numeric distance between two numeric strings as ABS(n1 - n2).
    */
  EXPORT UNSIGNED4 numDistance(VARSTRING str1, VARSTRING str2) := EMBED(C++)
    #body
    int32_t n1, n2, dist;
    n1 = atol(str1);
    n2 = atol(str2);
    dist = abs(n1 - n2);
    return (uint32_t) dist;
  ENDEMBED;
  /**
    * Implements vec1 + (vec2 * multiplier)
    * Allows (potentially) scaled addition of vectors as well as subtraction (using a negative
    * multiplier.
    */
  EXPORT t_Vector addVecs(t_Vector vec1, t_Vector vec2, UNSIGNED4 multiplier = 1) := EMBED(C++)
    #body
    uint32_t vecLen = lenVec1 / sizeof(double);
    uint32_t i;
    __lenResult = lenVec1;
    __isAllResult = false;
    double * result = (double *) rtlMalloc(__lenResult);
    __result = (void *) result;
    double * vec1N = (double *) vec1;
    double * vec2N = (double *) vec2;
    for (i = 0; i < vecLen; i++)
    {
      result[i] = vec1N[i] + (vec2N[i] * multiplier);
    }
  ENDEMBED;
END;