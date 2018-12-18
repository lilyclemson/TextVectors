/*##############################################################################
## HPCC SYSTEMS software Copyright (C) 2019 HPCC Systems.  All rights reserved.
############################################################################## */
/**
  * Common Type definitions for TextVectors bundle
  */
EXPORT Types := MODULE
  /**
    * Type definition for a Word Id attribute
    */
  EXPORT t_WordId := UNSIGNED4;
  /**
    * Type definition for a Sentence Id attribute.
    */
  EXPORT t_SentId := UNSIGNED8;
  /**
    * Type definition for an attribute that can hold either a Word Id.
    * or a Sentence Id.
    */
  EXPORT t_TextId := t_SentId; // Can hold a word or sentence id
  /**
    * Type definition for a vector attribute (used for Word or Sentence Vectors).
    */
  EXPORT t_Vector := SET OF REAL8;
  /**
    * Type definition for the text of a Word.
    */
  EXPORT t_Word := STRING;
  /**
    * Type definition for the text of a Sentence.
    */
  EXPORT t_Sentence := STRING;
  /**
    * Enumeration of the record type for a Text Model.
    */
  EXPORT t_ModRecType := ENUM(UNSIGNED1, unused=0, word=1, sentence=2);

  /**
    * Dataset Record to hold a Sentence.
    * @field sentId The numeric record id for this sentence.
    * @field text The text content of the sentence.
    */
  EXPORT Sentence := RECORD
    t_SentId sentId;
    t_Sentence text;
  END;
  /**
    * Sentence Information record, including the sentence vector.
    * @field sentId The numeric record id for this sentence.
    * @field text The text content of the sentence.
    * @field vec The Text Vector for the sentence.
    */
  EXPORT SentInfo := RECORD
    t_SentId sentId;
    t_Sentence text;
    t_Vector vec := [];
  END;
  /**
    * Record to hold the discard probability of each word in the vocabulary
    * Currently unused.  Reserved for future.
		* @internal
		* @field nodeId The node on which this record resides.
		* @field probs The set of discard probabilities for each word
		*              in the vocabulary from 1 - N.
    */
  EXPORT NegativeRec := RECORD
    UNSIGNED2 nodeId;
    SET OF REAL4 probs;
  END;
  /**
    * Dataset Record to hold a Word.
    * @field id The numeric record id for this word.
    * @field text The text content of the word.
    */
  EXPORT Word := RECORD
    t_WordId id;
    t_Word text;
  END;
  /**
    * Dataset record to hold information about a word, including its vector,
    * number of occurrences in the corpus, and discard probability.
    * @field wordId The numeric record id for this word.
    * @field text The text content of the sentence.
    * @field occurs The number of times this word occurs in the Corpus.
    * @field pdisc The computed probability of discard, based on the frequency
    *               of the word in the corpus.
    * @field vec The Text Vector for the word.
    */
  EXPORT WordInfo := RECORD
    t_WordId wordId;
    t_Word text;
    UNSIGNED4 occurs := 1;
    REAL pdisc := 0;
    t_Vector vec := [];
  END;
  /**
    * Dataset record to hold a sentence or word vector.
    * @field id The record id for this vector.
    * @field vec The contents of the vector.
    */
  EXPORT Vector := RECORD
    UNSIGNED4 id;
    t_Vector vec;
  END;
  /**
    * Record definition for the TextVectors Model.
    * Text Model contains both the word and sentence vectors
    * for the trained corpus.
    * @field typ The type of the record -- Word or Sentence (see t_ModRecType above).
    * @field id The id of the word or sentence.
    * @field text The textual content of the item (word or sentence).
    * @field vec The vector for the word or sentence.
    */
  EXPORT TextMod := RECORD
    t_ModRecType typ;
    t_TextId id;
    t_Sentence text;
    t_Vector vec;
  END;
  /**
    * Weight index used for mapping
    * the neural network weights: (Layer, j, i).
    * L (layer) represents the neural network layer of the
    * weights (1 or 2) for a three-layer NN.
    * J represents the index of the node in layer L.
    * I represents the index of the node in layer L + 1.
    * weight(L=1, J=3, I=5) would be the weight between
    * Layer 1 (input layer) node 3 and Layer 2 (hidden layer)
    * node 5.
		* @internal
    * @field L Layer of the neural network.
    * @field J The node in layer L to which this weight applies.
    * @field I The node in Layer L+2 to which this weight applies.
    */
  EXPORT wIndex := RECORD
    UNSIGNED2 L;
    UNSIGNED4 J;
    UNSIGNED4 I;
  END;
  /**
    * Weights are organized as slices, one or more per node in the
    * cluster.  This provides a scalable mechanism for transferring
    * and manipulating Neural Network weights, since the entire set
    * can be quite large.  The number of weights in a TextVectors
    * neural network is 2 * nWords * vecLen.
		* @internal
    * @field sliceId The id of this slice.
    * @field weights The set of weights contained in the slice.
    */
  EXPORT Slice := RECORD
    UNSIGNED2 sliceId;
    t_Vector weights;
  END;
  /**
    * Extended Slice is a replicated form of weights used during Gradient
    * Descent (i.e. NN training).
    * Weight slices are assigned a nodeId and replicated to all nodes.
    * It also contains some bookkeeping information used by Gradient Descent
    * to track progress across batches.
		* @internal
    * @field nodeId The node on which this slice copy resides.
    * @field sliceId The id of this slice.
    * @field loss The total loss (i.e. error-level) incurred in processing an epoch.
    * @field minLoss The lowest epoch loss seen during this run.
    * @field minEpoch The epoch number that achieved the lowest loss (see minLoss above).
    * @field maxNoProg The longest number of epochs processed without achieving a new minimum loss.
    * @field batchPos The current record pointer into the training data at the end of the batch.
    * @field weights The set of weights contained in the slice.
    */
  EXPORT SliceExt := RECORD
    UNSIGNED2 nodeId;
    UNSIGNED2 sliceId;
    REAL4 loss;
    REAL4 minLoss;
    UNSIGNED4 minEpoch;
    UNSIGNED4 maxNoProg;
    UNSIGNED8 batchPos;
    t_Vector weights;
  END;
  /**
    * A compressed form of SliceExt used to carry weight updates
    * (rather than final weights).  The cweights attribute contains
    * a packed set of [<index>, <weight>], allowing for less storage
    * when updates are sparse (i.e. eliminates the zero cells).
		* @internal
    * @field nodeId The node on which this slice copy resides.
    * @field sliceId The id of this slice.
    * @field loss The total loss (i.e. error-level) incurred in processing an epoch.
    * @field minLoss The lowest epoch loss seen during this run.
    * @field minEpoch The epoch number that achieved the lowest loss (see minLoss above).
    * @field maxNoProg The longest number of epochs processed without achieving a new minimum loss.
    * @field batchPos The current record pointer into the training data at the end of the batch.
    * @field cweights The compressed set of weights contained in the slice.  Weights are compressed
    *                to pairs of {index, weight}, eliminating all zero updates for weights not changed.
    */
  EXPORT CSlice := RECORD
    UNSIGNED2 nodeId;
    UNSIGNED2 sliceId;
    REAL4 loss;
    REAL4 minLoss;
    UNSIGNED4 minEpoch;
    UNSIGNED4 maxNoProg;
    UNSIGNED8 batchPos;
    DATA cweights;
  END;
  /**
    * Training Data record.  Each training record consists of a main word, and a set of
    * context words (i.e. other words that were found in the same
    * sentence or vicinity.
		* @internal
    * @field main The id of the main word (i.e. the word to which the context applies).
    * @field context A set of context words -- words that were observed in the vicinity of the
    *                 main word.
    */
  EXPORT TrainingDat := RECORD
    UNSIGNED4 main;
    SET OF t_WordId context;
  END;
  /**
    * Holds a sentence as a set of words.
		* @internal
    * @field sentId The record id of the sentence.
    * @field words The set of words that make up the sentence.
    */
  EXPORT WordList := RECORD
    UNSIGNED4 sentId;
    SET OF t_Word words;
  END;
  /**
    * Used to return a set of closest items (words or sentences) for
    * each of a given set of items to match.
    * Closest contains the set of closest items, while Similarity is
    * the cosine similarity between the text sentence and each of the closest
    * items.  For example, Similarity[1] is the Cosine similarity between
    * Text and Closest[1].  Likewise for each of the K items in Closest and
    * Similarity.
    * @field id The id of the word or sentence.
    * @field text The text of the word or sentence
    * @field closest The text of the K closest words or sentence.
    * @field similarity The cosine similarity between this word / sentence
    *                   and each of the K closest words or sentences.  This
    *                   set corresponds 1:1 to the contents of the 'closest' field.
    */
  EXPORT Closest := RECORD
    t_TextId id;
    t_Sentence text;
    SET OF t_Sentence closest;
    SET OF REAL similarity;
  END;
  /**
    * Record to hold the return values from GetTrainStats function. This records describes
    * the set of parameters used for the current training session.
    * @field vecLen The dimensionality of the word and sentence vectors.
    * @field nWeights The number of weights needed to train the word vectors.
		* @field nSlices The number of slices used to hold the weights.
    * @field sliceSize The number of weights in each weight Slice.
    * @field nWords The number of words in the vocabulary including N-Grams.
    * @field nSentences The number of sentences in the Corpus.
    * @field maxNGramSize The maximum N-Gram size to consider.
    * @field nEpochs The maximum number of epochs for which to train.  Zero (default)
    *                means auto-compute.
    * @field negSamples The number of negative samples used in training for each training
    *                   sample.
    * @field batchSize The batch size used to train the vectors.  Zero (default) indicates
    *                   auto-compute.
    * @field minOccurs The minimum number of occurrences in the Corpus in order for a word
    *                   to be considered part of the vocabulary.
    * @field maxTextDist The maximum number of edits (in edit distance) to make in matching
    *                     a previously unseen word to a word in the vocabulary.
    * @field maxNumDist The maximum numeric distance to consider one previously unseen
    *                     number a match for a number in the vocabulary.
    * @field discardThreshold Words with frequency below this number are never discarded from
    *                     training data.  Words with frequency above this number are stochastically
    *                     sampled, based on their frequency.
		* @field learningRate The learning rate used to train the Neural Network.
    * @field upb Updates per batch.  The approximate number of weights that are updated across all
    *            nodes during a single batch.
    * @field upbPerNode Updates per batch per node.  The number of weights updated by each node
    *           during a single batch.
    * @field updateDensity The proportion of weights updated across all nodes during a single batch.
    * @field udPerNode The proportion of weights updated by a single node during a single batch.
    */
  EXPORT TrainStats := RECORD
    UNSIGNED4 vecLen := 0;
    UNSIGNED4 nWeights := 0;
    UNSIGNED4 nSlices := 0;
    UNSIGNED4 sliceSize := 0;
    UNSIGNED4 nWords := 0;
    UNSIGNED8 nSentences := 0;
    UNSIGNED4 maxNGramSize := 0;
    UNSIGNED4 nEpochs := 0;
    UNSIGNED4 negSamples := 0;
    UNSIGNED4 batchSize := 0;
    UNSIGNED4 minOccurs := 0;
    UNSIGNED4 maxTextDist := 0;
    UNSIGNED4 maxNumDist := 0;
    REAL4 discardThreshold := 0;
    REAL4 learningRate := 0;
    UNSIGNED4 upb := 0;
    UNSIGNED4 upbPerNode := 0;
    REAL4 updateDensity := 0;
    REAL4 udPerNode := 0;
  END;
  /**
    * Working structure allowing a separate record
    * for each word in a sentence, along with its vector.
		* @internal
		* @field sentId The unique ID for the word.
		* @field text The text of the word.
		* @field wordVec The vector associated with that word.
    */
  EXPORT wordExt := RECORD
    UNSIGNED8 sentId;
    t_Word text;
    t_Vector wordVec := []; // Gets assigned later
  END;
END;
