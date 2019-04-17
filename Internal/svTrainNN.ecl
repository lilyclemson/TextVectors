/*##############################################################################
## HPCC SYSTEMS software Copyright (C) 2019 HPCC Systems.  All rights reserved.
############################################################################## */
IMPORT $.^.Types;
SliceExt := Types.SliceExt;
TrainingDat := Types.TrainingDat;
/**
  * Neural Network Training for SentenceVectors.
  * <p>Train specialized SentenceVector neural network given a batch of training
  * data. Takes in weights as a set of weights slices (SliceExt), and returns
  * a set of weight adjustments, also formatted as slices.
  * @param wts The weights slices.
  * @param train The batch of training data formatted as a main word and set of context words.
  * @param slicesize The maximum number of weights in a slice.
  * @param nWeights The total number of weights across all slices.
  * @param numwords The number of words in the vocabulary.
  * @param dim The dimensionality of the vectors being trained
  * @param mbsize The number of training records in the mini-batch.
  * @param lr The learning rate to use for this batch.
  * @param negsamp The number of negative samples to choose for each main word.
  * @return weight updates as DATASET(SliceExt).  Note that these are additive 
  *         changes to the weights, not the final weight values.
  */
EXPORT STREAMED DATASET(SliceExt) svTrainNN(STREAMED DATASET(SliceExt) wts,
          STREAMED DATASET(trainingDat) train,
          UNSIGNED4 slicesize,
          UNSIGNED4 nWeights,
          UNSIGNED4 numwords,
          UNSIGNED4 dim,
          UNSIGNED4 mbsize,
          REAL lr,
          UNSIGNED4 negsamp) := EMBED(C++ : activity)
  #include <stdlib.h>
  #include <stdio.h>
  #include <stdint.h>
  #include <string.h>
  #include <math.h>
  #include <assert.h>
  /**
  * Training Point Structure
  */
  struct tp
  {
    uint32_t main;
    uint32_t contxCount;
    uint32_t * contx;
  };
  /**
    * Specialized Neural Network with 3 Layers.
    * The input layer is the size of the vocabulary, the hidden layer is the
    * size of the word vectors, and the output layer is the size of the vocabulary.
    * Training data is presented as
    * a tuple of a main word (id), and a list of context words (ids). Context
    * is generally the words that were near or in the same sentence as the main
    * word.
    * The context words are simultaneously active at the input layer, and the
    * main word is the training target at the output layer.
    * The hidden layer weights represent the word vectors.
    * The hidden layer is linear (no activation function)
    * and the output layer uses a sigmoid activation function.
    */
  class NeurNet
  {
    double* w_; // Weights
    double* wUpdates_; // Weight updates (output)
    tp* t_; // Training Points
    uint32_t tsize_; // Number of training recs
    uint32_t wc_;  // Number of words in vocab
    uint32_t dim_; // Length of word vector
    double* hidden_;  // Forward pass results for hidden layer
    uint32_t negSamp_; // Number of negative samples
    double lr_; // Learning Rate
    float loss_; // Accumulated loss
    public:
      NeurNet()
      {
      }
      NeurNet(double * wts, double * wUpdates, uint32_t wc, uint32_t dim,
              double lr, uint32_t negSamp)
      {
        w_ = wts;
        wUpdates_ = wUpdates;
        wc_ = wc;
        dim_ = dim;
        hidden_ = (double *)rtlMalloc(dim * sizeof(double));
        lr_ = lr;
        negSamp_ = negSamp;
        loss_ = 0;
      }
      float getLoss()
      {
        return loss_;
      }
      // Convert triple index (l, j, i) to a flat index into weights
      // l is the layer [0, 1]
      // j,i is the weight from node j in layer l to node i in layer l+1
      uint32_t toFlatIndx(uint16_t l, uint32_t j, uint32_t i)
      {
        uint32_t iSize, indx;
        if (l == 0)
          iSize = dim_;
        else
          iSize = wc_;
        indx = l * dim_ * wc_;
        indx += j * iSize;
        indx += i;
        return indx;
      }
      /**
        * Get negative samples probabilistically using the word frequency
        * table.  This is not used for now.
        */
//      uint32_t getNegativeProb(uint32_t target)
//      {
//        uint32_t r;
//        float prob;
//        do
//        {
//          // Pick a random word
//          r = rand() % wc_;
//          // Look up the sampling probability in the negatives table
//          prob = negTab_[r];
//          // Skip with probability (1 - prob)
//          if (rand() / RAND_MAX > prob)
//            r = target; // Force a skip
//        } while (r == target); // Don't return the target.
//        return r;
//      }
      /**
        * Randomly choose a word for negative sampling.  This does not
        * respect the probability distribution of the words in the corpus,
        * but is adequate and much more efficient.
        */ 
      uint32_t getNegative(uint32_t target)
      {
        uint32_t r;
        do
        {
          r = rand() % wc_;
        } while (r == target);
        return r;
      }
      /**
        * Calculate the neural network Sigmoid function for forward
        * propagation.
        */
      double sigmoid(double x)
      {
        double y;
        if (x > 25)
          y = 1.0;
        else if (x < -25)
          y = 0.0;
        else
          y = 1.0 / (1.0 + exp(-x));
        return y;
      }
      /** Calculate the loss of a given sample.
        * Uses linear loss rather than log loss as it
        * is easier to evaluate and understand.
        * label is either 1 or 0, and represents to
        * target value.
        */
      float calcLoss(float score, uint32_t label)
      {
        float loss;
        if (label == 1)
          loss = 1 - score;
          //loss = -log(score + .00001);
        else
          loss = score;
          //loss = -log((1-score) + .00001);
        return loss;
      }
      /**
        * Forward and Back Propagation to adjust weights.
        * Processes a single training sample.
        * For each sample, processes the positive case
        * (i.e. the sample) as well as a set of randomly
        * selected negative samples.  Saves the weights
        * and the weight updates (deltas).
        * First, propagate the inputs (the context variables)
        * to the output to compute the output (score).
        * Then back-propagate errors (i.e. score - label)
        * back to the output weights and the input weights.
        */
      void fbPropagate(tp& train)
      {
        uint32_t main = train.main - 1; // Map 1-based ECL index to 0-based
        uint32_t contxCount = train.contxCount;
        uint32_t * contx = train.contx;
        double contxCountInv = 1.0 / (double)contxCount;
        // FORWARD PATH ***
        uint32_t j, i, n, windx, target, label;
        double cum, score, alpha, grad, wUpdate;
        // Zero the hidden layer.
        for (uint32_t i = 0; i < dim_; i++)
          hidden_[i] = 0;
        // Update the hidden layer outputs based the context words.
        for (j = 0; j < contxCount; j++)
        {
          uint32_t cx = contx[j] - 1; // Map 1-based ECL index to 0-based
          for ( i = 0; i < dim_; i++)
          {
              windx = toFlatIndx(0, cx, i);
              hidden_[i] += w_[windx] * contxCountInv;
          }
        }
        // Accumulate the final layer output for the 
        // 'main' output node as well as a random selection of
        // 'negative' words.
        for (n = 0; n <= negSamp_; n++) // Note executes negSamp + 1 times
        {
          // On the first iteration, process the positive sample given
          if (n == 0)
          {
            target = main;
            label = 1;
          }
          else
          // On subsequent iterations, process randomly selected negative
          // samples.
          {
            target = getNegative(main);
            label = 0;
          }
          cum = 0;
          // Propagate the hidden layer to the output layer
          for (j = 0; j < dim_; j++)
          {
            windx = toFlatIndx(1, j, target);
            cum += w_[windx] * hidden_[j];
          }
          // Apply the sigmoid to the accumulated (weight * hidden)
          // to get the final output value.
          double score = sigmoid(cum);

          // Calculate the loss, and add it to the total loss.
          loss_ += calcLoss(score, label);

          // BACK PATH ***
          // Compute the gradient at the output layer
          // = learning rate * (label - score) * derivative(lossFunc)
          // derivative(lossFunc) = score * (1-score)
          double grad2 = (float(label) - score) * score * (1-score);
          // Apply the gradient to the output weights
          // and the input weights.
          for (j = 0; j < dim_; j++)
          {
            windx = toFlatIndx(1, j, target);
            // Update output (i.e. L2) weights
            // weight update = grad * hidden * weight
            wUpdate = (hidden_[j] * grad2 * lr_);
            w_[windx] += wUpdate;
            wUpdates_[windx] += wUpdate;

            // Layer 1 gradient is Layer 2 gradient *  w[j, target]
            double grad1 = grad2 * w_[windx];

            // Update input (i.e. L1) weights for each context word
            i=j;
            for (uint32_t c = 0; c < contxCount; c++)
            {
              uint32_t cx = contx[c] - 1; // Map 1-based ECL index to 0-based
              windx = toFlatIndx(0, cx, i);
              // Weight update is learning rate * gradient * input (i.e. 1/contextCount).
              wUpdate = lr_ * grad1 * contxCountInv;
              w_[windx] += wUpdate;
              wUpdates_[windx] += wUpdate;
            }
          }
        }
      }
      /**
        * Train the neural network and output the weight updates.
        * Runs all of the training samples.
        */
      void Train(tp * train, uint32_t trainSize)
      {
        t_ = train;
        tsize_ = trainSize;
        for (uint32_t i = 0; i < trainSize;i++)
        {
          // Do the Forward and Back Propagation to train the weights
          fbPropagate(train[i]);
        }
      }
      // NN Destructor
      ~NeurNet()
      {
        rtlFree(hidden_);
      }
  };
  /**
    * Process the returned stramed dataset, one slice at a time.
    * Before returning the first slice, retrieve the weights and the
    * training data from the input streams, and then run the neural
    * network to update the weights.  The returned dataset contains
    * the weight updates (i.e. deltas), and not the final weights.
    */
  class MyStreamInlineDataset : public RtlCInterface, implements IRowStream
  {
    double * fW_; // Flattened weights
    double * fU_; // Updated weights
    tp* fT_;     // Flattened training data
    uint32_t fT_count_;  // Number of training records
    NeurNet *nn_;       // Neural network instance
    bool calculated_ = false; // Flag so that we know we've already
                              // calculated the weight updates (i.e. before
                              // returning the first slice.
    uint16_t nodeId_ = 0; // Current node #
    uint16_t outSlice_ = 0; // Current outSlice index.
    uint32_t weightCount_; // The total number of weights
    float loss_; // The computed Loss
    float minLoss_; // The lowest lost we've seen so far (passthrough)
    uint32_t minEpoch_; // Epoch of minimum loss
    uint32_t maxNoProg_; // Maximum epochs with no progress (passthrough)
    uint64_t batchPos_; // The current position within the full training set (passthrough)
  public:
    MyStreamInlineDataset(IEngineRowAllocator * _resultAllocator, IRowStream * _wts, IRowStream * _train,
          uint32_t _slicesize, uint32_t _nweights,
          uint32_t _numwords, uint32_t _dim, uint32_t _mbsize, float _lr, uint32_t _negsamp)
         : resultAllocator(_resultAllocator), wts(_wts), train(_train), slicesize(_slicesize),
                           nweights(_nweights),
                           numwords(_numwords), dim(_dim), mbsize(_mbsize), negsamp(_negsamp), lr(_lr)
    {
       weightCount_ = nweights;
       // Allocate space for the weights and the weight updates
       fW_ = (double *)rtlMalloc(weightCount_ * sizeof(double)); // Flattened weight array
       fU_ = (double *)rtlMalloc(weightCount_ * sizeof(double)); // Flattened weight update array
       // Initialize weight updates to zero
       for (uint32_t i = 0; i < weightCount_;i++)
          fU_[i] = 0.0;
       fT_ = (tp *)rtlMalloc(mbsize * sizeof(tp)); // Flattened training data array
       fT_count_ = 0;
       loss_ = 0;
       minLoss_ = 0;
       minEpoch_ = 0;
       batchPos_ = 0;
    }
    /**
      * Destructor for streamed dataset
      */
    ~MyStreamInlineDataset()
    {
      rtlFree(fW_);
      for (uint32_t i = 0; i < fT_count_; i++)
      {
        rtlFree(fT_[i].contx);
      } 
      rtlFree(fT_);
      rtlFree(fU_);
    }
    RTLIMPLEMENT_IINTERFACE
    /**
      * Return one row of the results (i.e. one slice of updates)
      */
    virtual const void *nextRow() override
    {
        // Before we output the first record, read all the input streams
        // and put each into a flat array.
        if (!calculated_)
        {
          // First time through.  Do the heavy lifting here:
          // - Read the input datasets
          // - Run the neural network training
          // - Calculate the weight updates 
          while (true)
          {
             // Process all the input slices and put in flattened array (fW_)
             const byte * next = (const byte *)wts->nextRow();
             if (!next) break;
             byte * pos = (byte *)next;
             nodeId_ = *(uint16_t *)pos;  // Save the node id.  Should be the same for all records.
             pos += sizeof(uint16_t);
             uint16_t sliceId = *(uint16_t *)pos;
             pos += sizeof(uint16_t);
             loss_ = *(float *)pos;  // Loss from previous
             pos += sizeof(float);
             minLoss_ = *(float *) pos; // Minimum observed loss
             pos += sizeof(float);
             minEpoch_ = *(uint32_t *) pos; // Epoch of minimum observed loss
             pos += sizeof(uint32_t);
             maxNoProg_ = *(uint32_t *) pos; // Maximum Epochs with no progress
             pos += sizeof(uint32_t);
             batchPos_ = *(uint64_t *) pos; // Current position in training set
             pos += sizeof(uint64_t);
             pos += sizeof(bool) + sizeof(uint32_t); // pass over SET's all flag and length
             uint32_t offset = (sliceId-1) * slicesize * sizeof(double);
             memcpy(((byte *)fW_ + offset), pos, slicesize * sizeof(double));
             rtlReleaseRow(next);
          }
          // Get the training records and store in flattened training array (fT_).
          uint32_t i = 0, j = 0;
          while (true)
          {
            const byte * next = (const byte *)train->nextRow();
            if (!next) break;
            byte * pos = (byte *) next;
            // Add to flattened training set
            fT_[fT_count_].main = *(uint32_t *) pos; // Main word
            pos += sizeof(uint32_t);
            // Set of context words.  Bypass 'ALL' flag.
            pos += sizeof(bool);
            // Calculate the number of context words.
            uint32_t contxLen = *(uint32_t *) pos;
            uint32_t contxCount = contxLen / sizeof(uint32_t);
            fT_[fT_count_].contxCount = contxCount;
            // Allocate storage for the context array
            uint32_t * context = (uint32_t *)rtlMalloc(contxLen);
            fT_[fT_count_].contx = context;
            pos += sizeof(uint32_t);
            // Copy the context wordIds into the array
            memcpy((byte *)context, pos, contxLen);
            rtlReleaseRow(next);
            fT_count_++;
          }
          // Now we have the weights and the training pairs.  Create a neural network
          // and train it on this data.
          nn_ = new NeurNet(fW_, fU_, numwords, dim, lr, negsamp);
          nn_->Train(fT_, fT_count_);
          // Calculate the overall loss for this batch
          loss_ = nn_->getLoss();
          // Delete the neural network.  We're done with it.
          delete nn_;
          // Mark so we know we've already calculated the results.
          calculated_ = true;
        } // end if (!calculated)
        // Now return the next slice.
        // Copy the weight updates into slices to return.
        uint32_t returnSize;
        byte * row;
        RtlDynamicRowBuilder rowBuilder(resultAllocator);
        if ((outSlice_ + 1) * slicesize <= weightCount_)
        {
          uint32_t returnSize = sizeof(uint16_t)*2 + sizeof(float) + sizeof(float) + sizeof(uint32_t) +
                                  sizeof(uint32_t) + sizeof(uint64_t) + sizeof(bool) + sizeof(uint32_t) +
                                  slicesize * sizeof(double);
          // Create a return row.  Need to use ensureCapacity for variable lenghth
          // data.
          row = rowBuilder.ensureCapacity(returnSize, NULL);
          byte *pos = row;
          *(uint16_t *)pos = nodeId_;  // nodeId (pass through)
          pos += sizeof(uint16_t);
          *(uint16_t *)pos = outSlice_ + 1; // Slice Id
          pos += sizeof(uint16_t);
          *(float *)pos = loss_; // Computed Loss
          pos += sizeof(float);
          *(float *)pos = minLoss_; // Min Loss (pass through)
          pos += sizeof(float);
          *(uint32_t *)pos = minEpoch_; // Min Epoch (pass through)
          pos += sizeof(uint32_t);
          // Pass through the maxNoProg
          *(uint32_t *)pos = maxNoProg_;
          pos += sizeof(uint32_t);
          // Pass through the current batch position
          *(uint64_t *)pos = batchPos_;
          pos += sizeof(uint64_t);
          *(bool *)pos = (bool)false; // All flag
          pos += sizeof(bool);
          *(uint32_t *)pos = (uint32_t)slicesize * sizeof(double); // Length of SET
          pos += sizeof(uint32_t);
          // Copy the weight updates into the row.
          memcpy(pos, (byte*)fU_ + outSlice_ * slicesize * sizeof(double), slicesize * sizeof(double));
          outSlice_++; // Increment slice number.
          return rowBuilder.finalizeRowClear(returnSize);
        }
        else
        {
          // No more slices to output.
          return NULL;
        }
    }
    virtual void stop() override
    {
    }

  protected:
    Linked<IEngineRowAllocator> resultAllocator;
    IRowStream * wts;
    IRowStream * train;
    uint32_t slicesize, nweights, numwords, dim, mbsize, negsamp;
    float lr;
  };
  #body
  // Create the returned streamed dataset.
  return new MyStreamInlineDataset(_resultAllocator, wts, train, slicesize,
                                    nweights, numwords, dim, mbsize, lr, negsamp);
ENDEMBED;
