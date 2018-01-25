package com.uark.ml;

// ----------------------------------------------------------------
// The contents of this file are distributed under the CC0 license.
// See http://creativecommons.org/publicdomain/zero/1.0/
// ----------------------------------------------------------------

import java.util.*;

abstract class SupervisedLearner {
    /// Return the name of this learner
    abstract String name();

    /// Train this supervised learner
    abstract void train(Matrix features, Matrix labels);

    /// Make a prediction
    abstract Vec predict(Vec in);

    /// Measures the misclassifications with the provided test data
    int countMisclassifications(Matrix features, Matrix labels) {
        if (features.rows() != labels.rows())
            throw new IllegalArgumentException("Mismatching number of rows");
        int mis = 0;
        for (int i = 0; i < features.rows(); i++) {
            Vec feat = features.row(i);
            Vec pred = predict(feat);
            Vec lab = labels.row(i);
            for (int j = 0; j < lab.size(); j++) {
                if (pred.get(j) != lab.get(j))
                    mis++;
            }
        }
        return mis;
    }

    double sumSquaredError(Matrix features, Matrix labels) {
        if (features.rows() != labels.rows())
            throw new IllegalArgumentException("Mismatching number of rows");
        double error = 0;
        for (int i = 0; i < features.rows(); i++) {
            Vec feat = features.row(i);
            Vec pred = predict(feat);
            Vec lab = labels.row(i);
            for (int j = 0; j < lab.size(); j++) {
                error += Math.pow(pred.get(j) - lab.get(j), 2);
            }
        }
        return error;
    }

    void crossValidation(Matrix features, Matrix labels, int mRepetitions, int nFold) {

        System.out.println();
        Random random = new Random();

        List<Matrix> featureFolds = new ArrayList<Matrix>();
        for (int i=0; i<nFold; ++i) {
            Matrix m = new Matrix(0, features.cols());
            featureFolds.add(m);
        }
        List<Matrix> labelFolds = new ArrayList<Matrix>();
        for (int i=0; i<nFold; ++i) {
            Matrix m = new Matrix(0, labels.cols());
            labelFolds.add(m);
        }
        int rowsInFold = (int) Math.ceil((float)features.rows()/nFold);

        // shuffling
        for (int i=0; i<features.rows()/2; ++i) {
            int index1 = random.nextInt(features.rows());
            int index2 = random.nextInt(features.rows());
            if (index1 != index2) {
                features.swapRows(index1, index2);
                labels.swapRows(index1, index2);
            }
        }

        // n-fold construction
        for (int foldNo = 0; foldNo<nFold; ++foldNo) {
            for (int i=0; i<rowsInFold; ++i) {
                if (foldNo*rowsInFold+i < features.rows()) {
                    featureFolds.get(foldNo).takeRow(features.row(foldNo*rowsInFold+i).vals);
                    labelFolds.get(foldNo).takeRow(labels.row(foldNo*rowsInFold+i).vals);
                } else {
                    break;
                }
            }
        }

        // m-repetitions
        Set<Integer> repeatedInts = new HashSet<Integer>();
        for (int i=0; i<mRepetitions; ++i) {
            int testBlockIndex = random.nextInt(nFold-1);
            while (repeatedInts.contains(testBlockIndex)) {
                testBlockIndex = random.nextInt(nFold-1);
            }
            repeatedInts.add(testBlockIndex);
            System.out.println();
            System.out.println("Selected test block index = " + testBlockIndex);
            Matrix testFeaturesBlock = featureFolds.get(testBlockIndex);
            Matrix testLabelsBlock = labelFolds.get(testBlockIndex);
            Matrix trainFeaturesBlock = new Matrix(0, features.cols());
            Matrix trainLabelsBlock = new Matrix(0, labels.cols());
            for (int j=0; j<nFold; ++j) {
                if (j != testBlockIndex) {
                    for (int k=0; k<featureFolds.get(j).rows(); ++k) {
                        trainFeaturesBlock.takeRow(featureFolds.get(j).row(k).vals);
                        trainLabelsBlock.takeRow(labelFolds.get(j).row(k).vals);
                    }
                }
            }
            train(trainFeaturesBlock, trainLabelsBlock);
            double error = sumSquaredError(testFeaturesBlock, testLabelsBlock);
            double RMSE = Math.sqrt(error/(labels.rows()*labels.cols()));
            System.out.println("Repitition:" + (i+1) + " RMSE = " + RMSE);
        }
    }
}
