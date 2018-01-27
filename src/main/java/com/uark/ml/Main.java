package com.uark.ml;

// ----------------------------------------------------------------
// The contents of this file are distributed under the CC0 license.
// See http://creativecommons.org/publicdomain/zero/1.0/
// ----------------------------------------------------------------

class Main
{
	static void test(SupervisedLearner learner, String challenge)
	{
        String fn = "C:\\Users\\rakib\\Documents\\GitHub\\ml-project\\target\\classes\\data\\" + challenge;
		Matrix trainFeatures = new Matrix();
		trainFeatures.loadARFF(fn + "_train_feat.arff");
		Matrix trainLabels = new Matrix();
		trainLabels.loadARFF(fn + "_train_lab.arff");

		// Train the model
		learner.train(trainFeatures, trainLabels);

		// Load the test data
		Matrix testFeatures = new Matrix();
		testFeatures.loadARFF(fn + "_test_feat.arff");
		Matrix testLabels = new Matrix();
		testLabels.loadARFF(fn + "_test_lab.arff");

		// Measure and report accuracy
		int misclassifications = learner.countMisclassifications(testFeatures, testLabels);
		System.out.println("Misclassifications by " + learner.name() + " at " + challenge + " = " + Integer.toString(misclassifications) + "/" + Integer.toString(testFeatures.rows()));
	}

	static void test(SupervisedLearner learner) {
        String fn = "C:\\Users\\rakib\\Documents\\GitHub\\ml-project\\target\\classes\\data\\housing_";
        Matrix trainFeatures = new Matrix();
        trainFeatures.loadARFF(fn + "features.arff");
        Matrix trainLabels = new Matrix();
        trainLabels.loadARFF(fn + "labels.arff");

        learner.crossValidation(trainFeatures, trainLabels, 5, 10);
    }

	private static void testLearner(SupervisedLearner learner) {
        System.out.println("\n###Testing Neural Net Classifier###");
        test(learner);
	}

    private static void testActivationFunction() {
        System.out.println("\n###Testing Activation Function###");
        LayerLinear.test_activation_function();
    }

    private static void testOrdinaryLeastSquares() {
        System.out.println("\n###Testing Ordinary Least Squares###");
        try {
            LayerLinear.test_ordinary_least_squares();
        } catch (OrdinaryLeastSquareException e) {
            System.err.println(e.getMessage());
        }
    }

	public static void main(String[] args) {
        testActivationFunction();
        testOrdinaryLeastSquares();
        testLearner(new NeuralNet());
	}
}
