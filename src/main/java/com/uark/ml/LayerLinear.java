package com.uark.ml;

import java.util.Random;

/**
 * Created by rakib on 1/22/2018.
 */
public class LayerLinear extends Layer {
    LayerLinear(int inputs, int outputs) {
        super(inputs, outputs);
    }

    void activate(Vec weights, Vec x) {

        // get b
        Vec b = new Vec(weights, 0, outputs);

        // calculate Mx
        Vec Mx = new Vec(outputs);
        for (int i = 0; i < outputs; ++i) {
            Vec mRows = new Vec(weights, outputs + i * inputs, inputs);
            Mx.set(i, x.dotProduct(mRows));
        }

        // calculate activation=Mx+b
        activation.fill(0);
        activation.add(Mx);
        activation.add(b);
    }

    void ordinary_least_squares(Matrix X, Matrix Y, Vec weights) {
        Vec xMean = new Vec(X.cols());
        for (int i = 0; i < X.cols(); ++i) {
            xMean.set(i, X.columnMean(i));
        }

        Vec yMean = new Vec(Y.cols());
        for (int i = 0; i < Y.cols(); ++i) {
            yMean.set(i, Y.columnMean(i));
        }

        Matrix M1 = new Matrix(Y.cols(), X.cols());
        for (int i = 0; i < X.rows(); ++i) {
            Vec yi = new Vec(Y.cols());
            yi.copy(Y.row(i));
            vectorSubtract(yi, yMean);

            Vec xi = new Vec(X.cols());
            xi.copy(X.row(i));
            vectorSubtract(xi, xMean);
            Matrix tempM = outerProduct(yi, xi);
            matrixAddition(M1, tempM);
        }

        Matrix M2 = new Matrix(X.cols(), X.cols());
        for (int i = 0; i < X.rows(); ++i) {
            Vec xi = X.row(i);
            vectorSubtract(xi, xMean);
            Matrix tempM = outerProduct(xi, xi);
            matrixAddition(M2, tempM);
        }
        M2 = M2.pseudoInverse();

        Matrix M = Matrix.multiply(M1, M2, false, false);

        // calculate b = yMean - M.xMean
        Vec mDotxMean = new Vec(Y.cols());
        for (int i = 0; i < M.rows(); ++i) {
            mDotxMean.set(i, M.row(i).dotProduct(xMean));
        }
        Vec b = new Vec(yMean.size());
        b.copy(yMean);
        vectorSubtract(b, mDotxMean);
        copyValues(weights, b, 0, b.size());
        for (int i = 0; i < M.rows(); ++i) {
            copyValues(weights, M.row(i), b.size() + i * M.cols(), M.cols());
        }
    }

    private void copyValues(Vec x, Vec y, int from, int len) {
        for (int i = from, j = 0; j < len; ++i, ++j) {
            x.set(i, y.get(j));
        }
    }

    private Matrix matrixAddition(Matrix m1, Matrix m2) {
        Matrix result = new Matrix(0, m1.cols());
        for (int i = 0; i < m1.rows(); ++i) {
            Vec mr1 = m1.row(i);
            Vec mr2 = m2.row(i);
            mr1.add(mr2);
            result.takeRow(mr1.vals);
        }
        return result;
    }

    private Matrix outerProduct(Vec x, Vec y) {
        Matrix m = new Matrix(0, y.len);
        for (int i = 0; i < x.len; ++i) {
            Vec temp = new Vec(y.len);
            temp.copy(y);
            temp.scale(x.get(i));
            m.takeRow(temp.vals);
        }
        return m;
    }

    private void vectorSubtract(Vec a, Vec b) {
        Vec temp = new Vec(b.len);
        temp.copy(b);
        temp.scale(-1);
        a.add(temp);
    }

    static void test_activation_function() {
        Vec x = new Vec(new double[]{0, 1, 2});
        Vec weights = new Vec(new double[]{1, 5, 1, 2, 3, 2, 1, 0});
        LayerLinear layerLinear = new LayerLinear(3, 2);
        layerLinear.activate(weights, x);
        System.out.println("Activation = [" + layerLinear.activation.toString() + " ]");
    }

    static void test_ordinary_least_squares() throws OrdinaryLeastSquareException {
        Random random = new Random();

        Matrix X = new Matrix(0, 2);
        X.takeRow(new double[]{random.nextInt(10), random.nextInt(10)});
        X.takeRow(new double[]{random.nextInt(10), random.nextInt(10)});
        X.takeRow(new double[]{random.nextInt(10), random.nextInt(10)});

        Matrix Y = new Matrix(0, 1);
        Matrix yNoise = new Matrix(0, 1);

        Vec weights = new Vec(new double[]{random.nextInt(20), random.nextInt(20), random.nextInt(20)});
        LayerLinear layerLinear = new LayerLinear(2, 1);
        for (int i = 0; i < X.rows(); ++i) {
            Vec x = new Vec(X.cols());
            x.copy(X.row(i));
            layerLinear.activate(weights, x);
            Vec y = new Vec(Y.cols());
            y.copy(layerLinear.activation);
            Y.takeRow(y.vals);

            Vec noise = new Vec(Y.cols());
            noise.copy(layerLinear.activation);
            for (int j = 0; j < noise.size(); ++j) {
                noise.set(j, noise.get(j) + random.nextInt(2));
            }
            yNoise.takeRow(noise.vals);
        }

        Vec originalWeights = new Vec(weights.len);
        originalWeights.copy(weights);
        System.out.println("Old weights: " + originalWeights.toString());

        weights.fill(0);

        layerLinear.ordinary_least_squares(X, yNoise, weights);
        System.out.println("New weights: " + weights.toString());
        double squaredDist = weights.squaredDistance(originalWeights);
        System.out.println("Squared Distance = " + squaredDist);
        if (squaredDist > 10.0) {
            throw new OrdinaryLeastSquareException("Too much distance (Squared Distance=" + squaredDist
                    + ") between original weights and calculated weights.");
        }
    }
}
