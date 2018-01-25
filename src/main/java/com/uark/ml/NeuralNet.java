package com.uark.ml;

import java.util.ArrayList;
import java.util.List;

/**
 * Created by rakib on 1/24/2018.
 */
public class NeuralNet extends SupervisedLearner {

    List<Layer> layers = new ArrayList<Layer>();
    Vec weights;

    String name() {
        return "NN";
    }

    void train(Matrix features, Matrix labels) {

        // FIXME: more layers on next assignment
        layers.clear();

        LayerLinear layerLinear = new LayerLinear(features.cols(), labels.cols());
        layers.add(layerLinear);
        this.weights = new Vec(labels.cols() + labels.cols() * features.cols());
        layerLinear.ordinary_least_squares(features, labels, weights);
    }

    Vec predict(Vec in) {
        layers.get(0).activate(weights, in);
        return layers.get(0).activation;
    }
}
