# High Piezoelectric Material Classification using GCNs

## Problem Statement

Finding materials with high piezoelectric properties is computationally expensive, with some molecules taking up to 8 hours to evaluate on our local machine. Piezoelectric materials, which convert mechanical stress into electrical charge (or vice versa), are critical for a variety of applications, but identifying compounds with high piezoelectric coefficients (both piezo stress and piezo charge) is a challenging task due to the high dimensionality of molecular data and the scarcity of known high-performing materials.

## Project Goal

This project aims to train Graph Convolutional Networks (GCNs) to classify materials based on their piezoelectric properties, specifically the piezoelectric stress coefficient (PSC) and piezoelectric charge coefficient (PCC). By leveraging GCNs, we aim to efficiently predict materials with high piezoelectric constants, reducing the need for intensive computational resources.

## Approach

We used two different fully connected graph representations for the molecules:
- **Weighted Graphs**: Each molecule was encoded with atom-specific properties as node features and bond information as edge weights.
- **Unweighted Graphs**: A simpler graph representation where the nodes represent atoms and edges represent connectivity without weights.

For both graph representations, we trained two sets of models:
1. **PSC Model Set**: Classifying materials based on their piezoelectric stress coefficient.
2. **PCC Model Set**: Classifying materials based on their piezoelectric charge coefficient.

For each set, we used different thresholds for binary classification, which separates the top portion of the training data from the rest. The thresholds used were:
- 10%
- 15%
- 20%
- 25%
- 50%

## Results

The models were trained and evaluated on both weighted and unweighted graph representations. By adjusting the classification thresholds, we provided different models that predict high piezoelectric materials under varying criteria. See the overall analysis sheets for detailed results, but over 500 epochs our model can featurize and classify a molecule in 30ms with a Reciever Operating Chararacteristic Area Under Curve above 0.95 in almost every model instance. These results are then verified by Density Functional Theory calculations, and using this method we have already identified many viable piezoelectric molecules.
