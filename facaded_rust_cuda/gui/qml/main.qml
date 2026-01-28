/*
 * MIT License
 * Copyright (c) 2025 Matthew Abbott
 */

import QtQuick 2.15
import QtQuick.Controls 2.15
import QtQuick.Layouts 1.15
import QtQuick.Dialogs 1.3
import com.glassboxai.cnn 1.0

ApplicationWindow {
    id: window
    visible: true
    width: 1200
    height: 900
    title: "Facaded CNN - CUDA"
    color: "#fafafa"

    CNNBridge {
        id: cnnBridge
    }

    ScrollView {
        anchors.fill: parent
        anchors.margins: 20
        contentWidth: availableWidth

        ColumnLayout {
            width: parent.width
            spacing: 20

            // Network Architecture Section
            GroupBox {
                title: "Network Architecture"
                Layout.fillWidth: true
                background: Rectangle { color: "#fff"; radius: 8 }

                ColumnLayout {
                    anchors.fill: parent
                    spacing: 10

                    Rectangle {
                        id: networkCanvas
                        Layout.fillWidth: true
                        Layout.preferredHeight: 150
                        color: "#f5f5f5"
                        radius: 4

                        Label {
                            anchors.centerIn: parent
                            text: "Network architecture visualization placeholder"
                            color: "#999"
                        }

                        function requestPaint() {
                            // Stub for compatibility
                        }
                    }
                }
            }

            // Network Configuration Section
            GroupBox {
                title: "Network Configuration"
                Layout.fillWidth: true
                background: Rectangle { color: "#fff"; radius: 8 }

                ColumnLayout {
                    anchors.fill: parent
                    spacing: 10

                    RowLayout {
                        spacing: 20

                        RowLayout {
                            Label { text: "Image Width:" }
                            SpinBox { id: imgWidth; from: 1; to: 256; value: 28 }
                        }
                        RowLayout {
                            Label { text: "Image Height:" }
                            SpinBox { id: imgHeight; from: 1; to: 256; value: 28 }
                        }
                        RowLayout {
                            Label { text: "Channels:" }
                            SpinBox { id: imgChannels; from: 1; to: 3; value: 1 }
                        }
                    }

                    RowLayout {
                        spacing: 20

                        RowLayout {
                            Label { text: "Conv Filters:" }
                            TextField { id: convFilters; text: "8,16"; implicitWidth: 100 }
                        }
                        RowLayout {
                            Label { text: "Kernel Sizes:" }
                            TextField { id: kernelSizes; text: "3,3"; implicitWidth: 100 }
                        }
                        RowLayout {
                            Label { text: "Pool Sizes:" }
                            TextField { id: poolSizes; text: "2,2"; implicitWidth: 100 }
                        }
                        RowLayout {
                            Label { text: "FC Sizes:" }
                            TextField { id: fcSizes; text: "64"; implicitWidth: 100 }
                        }
                        RowLayout {
                            Label { text: "Output Classes:" }
                            SpinBox { id: outputSize; from: 1; to: 1000; value: 10 }
                        }
                    }

                    RowLayout {
                        spacing: 20

                        RowLayout {
                            Label { text: "Learning Rate:" }
                            TextField { id: learningRate; text: "0.001"; implicitWidth: 80 }
                        }
                        RowLayout {
                            Label { text: "Dropout Rate:" }
                            TextField { id: dropoutRate; text: "0.25"; implicitWidth: 80 }
                        }
                        RowLayout {
                            Label { text: "Optimizer:" }
                            ComboBox {
                                id: optimizer
                                model: ["SGD", "Adam"]
                                currentIndex: 1
                            }
                        }
                    }

                    RowLayout {
                        spacing: 10

                        Button {
                            text: "Create Network"
                            onClicked: {
                                var result = cnnBridge.createNetwork(
                                    imgWidth.value, imgHeight.value, imgChannels.value,
                                    convFilters.text, kernelSizes.text, poolSizes.text,
                                    fcSizes.text, outputSize.value,
                                    parseFloat(learningRate.text)
                                );
                                networkStatus.text = result;
                                networkCanvas.requestPaint();
                            }
                        }

                        Button {
                            text: "Load Model (JSON)"
                            onClicked: loadDialog.open()
                        }

                        Button {
                            text: "Save Model (JSON)"
                            onClicked: saveDialog.open()
                        }

                        Label {
                            id: networkStatus
                            text: ""
                            color: "#666"
                        }
                    }
                }
            }

            // Dataset Section
            GroupBox {
                title: "Dataset"
                Layout.fillWidth: true
                background: Rectangle { color: "#fff"; radius: 8 }

                ColumnLayout {
                    anchors.fill: parent
                    spacing: 10

                    RowLayout {
                        spacing: 10

                        Button {
                            text: "Generate Synthetic Dataset"
                            onClicked: {
                                var result = cnnBridge.generateSyntheticDataset(samplesPerClass.value);
                                datasetInfo.text = result;
                            }
                        }

                        Label { text: "Samples/class:" }
                        SpinBox { id: samplesPerClass; from: 10; to: 1000; value: 50 }
                    }

                    Label {
                        id: datasetInfo
                        text: ""
                        color: "#666"
                    }

                    RowLayout {
                        spacing: 20

                        ColumnLayout {
                            Label { text: "Dataset Preview"; font.bold: true }
                            Rectangle {
                                id: datasetPreview
                                width: 200
                                height: 200
                                color: "#eee"
                                radius: 4
                                Label {
                                    anchors.centerIn: parent
                                    text: "Dataset preview"
                                    color: "#999"
                                }
                            }
                        }

                        ColumnLayout {
                            Label { text: "Input Image"; font.bold: true }
                            Rectangle {
                                id: inputCanvas
                                width: 140
                                height: 140
                                color: "#eee"
                                radius: 4
                                Label {
                                    anchors.centerIn: parent
                                    text: "Input"
                                    color: "#999"
                                }
                                function requestPaint() {
                                    // Stub for compatibility
                                }
                            }
                        }
                    }
                }
            }

            // Training Section
            GroupBox {
                title: "Training"
                Layout.fillWidth: true
                background: Rectangle { color: "#fff"; radius: 8 }

                ColumnLayout {
                    anchors.fill: parent
                    spacing: 10

                    RowLayout {
                        spacing: 20

                        RowLayout {
                            Label { text: "Epochs:" }
                            SpinBox { id: trainEpochs; from: 1; to: 1000; value: 10 }
                        }
                        RowLayout {
                            Label { text: "Batch Size:" }
                            SpinBox { id: batchSize; from: 1; to: 256; value: 16 }
                        }

                        Button {
                            id: trainBtn
                            text: "Train"
                            onClicked: {
                                trainBtn.enabled = false;
                                stopBtn.enabled = true;
                                cnnBridge.trainOnDataset(trainEpochs.value, batchSize.value);
                            }
                        }

                        Button {
                            id: stopBtn
                            text: "Stop"
                            enabled: false
                            onClicked: {
                                cnnBridge.stopTraining();
                                trainBtn.enabled = true;
                                stopBtn.enabled = false;
                            }
                        }
                    }

                    ProgressBar {
                        id: trainProgress
                        Layout.fillWidth: true
                        from: 0
                        to: 100
                        value: cnnBridge.trainingProgress
                    }

                    Label {
                        id: trainStatus
                        text: cnnBridge.trainingStatus
                        color: "#666"
                    }

                    ScrollView {
                        Layout.fillWidth: true
                        Layout.preferredHeight: 150

                        TextArea {
                            id: trainingLog
                            text: cnnBridge.trainingLog
                            readOnly: true
                            font.family: "monospace"
                            font.pixelSize: 12
                            color: "#0f0"
                            background: Rectangle { color: "#333"; radius: 4 }
                        }
                    }
                }
            }

            // Evaluation Section
            GroupBox {
                title: "Evaluation"
                Layout.fillWidth: true
                background: Rectangle { color: "#fff"; radius: 8 }

                ColumnLayout {
                    anchors.fill: parent
                    spacing: 10

                    RowLayout {
                        spacing: 10

                        Button {
                            text: "Evaluate"
                            onClicked: {
                                var result = cnnBridge.evaluateModel();
                                predictionResult.text = result;
                            }
                        }

                        Button {
                            text: "Predict Current Image"
                            onClicked: {
                                var result = cnnBridge.predictCurrentImage();
                                predictionResult.text = result;
                            }
                        }

                        Button {
                            text: "Randomize Image"
                            onClicked: {
                                cnnBridge.randomizeImage();
                                inputCanvas.requestPaint();
                            }
                        }
                    }

                    Label {
                        id: metricsLabel
                        text: cnnBridge.metricsText
                        font.pixelSize: 14
                        wrapMode: Text.WordWrap
                        Layout.fillWidth: true
                    }

                    Label {
                        id: predictionResult
                        text: ""
                        wrapMode: Text.WordWrap
                        Layout.fillWidth: true
                    }
                }
            }

            // Facade API Explorer Section
            GroupBox {
                title: "CNN Facade API Explorer"
                Layout.fillWidth: true
                background: Rectangle { color: "#e8f4f8"; radius: 8 }

                ColumnLayout {
                    anchors.fill: parent
                    spacing: 10

                    Label {
                        text: "Inspect and modify CNN internals:"
                        color: "#666"
                    }

                    GridLayout {
                        columns: 8
                        Layout.fillWidth: true
                        columnSpacing: 10
                        rowSpacing: 5

                        Label { text: "Layer Idx:" }
                        SpinBox { id: facadeLayerIdx; from: 0; to: 100; value: 0; implicitWidth: 70 }

                        Label { text: "Filter Idx:" }
                        SpinBox { id: facadeFilterIdx; from: 0; to: 100; value: 0; implicitWidth: 70 }

                        Label { text: "Channel Idx:" }
                        SpinBox { id: facadeChannelIdx; from: 0; to: 100; value: 0; implicitWidth: 70 }

                        Label { text: "Y Position:" }
                        SpinBox { id: facadeY; from: 0; to: 1000; value: 0; implicitWidth: 70 }

                        Label { text: "X Position:" }
                        SpinBox { id: facadeX; from: 0; to: 1000; value: 0; implicitWidth: 70 }

                        Label { text: "Value:" }
                        TextField { id: facadeValue; text: "0"; implicitWidth: 60 }

                        Label { text: "Num Bins:" }
                        SpinBox { id: facadeNumBins; from: 5; to: 100; value: 20; implicitWidth: 70 }

                        Label { text: "Attr Key:" }
                        TextField { id: facadeAttrKey; text: "tag"; implicitWidth: 60 }
                    }

                    // Feature Map Access
                    Label { text: "1. Feature Map Access"; font.bold: true }
                    RowLayout {
                        Button { text: "Get Feature Map"; onClicked: facadeOutput.text = cnnBridge.getFeatureMap(facadeLayerIdx.value, facadeFilterIdx.value) }
                        Button { text: "Set Feature Map (zeros)"; onClicked: facadeOutput.text = cnnBridge.setFeatureMapZeros(facadeLayerIdx.value, facadeFilterIdx.value) }
                        Button { text: "Visualize Feature Map"; onClicked: facadeOutput.text = cnnBridge.visualizeFeatureMap(facadeLayerIdx.value, facadeFilterIdx.value) }
                    }

                    // Pre-Activation Access
                    Label { text: "2. Pre-Activation Access"; font.bold: true }
                    RowLayout {
                        Button { text: "Get Pre-Activation"; onClicked: facadeOutput.text = cnnBridge.getPreActivation(facadeLayerIdx.value, facadeFilterIdx.value) }
                        Button { text: "Set Pre-Activation"; onClicked: facadeOutput.text = cnnBridge.setPreActivation(facadeLayerIdx.value, facadeFilterIdx.value, parseFloat(facadeValue.text)) }
                    }

                    // Kernel/Filter Access
                    Label { text: "3. Kernel/Filter Access"; font.bold: true }
                    RowLayout {
                        Button { text: "Get Kernel"; onClicked: facadeOutput.text = cnnBridge.getKernel(facadeLayerIdx.value, facadeFilterIdx.value, facadeChannelIdx.value) }
                        Button { text: "Set Kernel (random)"; onClicked: facadeOutput.text = cnnBridge.setKernelRandom(facadeLayerIdx.value, facadeFilterIdx.value, facadeChannelIdx.value) }
                        Button { text: "Get Bias"; onClicked: facadeOutput.text = cnnBridge.getBias(facadeLayerIdx.value, facadeFilterIdx.value) }
                        Button { text: "Set Bias"; onClicked: facadeOutput.text = cnnBridge.setBias(facadeLayerIdx.value, facadeFilterIdx.value, parseFloat(facadeValue.text)) }
                        Button { text: "Visualize All Kernels"; onClicked: facadeOutput.text = cnnBridge.visualizeAllKernels(facadeLayerIdx.value) }
                    }

                    // Pooling & Dropout
                    Label { text: "5. Pooling & Dropout"; font.bold: true }
                    RowLayout {
                        Button { text: "Get Pooling Indices"; onClicked: facadeOutput.text = cnnBridge.getPoolingIndices(facadeLayerIdx.value) }
                        Button { text: "Get Dropout Mask"; onClicked: facadeOutput.text = cnnBridge.getDropoutMask(facadeLayerIdx.value) }
                        Button { text: "Set Dropout Mask (all 1s)"; onClicked: facadeOutput.text = cnnBridge.setDropoutMaskOnes(facadeLayerIdx.value) }
                    }

                    // Gradients & Optimizer
                    Label { text: "6. Gradients & Optimizer"; font.bold: true }
                    RowLayout {
                        Button { text: "Get Filter Gradient"; onClicked: facadeOutput.text = cnnBridge.getFilterGradient(facadeLayerIdx.value, facadeFilterIdx.value) }
                        Button { text: "Get Bias Gradient"; onClicked: facadeOutput.text = cnnBridge.getBiasGradient(facadeLayerIdx.value, facadeFilterIdx.value) }
                        Button { text: "Get Optimizer State"; onClicked: facadeOutput.text = cnnBridge.getOptimizerState() }
                    }

                    // Flattened Features
                    Label { text: "7. Flattened Features"; font.bold: true }
                    RowLayout {
                        Button { text: "Get Flattened Features"; onClicked: facadeOutput.text = cnnBridge.getFlattenedFeatures() }
                    }

                    // Output Layer
                    Label { text: "8. Output Layer"; font.bold: true }
                    RowLayout {
                        Button { text: "Get Logits"; onClicked: facadeOutput.text = cnnBridge.getLogits() }
                        Button { text: "Get Softmax"; onClicked: facadeOutput.text = cnnBridge.getSoftmax() }
                    }

                    // Layer Config
                    Label { text: "9. Layer Config"; font.bold: true }
                    RowLayout {
                        Button { text: "Get Layer Config"; onClicked: facadeOutput.text = cnnBridge.getLayerConfig(facadeLayerIdx.value) }
                        Button { text: "Get Num Layers"; onClicked: facadeOutput.text = cnnBridge.getNumLayers() }
                        Button { text: "Get Num Conv Layers"; onClicked: facadeOutput.text = cnnBridge.getNumConvLayers() }
                        Button { text: "Get Num FC Layers"; onClicked: facadeOutput.text = cnnBridge.getNumFCLayers() }
                        Button { text: "Get Num Filters"; onClicked: facadeOutput.text = cnnBridge.getNumFilters(facadeLayerIdx.value) }
                    }

                    // Statistics
                    Label { text: "12. Statistics"; font.bold: true }
                    RowLayout {
                        Button { text: "Get Layer Stats"; onClicked: facadeOutput.text = cnnBridge.getLayerStats(facadeLayerIdx.value) }
                        Button { text: "Activation Histogram"; onClicked: facadeOutput.text = cnnBridge.getActivationHistogram(facadeLayerIdx.value, facadeNumBins.value) }
                        Button { text: "Weight Histogram"; onClicked: facadeOutput.text = cnnBridge.getWeightHistogram(facadeLayerIdx.value, facadeNumBins.value) }
                    }

                    // Receptive Field
                    Label { text: "13. Receptive Field"; font.bold: true }
                    RowLayout {
                        Button { text: "Get Receptive Field"; onClicked: facadeOutput.text = cnnBridge.getReceptiveField(facadeLayerIdx.value, facadeFilterIdx.value, facadeY.value, facadeX.value) }
                    }

                    // Output
                    Label { text: "Output"; font.bold: true }
                    ScrollView {
                        Layout.fillWidth: true
                        Layout.preferredHeight: 200

                        TextArea {
                            id: facadeOutput
                            text: "Facade output will appear here..."
                            readOnly: true
                            font.family: "monospace"
                            font.pixelSize: 12
                            wrapMode: TextEdit.Wrap
                            background: Rectangle { color: "#fff"; border.color: "#ccc"; radius: 4 }
                        }
                    }
                }
            }
        }
    }

    FileDialog {
        id: loadDialog
        title: "Load Model"
        nameFilters: ["JSON files (*.json)"]
        selectExisting: true
        onAccepted: {
            var result = cnnBridge.loadModel(fileUrl);
            networkStatus.text = result;
            networkCanvas.requestPaint();
        }
    }

    FileDialog {
        id: saveDialog
        title: "Save Model"
        selectExisting: false
        nameFilters: ["JSON files (*.json)"]
        onAccepted: {
            var result = cnnBridge.saveModel(fileUrl);
            networkStatus.text = result;
        }
    }

    Connections {
        target: cnnBridge
        function onTrainingFinished() {
            trainBtn.enabled = true;
            stopBtn.enabled = false;
        }
        function onNetworkChanged() {
            networkCanvas.requestPaint();
        }
    }
}
