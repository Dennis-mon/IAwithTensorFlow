require('@tensorflow/tfjs-node');
const tf = require('@tensorflow/tfjs');
const loadCSV = require('./loadCSV.js');
const RedNeuronal = require('./red_neuronal.js'); // Cargamos la clase encargada de realizar la regresión lineal
const pathCSV = '../csv/';

//cargamos csv 
const { features , labels, testFeatures, testLabels } = loadCSV(
    `${pathCSV}datos-marzo-horas.csv`,
    ',',
    {
        shuffle: true,
        splitTest: 9,
        dataColumns: ['dia'],
        labelColumns: ['value']
    }
)

const redneuronalPrueba = new RedNeuronal(
    features, 
    labels,
    {
        learningRate: 0.1,
        epochs: 2000,
        neurons: 60,
    }
);

redneuronalPrueba.compilar();
redneuronalPrueba.entrenar();
const resultado = redneuronalPrueba.prediccion(labels);

console.log("Resultados obtenidos:");
resultado.print();
console.log("Resultados reales:");
console.log(labels);

/*
let inputs = tf.tensor(features);
let outputs = tf.tensor(labels);

console.log("inputs:")
inputs.print()
console.log("outputs")
outputs.print()


console.log( inputs.shape[0] )

async function createModel(){
    const model = tf.sequential();

    const hiddenLayer = tf.layers.dense({
        units: 32,
        inputShape: [ inputs.shape[1] ],
        activation: 'relu'
    });

    model.add(hiddenLayer);

    const outputLayer = tf.layers.dense({
        units: 1,
        inputShape: [32],
    });

    model.add(outputLayer);

    model.compile({
        optimizer: tf.train.adam(0.1),
        loss: 'meanSquaredError'
    });

    const configTrain = {
        epochs: 2000,
    }

    const h = await model.fit(inputs, outputs, configTrain);

    console.log(h.history.loss[ h.history.loss.length - 1 ]);

    let prediccion = model.predict( inputs );


    console.log("Resultados obtenidos:");
    prediccion.print();
    console.log("Resultados reales:");
    outputs.print();



}

createModel();*/