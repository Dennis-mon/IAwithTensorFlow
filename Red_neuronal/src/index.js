require('@tensorflow/tfjs-node');
const tf = require('@tensorflow/tfjs');
const loadCSV = require('./loadCSV.js');

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

//let xs = [ [0,0],[0,1],[1,0],[1,1] ];
//let ys = [ [0], [1], [1], [1] ];



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
        units: 10,
        inputShape: [ inputs.shape[1] ],
        activation: 'tanh'
    });

    model.add(hiddenLayer);

    const outputLayer = tf.layers.dense({
        units: 1,
        inputShape: [10],
    });

    model.add(outputLayer);

    model.compile({
        optimizer: tf.train.adam(0.5),
        loss: 'meanSquaredError'
    });

    const configTrain = {
        epochs: 5000,
    }

    const h = await model.fit(inputs, outputs, configTrain);

    console.log(h.history.loss[ h.history.loss.length - 1 ]);

    let prediccion = model.predict( inputs );


    console.log("Resultados obtenidos:");
    prediccion.print();
    console.log("Resultados reales:");
    outputs.print();



}

createModel();