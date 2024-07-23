require('@tensorflow/tfjs-node');
const tf = require('@tensorflow/tfjs');
const loadCSV = require('./loadCSV.js');


let xs = [ [0,0],[0,1],[1,0],[1,1] ];
let ys = [ [0], [1], [1], [1] ];

let inputs = tf.tensor2d(xs);
let outputs = tf.tensor2d(ys);

console.log( inputs.shape[0] )

async function createModel(){
    const model = tf.sequential();

    const hiddenLayer = tf.layers.dense({
        units: 10,
        inputShape: [ 2 ],
        activation: 'tanh'
    });

    model.add(hiddenLayer);

    const outputLayer = tf.layers.dense({
        units: 1,
        inputShape: [10],
        activation: 'tanh'
    });

    model.add(outputLayer);

    model.compile({
        optimizer: 'sgd',
        loss: 'meanSquaredError'
    });

    const configTrain = {
        epochs: 1000
    }

    const h = await model.fit(inputs, outputs, configTrain);

    console.log(h.history.loss[ h.history.loss.length - 1 ]);

    let prediccion = model.predict( tf.tensor2d(xs) );


    console.log("Resultados obtenidos:");
    prediccion.print();
    console.log("Resultados reales:");
    outputs.print();


}

createModel();