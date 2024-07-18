require('@tensorflow/tfjs-node');
const tf = require('@tensorflow/tfjs');
const loadCSV = require('./loadCSV');
const { random } = require('lodash');
const pathCSV = './src/csv/';


const { features , labels, testFeatures, testLabels } = loadCSV(
    `${pathCSV}data-marzo-planta-fotovoltaica.csv`,
    ';',
    {
        shuffle: true,
        splitTest: 22,
        dataColumns: ['dia'],
        labelColumns: ['Valor']
    }
)

let x_vals = tf.tensor(features);
let y_vals = tf.tensor(labels);

let w = tf.variable(tf.scalar(random()));;
let b = tf.variable(tf.scalar(random()));;

const learningRate = 0.2;

function loss(predictions, auxLabels){
    const error = predictions.sub(auxLabels).square().mean();
    return error;
}

function predict(x,w,b){
    const y_hat = w.mul(x).add(b);
    return y_hat;
}

function train(x,y,w,b){

    const optimizer = tf.train.sgd(learningRate);

    optimizer.minimize( function(){
        const pred = predict( x, w,b );
        const stepLoss = loss(pred, y);
        return stepLoss;
    });
}

for( let i = 0; i < 2000; i++ ){
    train(x_vals, y_vals, w, b);
}

w.print();
b.print();

predict( tf.tensor([18]),w,b ).print();