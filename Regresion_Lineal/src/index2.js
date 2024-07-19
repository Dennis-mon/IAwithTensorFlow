require('@tensorflow/tfjs-node');
const tf = require('@tensorflow/tfjs');
const loadCSV = require('./loadCSV');
const pathCSV = './src/csv/';


const { features , labels, testFeatures, testLabels } = loadCSV(
    `${pathCSV}datos-marzo-horas.csv`,
    ',',
    {
        shuffle: true,
        splitTest: 22,
        dataColumns: ['dia'],
        labelColumns: ['value']
    }
)

console.log("features:", features)
console.log("labels:", labels)

let x_vals = tf.tensor(features);
let y_vals = tf.tensor(labels);

let w = tf.variable(tf.scalar(0));
let b = tf.variable(tf.scalar(0));

let learningRate = 0.002;

x_vals.print()
y_vals.print()
w.print()
b.print()

function loss(predictions, auxLabels){
    let error = predictions.sub(auxLabels).square().mean();
    return error;
}

function predict(x,w,b){
    let y_hat = w.mul(x).add(b);
    return y_hat;
}

function train(x,y,w,b,i){

    let optimizer = tf.train.sgd(learningRate);

    optimizer.minimize( function(){
        let pred = predict( x, w,b );
        let stepLoss = loss(pred, y);

        console.log("============")
        console.log(i);
        w.print()
        b.print()
        console.log("============")

        return stepLoss;
    });
}

for( let i = 0; i < 1000; i++ ){
    train(x_vals, y_vals, w, b,i);
}

console.log("w:" )
w.print()
console.log("b:" )
b.print();

//prueba
console.log('Prueba');
console.log('Datos =>' , features[1]);
console.log('Resultado Real =>' , labels[1]);
const resultado = predict( tf.tensor( features[1] ),w,b );
console.log('Resultado Predicción =>' ,resultado.sum().get() );

//Error cometido
console.log("Error: ", labels[1] - resultado.sum().get() );

let porcentaje = labels[1] / resultado.sum().get()

if( porcentaje > 1 ) {
    let error = porcentaje - 1;
    porcentaje = 1 - error;
}

console.log("% cercania: ", porcentaje)
