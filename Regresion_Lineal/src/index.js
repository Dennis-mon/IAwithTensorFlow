require('@tensorflow/tfjs-node');
const tf = require('@tensorflow/tfjs');
const loadCSV = require('./loadCSV');
const RegresionLineal = require('./regresion_lineal'); // Cargamos la clase encargada de realizar la regresión lineal

const pathCSV = './src/csv/';

/*const {features, labels, testFeatures, testLabels} = loadCSV(
    `${pathCSV}datos.csv`,
    ',', 
    {
        dataColumns: ['mes','dia'],
        labelColumns: ['value'],
        shuffle: true,
        splitTest: false,
        converters:{}
    }
)

console.log("features: ", features );
console.log("labels: ", labels );
console.log("testFeatures: ", testFeatures );
console.log("testLabels: ", testLabels );*/


const { features , labels, testFeatures, testLabels } = loadCSV(`${pathCSV}cars.csv`, ',', {
    shuffle: true,
    splitTest: 50,
    dataColumns: ['horsepower'],
    labelColumns: ['mpg']
})

const regresionPrueba = new RegresionLineal(features, labels, {
    learningRate: 0.00001,
    iterations: 100
});

console.log("features: ", features );
console.log("labels: ", labels );
console.log("testFeatures: ", testFeatures );
console.log("testLabels: ", testLabels );

regresionPrueba.train();

console.log('Update b => ', regresionPrueba.weights.get(0,0));
console.log('Update m => ', regresionPrueba.weights.get(1,0));