require('@tensorflow/tfjs-node');       //  Tensorflow
const tf = require('@tensorflow/tfjs'); //  Tensorflow
const loadCSV = require('./load-csv');  //  Cargamos el archivo para leer CSVs
const RegresionLineal = require('./regresion_lineal'); // Cargamos la clase encargada de realizar la regresión lineal

const { features , labels, testFeatures, testLabels } = loadCSV('./cars.csv', {
    shuffle: true,
    splitTest: 50,
    dataColumns: ['horsepower'],
    labelColumns: ['mpg']
})

const regresionPrueba = new RegresionLineal(features, labels, {
    learningRate: 0.0001,
    iterations: 100
});

regresionPrueba.train();
console.log('Update M => ', regresionPrueba.m);
console.log('Update B => ', regresionPrueba.b);