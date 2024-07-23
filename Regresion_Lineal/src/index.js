require('@tensorflow/tfjs-node');
const tf = require('@tensorflow/tfjs');
const loadCSV = require('./loadCSV.js');
const RegresionLineal = require('./regresion_lineal'); // Cargamos la clase encargada de realizar la regresión lineal

const pathCSV = '../csv/';

//cargamos csv 
const { features , labels, testFeatures, testLabels } = loadCSV(
    `${pathCSV}datos-marzo-horas.csv`,
    ',',
    {
        shuffle: true,
        splitTest: 9,
        dataColumns: ['dia', 'mes', '0-24'],
        labelColumns: ['value']
    }
)

//iniciamos regresion lineal parametrizada
const regresionPrueba = new RegresionLineal(
    features, 
    labels,
    {
        learningRate: 10,
        iterations: 100
    }
);


//entrenamos la ia con los valores pasados
regresionPrueba.train();

//realizamos un test
console.log('==========================');
const r = regresionPrueba.test(testFeatures, testLabels);
console.log('Eficacia del algoritmo');
console.log('R => ', r);

//mostramos valores de B y las  distintas M
regresionPrueba.mostrarPesos();

//hacemos una prueba
console.log('Prueba');
console.log('Datos =>' , features[1]);
console.log('Resultado Real =>' , labels[1]);
const resultado = regresionPrueba.predictResult(features[1]);
console.log('Resultado Predicción =>' ,resultado);