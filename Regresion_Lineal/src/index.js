require('@tensorflow/tfjs-node');
const tf = require('@tensorflow/tfjs');
const loadCSV = require('./loadCSV');
const RegresionLineal = require('./regresion_lineal'); // Cargamos la clase encargada de realizar la regresión lineal

const pathCSV = './src/csv/';

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
const r = regresionPrueba.test(testFeatures, testLabels);

//mostramos valores de B y las  distintas M
console.log('Valor de B =>', regresionPrueba.weights.get(0,0));
console.log('Valor de M1 =>', regresionPrueba.weights.get(1,0));
console.log('Valor de M2 =>', regresionPrueba.weights.get(2,0));
console.log('Valor de M3 =>', regresionPrueba.weights.get(3,0));
console.log('R => ', r);

//hacemos una prueba
console.log('==========================')
console.log('Prueba');
console.log('Datos =>' , features[1]);
console.log('Resultado =>' , labels[1]);
const resultado = regresionPrueba.predictResult(features[1]);
console.log(resultado);